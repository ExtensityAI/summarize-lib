import asyncio
import functools
import os
import re
import tempfile
import urllib.request
from textwrap import dedent
from typing import List, Optional

from chonkie import BaseChunker, BaseEmbeddings
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
import nest_asyncio
from loguru import logger
from pydantic import Field, field_validator
import tiktoken
from tiktoken import Encoding
from tokenizers import Tokenizer
from chonkie import RecursiveChunker
from symai.components import FileReader, Function
from symai.core_ext import bind
from symai.models import LLMDataModel

from .functions import ValidatedFunction
from .types import TYPE_SPECIFIC_PROMPTS, DocumentType


class Summary(LLMDataModel):
    summary: str = Field(
        description="An extremely comprehensive summary of the document. Do not start with 'This document is about...' or similar phrases."
    )
    facts: List[str] = Field(
        description="Important facts and subjects extracted from the document."
    )
    quotes: Optional[List[str]] = Field(
        default=None,
        description="Significant quotes extracted from the document **verbatim** if there are any.",
    )
    type: Optional[str] = None

    def validate():
        # TODO: validate that quotes are verbatim from the document
        pass


def gather(chunks: List[LLMDataModel]):
    res_dict = {}
    type_dict = {
        list: {"default": list, "func": "append"},
        str: {"default": str, "func": "concatenate"},
    }
    for chunk in chunks:
        chunk_fields = chunk.model_fields
        for field_name, field_type in chunk_fields.items():
            field = getattr(chunk, field_name)
            if type(field) in type_dict and not field_type.exclude:
                _type = type_dict[type(field)]
                # setup field
                if field_name not in res_dict:
                    res_dict[field_name] = type(field)()

                # append or concatenate
                if _type["func"] == "append":
                    res_dict[field_name].extend(field)
                elif _type["func"] == "concatenate":
                    res_dict[field_name] += field + "\n"

    return res_dict


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


class HierarchicalSummary(ValidatedFunction):
    # Define the prompt types as class variables
    def __init__(
        self,
        file_link: str = None,
        content: str = None,
        document_name: str = None,
        asset_name: str = None,
        data_model: LLMDataModel = Summary,
        min_num_chunks: int = 5,
        min_chunk_size: int = 250,
        max_chunk_size: int = 1000,
        max_output_tokens: int = 10000,
        user_prompt: str = None,
        include_quotes: bool = False,
        tokenizer: str | BaseEmbeddings | Encoding = "gpt2",
        chunker: BaseChunker = RecursiveChunker,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        # only allow file_link or content
        assert (file_link and not content) or (content and not file_link)

        if document_name is None and asset_name is not None:
            document_name = asset_name

        if content is not None:
            assert document_name is not None

        assert issubclass(data_model, LLMDataModel)

        super().__init__(data_model=data_model, retry_count=5, *args, **kwargs)
        self.file_link = file_link
        self.min_num_chunks = min_num_chunks
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_output_tokens = max_output_tokens
        self.user_prompt = user_prompt
        self.include_quotes = include_quotes
        self.seed = seed

        file_content = None
        file_name = None
        if file_link is not None:
            if file_link.startswith("http"):
                file_content, file_name = self.download_file(file_link)
            else:
                file_content, file_name = self.read_file(file_link)
        else:
            file_name = document_name
            file_content = str(content)
        self.content = f"[[DOCUMENT::{file_name}]]: <<<\n{str(file_content)}\n>>>\n"
        self.content_only = str(file_content)

        # init tokenizer
        if isinstance(tokenizer, str):
            try:
                self.tokenizer = tiktoken.encoding_for_model(tokenizer)
            except:
                try:
                    self.tokenizer = Tokenizer.from_pretrained(tokenizer)
                except:
                    raise ValueError("Invalid tokenizer or model name")
        else:
            self.tokenizer = tokenizer
        self.chunker = chunker

        # Content type is unknown at initialization
        self.document_type = None

    def read_file(self, file_link: str):
        logger.info(f"Reading file from {file_link}")
        reader = FileReader()
        content = reader(file_link)
        file_name = os.path.basename(file_link)
        val = f"[[DOCUMENT::{file_name}]]: <<<\n{str(content)}\n>>>\n"
        return val, file_name

    def download_file(self, file_link: str):
        logger.info(f"Downloading file from {file_link}")

        with urllib.request.urlopen(file_link) as f:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(f.read())
                tmp_file.flush()
                tmp_file_name = tmp_file.name

        content, file_name = self.read_file(tmp_file_name)
        os.remove(tmp_file_name)
        return content, file_name

    @property
    def prompt(self):
        # Get type-specific prompt
        type_specific_prompt = ""
        if self.document_type and self.document_type in TYPE_SPECIFIC_PROMPTS:
            type_specific_prompt = dedent(
                f"""[Type-Specific Instructions]
            For this {self.document_type.value}: {TYPE_SPECIFIC_PROMPTS[self.document_type]}"""
            )

        if self.user_prompt is not None:
            user_prompt = dedent(
                f"""[Goal-specific Instructions]
            This summary is intended for a specific audience or purpose, which is defined in <purpose> below.
            **In addition to the general summary and list of facts, ensure that if information relevant to the information below is present in the text, it is included in the summary and additional fields present in the JSON format.**
                                    
            <purpose>
            {self.user_prompt}
            </purpose>"""
            )

        prompt_text = dedent(
            f"""
            [[Document Processing Task]]

            [Main Objective]
            Create an extremely comprehensive summary of the provided content and return the result as JSON.
            The document is split up into chunks, and each chunk is summarized separately. 
            The final summary is the concatenation of all chunk summaries.
            The type of the provided content is specified in [[CONTENT TYPE]].
            In addition to the summary extract additional information as specified in the JSON format.
                     
            {type_specific_prompt}
                     
            {user_prompt if self.user_prompt is not None else ""}
                        
            [Language Requirements]
            The summary must be in the language specified in [[CONTENT LANGUAGE]], regardless of the source material.

            [Key Requirements]
            - Summarize the content, ensuring that all relevant points are captured.
            - Do not start the summary with phrases like 'This document is about...' or similar.
            - Make the summary as comprehensive as necessary to cover all key points.
            - Extract all additional information as specified in the JSON format.
            - **Important**: Ensure that the summary is consistent with the additional information extracted.
        """
        )
        prompt_text += self.data_model.instruct_llm()
        return prompt_text

    @property
    def static_context(self):
        return dedent(
            """
            Create a comprehensive summary of the provided text and extract important facts.
            The summary must be in the same language as the text.
            Return the summary in JSON format with the provided JSON schema.
        """
        )

    @bind(engine="neurosymbolic", property="compute_required_tokens")(lambda: 0)
    def _compute_required_tokens(self):
        pass

    @bind(engine="neurosymbolic", property="max_context_tokens")
    def _max_context_tokens(_):
        pass

    @bind(engine="neurosymbolic", property="max_response_tokens")
    def _max_response_tokens(_):
        pass

    @bind(engine="neurosymbolic", property="compute_remaining_tokens")(lambda: 0)
    def _compute_remaining_tokens(self):
        pass

    def compute_required_tokens(self, data, count_context=True):
        # construct preview function
        if count_context:
            preview_function = Function(
                prompt=self.prompt,
                static_context=self.static_context,
                dynamic_context=self.dynamic_context,
            )
        else:
            preview_function = Function()

        # execute preview
        preview = preview_function(
            data,
            preview=True,
            response_format={"type": "json_object"},
            seed=self.seed,
        )

        # count prompt tokens
        return self._compute_required_tokens(preview.prop.prepared_input)

    def split_words(self, text):
        return re.split(r"(\W+)", text)

    def chunk_by_token_count(self, text, chunk_size, include_context=False):
        # prepare results
        logger.debug(f"Chunking with chunk size: {chunk_size}")
        chunks = self.chunker(self.tokenizer, chunk_size=chunk_size)(text)
        logger.debug(f"Number of chunks: {len(chunks)}")
        return chunks

    async def summarize_chunks(self, chunks, **kwargs):
        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential_jitter(initial=0.25, max=60),
            stop=stop_after_attempt(10),
            before_sleep=before_sleep_log(logger, logger.level("DEBUG").no),
        )
        async def summarize_chunk(chunk):
            loop = asyncio.get_event_loop()
            forward_fn = functools.partial(
                super(HierarchicalSummary, self).forward,
                chunk,
                preview=False,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return await loop.run_in_executor(None, forward_fn)

        tasks = [summarize_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        final_res = self.data_model(**gather(results))

        return final_res, self.compute_required_tokens(final_res, count_context=False)

    def calculate_chunk_size(self, total_tokens):
        num_prompt_tokens = self.compute_required_tokens("", count_context=True)
        max_tokens_per_chunk = int(
            self._max_context_tokens() - num_prompt_tokens * 0.8
        )  # leave some headroom
        chunk_size = total_tokens // self.min_num_chunks - num_prompt_tokens

        if chunk_size > self.min_chunk_size:
            num_chunks = self.min_num_chunks
            while (chunk_size > max_tokens_per_chunk) or (
                chunk_size > self.max_chunk_size
            ):
                num_chunks += 1
                chunk_size = total_tokens // num_chunks - num_prompt_tokens

            return max(self.min_chunk_size, chunk_size)
        else:
            return self.min_chunk_size

    def get_document_type(self, content):
        # Prepare a list of all values in the enum DocumentType
        allowed_types = [doc_type.value for doc_type in DocumentType]

        class ContentType(LLMDataModel):
            type: str

            @field_validator("type")
            def validate_type(cls, v):
                assert (
                    v in allowed_types
                ), f"Type must be one of: {', '.join(sorted(allowed_types))}"
                return v

        # construct function to determine document type
        doc_type_func = ValidatedFunction(
            data_model=ContentType,
            retry_count=self.retry_count,
            prompt=(
                "What type of content is this text?\n"
                + f"Allowed types: {', '.join(sorted(allowed_types))}\n"
                + "The content type must be mapped exactly/literally to one of the listed types. No other type allowed!\n\n"
            ),
            static_context=r"Return JSON: {'type': string}",
        )

        res = doc_type_func(
            content,
            preview=False,
            response_format={"type": "json_object"},
            seed=self.seed,
        )

        # Store the content type for use in prompt

        self.document_type = DocumentType(res.type)

        return self.document_type

    def get_document_language(self, content):
        class ContentLanguage(LLMDataModel):
            language: str

        # construct function to determine document language, use ValidatedFunction to restrict to allowed types
        doc_lang_func = ValidatedFunction(
            data_model=ContentLanguage,
            retry_count=self.retry_count,
            prompt=dedent(
                """Which language is this document in?
            - Follow the ISO 639 standard for language names, country and language codes. 
            - Use string format: '[[language_name]] ([[country]]) [[language_code]]'"""
            ),
            static_context=r"Return JSON: {'language': string}",
        )

        res = doc_lang_func(
            content,
            preview=False,
            response_format={"type": "json_object"},
            seed=self.seed,
        )

        return res.language

    def forward(self, **kwargs) -> Summary:
        logger.debug("Starting Hierarchical Summary...")
        self.clear()

        # compute required tokens
        logger.debug("Computing required tokens...")
        total_tokens = self.compute_required_tokens(self.content, count_context=False)
        chunk_size = self.calculate_chunk_size(total_tokens)

        if total_tokens > chunk_size:
            summary_token_count = self._max_context_tokens() + 1
            data = self.content
            doc_type = None

            while summary_token_count > self.max_output_tokens:
                logger.debug("Chunking content...")
                chunks = self.chunk_by_token_count(str(data), chunk_size)
                if doc_type is None:
                    logger.debug("Determining document type and language...")
                    doc_type = self.get_document_type(chunks[0])
                    doc_lang = self.get_document_language(chunks[0])
                    self.adapt("[[DOCUMENT TYPE]]\n" + doc_type.value)
                    self.adapt("[[DOCUMENT LANGUAGE]]\n" + doc_lang)

                logger.debug("Patching asyncio...")
                nest_asyncio.apply()
                logger.debug("Preparing asyncio loop...")
                loop = always_get_an_event_loop()
                logger.debug(f"Processing {len(chunks)} chunks...")
                res, summary_token_count = loop.run_until_complete(
                    self.summarize_chunks(chunks, **kwargs)
                )
                logger.debug(f"Processing of {len(chunks)} chunks completed")
                data = res

            # overwrite type with initially detected type
            if hasattr(res, "type"):
                res.type = doc_type

        else:
            logger.debug("Content is within token limit, processing in one go...")
            logger.debug("Determining document type and language...")
            doc_type = self.get_document_type(self.content)
            doc_lang = self.get_document_language(self.content)

            self.adapt("[[DOCUMENT TYPE]]\n" + doc_type.value)
            self.adapt("[[DOCUMENT LANGUAGE]]\n" + doc_lang)

            logger.debug("Processing content...")
            res = super().forward(
                self.content,
                preview=False,
                response_format={"type": "json_object"},
            )
            res.type = doc_type

        # log compression ratio
        result_tokens = self.compute_required_tokens(res, count_context=False)
        logger.debug(
            f"Compression ratio: {total_tokens} -> {result_tokens} ({ result_tokens/total_tokens:.2f})"
        )
        return res
