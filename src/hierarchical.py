import os
import re
import tempfile
import urllib.request
from textwrap import dedent
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, field_validator
from symai.components import FileReader, Function, ValidatedFunction
from symai.core_ext import bind

from .types import TYPE_SPECIFIC_PROMPTS, DocumentType


class Summary(BaseModel):
    summary: str
    facts: List[str]
    type: Optional[str] = None
    quotes: Optional[List[str]] = None


# TODO: move to symai
class HierarchicalSummary(ValidatedFunction):
    # Define the prompt types as class variables
    def __init__(
        self,
        file_link: str = None,
        content: str = None,
        document_name: str = None,
        min_num_chunks: int = 5,
        min_chunk_size: int = 250,
        max_output_tokens: int = 10000,
        user_prompt: str = None,
        include_quotes: bool = False,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        # only allow file_link or content
        assert (file_link and not content) or (content and not file_link)

        if content is not None:
            assert document_name is not None

        super().__init__(data_model=Summary, retry_count=5, *args, **kwargs)
        self.file_link = file_link
        self.min_num_chunks = min_num_chunks
        self.min_chunk_size = min_chunk_size
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

        # Content type is unknown at initialization
        self.document_type = None

    def read_file(self, file_link: str):
        self.print_verbose(f"Reading file from {file_link}")
        reader = FileReader()
        content = reader(file_link)
        file_name = os.path.basename(file_link)
        val = f"[[DOCUMENT::{file_name}]]: <<<\n{str(content)}\n>>>\n"
        return val, file_name

    def download_file(self, file_link: str):
        self.print_verbose(f"Downloading file from {file_link}")

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
            This summary is intended for a specific audience or purpose.
            Given the following details, ensure that the summary and the list of facts are tailored to the user's needs and contain all relevant information.
                        
            >>>
            {self.user_prompt}
            <<<"""
            )

        prompt_text = dedent(
            f"""
            [[Document Processing Task]]

            [Main Objective]
            Create a comprehensive summary of the provided content and return the result as JSON.
            The type of the provided content is specified in [[CONTENT TYPE]].
            Information relevant to the type of content should be stored in the list of facts.   
                     
            {type_specific_prompt}
                     
            {user_prompt if self.user_prompt is not None else ""}
                        
            [Language Requirements]
            The summary must be in the language specified in [[CONTENT LANGUAGE]], regardless of the source material.

            [Key Requirements]
            - Summarize the content in a clear and concise manner, ensuring that all relevant points are captured.
            - Extract important facts from the text and return them in a list in JSON format as 'facts'.            
            - **IMPORTANT**: Ensure that the summary is consistent with the facts. Do not add information not contained in the document.
            {"- Extract significant quotes that support the main points and return them in a list in JSON format as 'quotes'" if self.include_quotes else ""}
            {"- The quotes should be chosen based on relevancy to the type-specific and user instructions, especially if a particular audience is specified" if self.include_quotes else ""}
            
            [Output Format]
            JSON schema: {{"summary": "string", 
            "facts": "array of strings" 
            {', "quotes": "array of strings"' if self.include_quotes else ""}}}
        """
        )

        logger.debug(prompt_text)
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

    @bind(engine="neurosymbolic", property="api_max_context_tokens")(lambda: 0)
    def _max_context_tokens(_):
        pass

    @bind(engine="neurosymbolic", property="api_max_response_tokens")(lambda: 0)
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
        chunks = []

        # split text into words, punctuation, and spaces
        words = self.split_words(text)

        # chunking
        num_words = len(words)
        step_size = max(num_words // 2, 1)
        min_step_size = 10

        idx = 0
        chunked_word_count = 0
        cur_chunk = []

        # combine chunks based on token length of full request
        while chunked_word_count != len(words):
            if idx + step_size < num_words:
                candidate = words[idx : idx + step_size]
            else:
                candidate = words[idx:]
            candidate_len = self.compute_required_tokens(
                "".join(cur_chunk + candidate), count_context=include_context
            )

            if candidate_len > chunk_size:
                step_size = step_size // 2
                if step_size < min_step_size:
                    chunks.append("".join(cur_chunk))
                    chunked_word_count += len(cur_chunk)
                    step_size = len(cur_chunk)
                    cur_chunk = []
            else:
                cur_chunk += candidate
                idx += len(candidate)
                step_size = min(int(step_size * 1.05), num_words - idx)

                if step_size == 0:
                    chunks.append("".join(cur_chunk))
                    chunked_word_count += len(cur_chunk)
                    step_size = len(cur_chunk)
                    cur_chunk = []

        return chunks

    def summarize_chunks(self, chunks):
        chunk_summaries = []
        chunk_facts = []
        chunk_quotes = []

        for chunk in chunks:
            res, usage = super().forward(
                chunk,
                preview=False,
                response_format={"type": "json_object"},
            )
            chunk_summaries.append(res.summary)
            chunk_facts.extend(res.facts)
            if res.quotes:
                chunk_quotes.extend(res.quotes)

        res = Summary(
            summary="\n".join(chunk_summaries),
            facts=chunk_facts,
            quotes=chunk_quotes,
        )
        return res, self.compute_required_tokens(res.summary, count_context=False)

    def calculate_chunk_size(self, total_tokens):
        num_prompt_tokens = self.compute_required_tokens("", count_context=True)
        max_tokens_per_chunk = int(
            self._max_context_tokens() - num_prompt_tokens * 0.8
        )  # leave some headroom
        chunk_size = total_tokens // self.min_num_chunks

        if self.min_chunk_size < chunk_size:
            num_chunks = self.min_num_chunks
            while chunk_size - num_prompt_tokens > max_tokens_per_chunk:
                num_chunks += 1
                chunk_size = total_tokens // num_chunks - num_prompt_tokens

            return max(self.min_chunk_size, total_tokens // num_chunks)
        else:
            return self.min_chunk_size

    def get_document_type(self, content):
        # Prepare a list of all values in the enum DocumentType
        allowed_types = [doc_type.value for doc_type in DocumentType]

        class ContentType(BaseModel):
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

        res, usage = doc_type_func(
            content,
            preview=False,
            response_format={"type": "json_object"},
            seed=self.seed,
        )

        # Store the content type for use in prompt

        self.document_type = DocumentType(res.type)

        self.add_usage(usage)
        return self.document_type

    def get_document_language(self, content):
        class ContentLanguage(BaseModel):
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

        res, usage = doc_lang_func(
            content,
            preview=False,
            response_format={"type": "json_object"},
            seed=self.seed,
        )

        # add to overall usage
        self.add_usage(usage)
        return res.language

    def forward(self) -> Summary:
        self.reset_usage()
        self.clear()

        # compute required tokens
        total_tokens = self.compute_required_tokens(self.content, count_context=False)
        chunk_size = self.calculate_chunk_size(total_tokens)

        if total_tokens > chunk_size:
            summary_token_count = self._max_context_tokens() + 1
            data = self.content
            facts = None
            quotes = None
            doc_type = None

            while summary_token_count > self.max_output_tokens:
                chunks = self.chunk_by_token_count(data, chunk_size)
                if doc_type is None:
                    doc_type = self.get_document_type(chunks[0])
                    doc_lang = self.get_document_language(chunks[0])
                    self.adapt("[[DOCUMENT TYPE]]\n" + doc_type.value)
                    self.adapt("[[DOCUMENT LANGUAGE]]\n" + doc_lang)

                res, summary_token_count = self.summarize_chunks(chunks)
                data = res.summary

                # store facts and quotes from first summarization pass, do not overwrite
                if facts is None:
                    facts = res.facts
                    quotes = res.quotes

            # collect and return results
            res = Summary(
                summary=data,
                facts=facts,
                type=doc_type,
                quotes=quotes,
            )
            return res, self.get_usage()
        else:
            doc_type = self.get_document_type(self.content)
            doc_lang = self.get_document_language(self.content)

            self.adapt("[[DOCUMENT TYPE]]\n" + doc_type.value)
            self.adapt("[[DOCUMENT LANGUAGE]]\n" + doc_lang)

            res, usage = super().forward(
                self.content,
                preview=False,
                response_format={"type": "json_object"},
            )
            res.type = doc_type

        return res, usage
