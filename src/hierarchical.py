import os
import re
from typing import List
import urllib.request

from pydantic import BaseModel, field_validator

from symai.components import FileReader, Function, ValidatedFunction
from symai.core_ext import bind


class Summary(BaseModel):
    summary: str
    facts: List[str]
    type: str = None


# TODO: move to symai
class HierarchicalSummary(ValidatedFunction):
    def __init__(
        self,
        file_link: str = None,
        content: str = None,
        asset_name: str = None,
        min_num_chunks: int = 5,
        min_chunk_size: int = 250,
        max_output_tokens: int = 10000,
        content_types: List[str] = None,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        # only allow file_link or content
        assert (file_link and not content) or (content and not file_link)

        if content is not None:
            assert asset_name is not None

        super().__init__(data_model=Summary, retry_count=5, *args, **kwargs)
        self.file_link = file_link
        self.min_num_chunks = min_num_chunks
        self.min_chunk_size = min_chunk_size
        self.max_output_tokens = max_output_tokens
        self.content_types = content_types
        self.seed = seed

        file_content = None
        file_name = None
        if file_link is not None:
            if file_link.startswith("http"):
                file_content, file_name = self.download_file(file_link)
            else:
                file_content, file_name = self.read_file(file_link)
        else:
            file_name = asset_name
            file_content = str(content)
        self.content = f"[[ASSET::{file_name}]]: <<<\n{str(file_content)}\n>>>\n"

    def read_file(self, file_link: str):
        self.print_verbose(f"Reading file from {file_link}")
        reader = FileReader()
        content = reader(file_link)
        file_name = os.path.basename(file_link)
        val = f"[[ASSET::{file_name}]]: <<<\n{str(content)}\n>>>\n"
        return val, file_name

    def download_file(self, file_link: str):
        self.print_verbose(f"Downloading file from {file_link}")
        with urllib.request.urlopen(file_link) as f:
            content = f.read().decode("utf-8")
            file_name = urllib.parse.urlparse(file_link).path
            val = f"[[ASSET::{file_name}]]: <<<\n{str(content)}\n>>>\n"
            return val, file_name

    @property
    def prompt(self):
        return (
            f"Create a comprehensive summary of the provided content and return the result as JSON.\n"
            + (
                f"The type of the provided content is specified in [CONTENT TYPE].\n"
                if self.content_types is not None
                else ""
            )
            + "The summary must be in the language specified in [[CONTENT LANGUAGE]], regardless of the source material.\n"
            + f"Extract important facts from the text and return them in a list in JSON format as 'facts'.\n"
            + f"[[IMPORTANT]] Ensure that the summary is consistent with the facts. Do not add information not contained in the text.\n"
            + r'JSON schema: {"summary": "string", "facts": "array of strings"}'
        )

    @property
    def static_context(self):
        return (
            "Create a comprehensive summary of the provided text and extract important facts.\n"
            + "The summary must be in the same language as the text.\n"
            + "Return the summary in JSON format with the provided JSON schema.\n"
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

        for chunk in chunks:
            res, usage = super().forward(
                chunk,
                preview=False,
                response_format={"type": "json_object"},
            )
            chunk_summaries.append(res.summary)
            chunk_facts.extend(res.facts)

        res = Summary(
            summary="\n".join(chunk_summaries),
            facts=chunk_facts,
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

    def get_asset_type(self, content):
        if self.content_types is not None:
            # construct pydantic BaseModel for content types
            class ContentType(BaseModel):
                type: str

                @field_validator("type")
                def validate_type(cls, v):
                    assert v in self.content_types
                    return v

            # construct function to determine asset type, use ValidatedFunction to restrict to allowed types
            asset_type_func = ValidatedFunction(
                data_model=ContentType,
                retry_count=self.retry_count,
                prompt="What type of content is this text?\n"
                + f"Allowed types: {', '.join(self.content_types)}\n"
                + "The content type must be mapped exactly/literally to one of the listed types. No other type allowed!\n\n",
                static_context=r"Return JSON: {'type': string}",
            )

            res, usage = asset_type_func(
                content,
                preview=False,
                response_format={"type": "json_object"},
                seed=self.seed,
            )

            # add to overall usage
            self.add_usage(usage)
            return res.type
        else:
            return "Unknown"
    
    def get_asset_language(self, content):
        class ContentLanguage(BaseModel):
            language: str

        # construct function to determine asset type, use ValidatedFunction to restrict to allowed types
        asset_type_func = ValidatedFunction(
            data_model=ContentLanguage,
            retry_count=self.retry_count,
            prompt="Which language is this text in?\n"
            + "Follow the ISO 639 standard for language names, country and language codes; use string format: '[[language_name]] ([[country]]) [[language_code]]'\n",
            static_context=r"Return JSON: {'language': string}",
        )

        res, usage = asset_type_func(
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
            asset_type = None

            while summary_token_count > self.max_output_tokens:
                chunks = self.chunk_by_token_count(data, chunk_size)
                if asset_type is None:
                    asset_type = self.get_asset_type(chunks[0])
                    asset_language = self.get_asset_language(chunks[0])
                    self.adapt("[[CONTENT TYPE]]\n" + asset_type)
                    self.adapt("[[CONTENT LANGUAGE]]\n" + asset_language)

                res, summary_token_count = self.summarize_chunks(chunks)
                data = res.summary

                # store facts from first summarization pass, do not overwrite
                if facts is None:
                    facts = res.facts

            # collect and return results
            res = Summary(
                summary=data,
                facts=facts,
            )
            res.type = asset_type
            return res, self.get_usage()
        else:
            asset_type = self.get_asset_type(self.content)
            asset_language = self.get_asset_language(self.content)

            self.adapt("[[CONTENT TYPE]]\n" + asset_type)
            self.adapt("[[CONTENT LANGUAGE]]\n" + asset_language)

            res, usage = super().forward(
                self.content,
                preview=False,
                response_format={"type": "json_object"},
            )
            res.type = asset_type

        return res, usage
