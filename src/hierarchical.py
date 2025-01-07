import os
import re
from typing import List
import urllib.request

from pydantic import BaseModel, field_validator

from symai.components import FileReader, Function, ValidatedFunction
from symai.core_ext import bind
import tempfile


class Summary(BaseModel):
    summary: str
    facts: List[str]
    type: str = None


# TODO: move to symai
class HierarchicalSummary(ValidatedFunction):
    # Define the prompt types as class variables
    base_prompts = {
        "Paper": {
            "base": "Extract the title, authors, and publication details. Identify the main topic and scope.",
            "subtypes": {
                "Scientific Paper": "Extract key statements, contributions, main results, and important references. Focus on methodology and findings.",
                "Research Paper": "Focus on research questions, methodology, data analysis, and conclusions. Include limitations and future work.",
                "Review Paper": "Highlight the reviewed topics, key findings from literature, and synthesis of current knowledge."
            }
        },
        "Presentation": {
            "base": "Identify the presenter, target audience, and overall structure.",
            "subtypes": {
                "Keynote": "Include speaker details and their expertise. Highlight key messages and main takeaways.",
                "Presentation Slides": "Determine if this is a motivational talk, results presentation, or idea/pitch. For motivational talks, focus on key messages and call-to-action. For result presentations, emphasize numerical results and achievements. For idea/pitch presentations, highlight the core idea and value proposition."
            }
        }
    }

    standalone_prompts = {
        "Interview": "Identify and distinguish between different speakers. Include key quotes and main discussion points.",
        "Report": "Highlight numerical results, key statistics, and main takeaways. Include significant findings and conclusions.",
        "Book": "Include author information, main plot points, and key character descriptions. Highlight character development and relationships.",
    }

    def __init__(
        self,
        file_link: str = None,
        content: str = None,
        asset_name: str = None,
        min_num_chunks: int = 5,
        min_chunk_size: int = 250,
        max_output_tokens: int = 10000,
        content_types: List[str] = None,
        user_prompt: str = None,
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
        self.user_prompt = user_prompt
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
        self.content_only = str(file_content)

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
        type_prompt = ""
        if self.content_types is not None and hasattr(self, '_content_type'):
            content_type = self._content_type
            
            # Check if this is a subtype
            for base_type, base_info in self.base_prompts.items():
                if content_type in base_info["subtypes"]:
                    # Combine base prompt with subtype prompt
                    type_prompt = f"\nFor this {content_type}:\n"
                    type_prompt += f"- {base_info['base']}\n"
                    type_prompt += f"- {base_info['subtypes'][content_type]}"
                    break
            # If not found in subtypes, check standalone prompts
            if not type_prompt and content_type in self.standalone_prompts:
                type_prompt = f"\nFor this {content_type}: {self.standalone_prompts[content_type]}"

        if self.user_prompt is not None:
            user_prompt = "Given the following information, extract important related information from the text and add them to the list of facts."
            user_prompt += "\n" + self.user_prompt
            
        return (
            f"[Summary Generation Task]\n\n"
            + "[Main Objective]\n"
            + "Create a comprehensive summary of the provided content and return the result as JSON.\n\n"
            + (
                "[Content Type]\n"
                + "The type of the provided content is specified in [[CONTENT TYPE]].\n\n"
                if self.content_types is not None
                else ""
            )
            + "[Type-Specific Instructions]\n"
            + type_prompt  # Add the type-specific prompt
            + "[User Instructions]\n"
            + user_prompt
            + "\n[Language Requirements]\n"
            + "The summary must be in the language specified in [[CONTENT LANGUAGE]], regardless of the source material.\n\n"
            + "[Key Requirements]\n"
            + "- Extract important facts from the text and return them in a list in JSON format as 'facts'\n"
            + "- **IMPORTANT**: Ensure that the summary is consistent with the facts\n"
            + "- Do not add information not contained in the text\n\n"
            + "[Output Format]\n"
            + r'JSON schema: {"summary": "string", "facts": "array of strings"}\n'
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
            # Flatten the allowed types to include both base types and subtypes
            allowed_types = set()
            for base_type, base_info in self.base_prompts.items():
                allowed_types.add(base_type)
                allowed_types.update(base_info["subtypes"].keys())
            allowed_types.update(self.standalone_prompts.keys())

            class ContentType(BaseModel):
                type: str

                @field_validator("type")
                def validate_type(cls, v):
                    assert v in allowed_types, f"Type must be one of: {', '.join(sorted(allowed_types))}"
                    return v

            # construct function to determine asset type
            asset_type_func = ValidatedFunction(
                data_model=ContentType,
                retry_count=self.retry_count,
                prompt="What type of content is this text?\n"
                + f"Allowed types: {', '.join(sorted(allowed_types))}\n"
                + "The content type must be mapped exactly/literally to one of the listed types. No other type allowed!\n\n"
                + "Note: Some types are subtypes of others:\n"
                + "\n".join(
                    f"- {base_type}: {', '.join(base_info['subtypes'].keys())}"
                    for base_type, base_info in self.base_prompts.items()
                    if base_info['subtypes']
                ),
                static_context=r"Return JSON: {'type': string}",
            )

            res, usage = asset_type_func(
                content,
                preview=False,
                response_format={"type": "json_object"},
                seed=self.seed,
            )

            # Store the content type for use in prompt
            self._content_type = res.type

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
                type=asset_type,
            )
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
