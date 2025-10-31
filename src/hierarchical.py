import asyncio
import functools
import os
import re
import tempfile
import threading
import urllib.request
from textwrap import dedent
from typing import List, Optional, Dict
import contextvars

import nest_asyncio
from loguru import logger
from pydantic import Field, field_validator
from symai import Import, Symbol
from symai.components import FileReader, Function, DynamicEngine
from symai.core_ext import bind
from symai.models import LLMDataModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from .functions import ValidatedFunction
from .types import TYPE_SPECIFIC_PROMPTS, DocumentType

# ChonkieChunker will be loaded lazily when needed

# Tokenizer cache: thread-safe cache with max size enforcement
_TOKENIZER_CACHE: Dict[str, object] = {}
_TOKENIZER_CACHE_LOCK = threading.RLock()
_TOKENIZER_CACHE_MAX_SIZE = 1  # Normally should be 1


def get_current_tokenizer():
    """Get the tokenizer from the current engine.

    Returns:
        tokenizer object or None if not available
    """
    try:
        from symai import EngineRepository

        engine_repo = EngineRepository()

        # Try to get dynamic engine first
        current_engine = engine_repo.get_dynamic_engine_instance()

        if current_engine and hasattr(current_engine, 'tokenizer'):
            return current_engine.tokenizer

        # Fallback to registered neurosymbolic engine
        engine = engine_repo.get('neurosymbolic')
        if engine and hasattr(engine, 'tokenizer'):
            return engine.tokenizer

        return None
    except Exception as e:
        logger.debug(f"Failed to get tokenizer from engine: {e}")
        return None


def _get_cached_tokenizer(cache_key: str = "default"):
    """Get tokenizer from cache or fetch and cache it.

    Args:
        cache_key: Key for the cache (currently unused, kept for future extensibility)

    Returns:
        tokenizer object or None if not available
    """
    global _TOKENIZER_CACHE

    with _TOKENIZER_CACHE_LOCK:
        # Check cache size and warn if exceeded
        cache_size = len(_TOKENIZER_CACHE)
        if cache_size > 2:
            logger.warning(
                f"Tokenizer cache size ({cache_size}) exceeds expected maximum (2). "
                f"This may indicate a memory leak or inefficient caching."
            )
        elif cache_size > _TOKENIZER_CACHE_MAX_SIZE:
            logger.debug(
                f"Tokenizer cache size ({cache_size}) exceeds normal maximum ({_TOKENIZER_CACHE_MAX_SIZE})."
            )

        # Log current cache size
        if cache_size > 0:
            logger.debug(f"Tokenizer cache size: {cache_size}")

        # Use cache key for potential future multi-tokenizer support
        if cache_key in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[cache_key]

        # Fetch tokenizer
        tokenizer = get_current_tokenizer()

        if tokenizer is not None:
            # Enforce max cache size: remove oldest entry if needed (FIFO)
            if len(_TOKENIZER_CACHE) >= _TOKENIZER_CACHE_MAX_SIZE:
                # Remove first (oldest) entry
                oldest_key = next(iter(_TOKENIZER_CACHE))
                del _TOKENIZER_CACHE[oldest_key]
                logger.debug(f"Removed oldest tokenizer from cache (key: {oldest_key})")

            _TOKENIZER_CACHE[cache_key] = tokenizer
            logger.debug(f"Cached tokenizer (key: {cache_key})")

        return tokenizer


def count_tokens_fast(text, log_offset: bool = False) -> Optional[int]:
    """Fast-path token counting using direct tokenizer encoding.

    Bypasses Function(preview=...) when only counting tokens.

    Args:
        text: Text to count tokens for (str, LLMDataModel, or other object that can be converted to string)
        log_offset: If True, log tokens added as offset

    Returns:
        Number of tokens or None if tokenizer not available
    """
    tokenizer = _get_cached_tokenizer()

    if tokenizer is None:
        return None

    try:
        # Convert input to string representation
        if isinstance(text, LLMDataModel):
            # For LLMDataModel, convert to JSON string
            text_str = text.model_dump_json()
        else:
            text_str = str(text)

        # Direct encoding for fast token count
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text_str)

            # Handle different tokenizer return types
            if isinstance(tokens, (list, tuple)):
                token_count = len(tokens)
            elif hasattr(tokens, 'shape'):
                # Handle numpy/torch tensors
                token_count = tokens.shape[0] if len(tokens.shape) > 0 else len(tokens)
            elif hasattr(tokens, '__len__'):
                # Handle other sequence-like types
                token_count = len(tokens)
            else:
                logger.debug(f"Unsupported tokenizer return type: {type(tokens)}")
                return None

            if token_count is not None and log_offset:
                logger.debug(f"Tokens added (offset): {token_count}")

            return token_count
        else:
            logger.debug("Tokenizer does not have encode method")
            return None
    except Exception as e:
        logger.debug(f"Fast token counting failed: {e}")
        return None



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
        document_lang: str = None,
        asset_name: str = None,
        data_model: LLMDataModel = Summary,
        min_num_chunks: int = 5,
        min_chunk_size: int = 250,
        max_chunk_size: int = 1000,
        max_output_tokens: int = 10000,
        user_prompt: str = None,
        include_quotes: bool = False,
        tokenizer_name: str = "gpt2",
        chunker_name: str = "RecursiveChunker",
        seed: int = 42,
        enable_initial_compression: bool = True,
        plain_text_only: bool = False,
        engine: Optional[object] = None,
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
        self.document_lang = document_lang
        self.file_link = file_link
        self.min_num_chunks = min_num_chunks
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_output_tokens = max_output_tokens
        self.user_prompt = user_prompt
        self.include_quotes = include_quotes
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self.enable_initial_compression = enable_initial_compression
        # If True, bypass symai FileReader (and Tika) and read files via Python open(); assumes plain-text inputs
        self.plain_text_only = plain_text_only
        self.engine = engine  # Store the engine instance

        # Prepare content and file metadata
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

        # Lazy load chunker only when needed (avoids importing transformers/torch at module level)
        self._chunker = None
        self.chunker_type = chunker_name

        # Content type is unknown at initialization
        self.document_type = None

        # Prompt memoization cache (thread-safe)
        self._prompt_lock = threading.RLock()
        self._cached_prompt = None
        self._cached_user_prompt = None
        self._cached_document_type = None

    def read_file(self, file_link: str):
        logger.info(f"Reading file from {file_link}")
        if self.plain_text_only:
            # Basic Python file read assuming UTF-8 plain text
            try:
                with open(file_link, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Plain-text read failed for {file_link}: {e}")
                raise
        else:
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
        with self._prompt_lock:
            # Check if cache is valid (user_prompt and document_type must match)
            # The == operator correctly handles None comparisons (None == None is True)
            if (
                self._cached_prompt is not None
                and self._cached_user_prompt == self.user_prompt
                and self._cached_document_type == self.document_type
            ):
                return self._cached_prompt

            # Build the prompt
            # Get type-specific prompt
            type_specific_prompt = ""
            if self.document_type and self.document_type in TYPE_SPECIFIC_PROMPTS:
                type_specific_prompt = dedent(
                    f"""[Type-Specific Instructions]
                For this {self.document_type.value}: {TYPE_SPECIFIC_PROMPTS[self.document_type]}"""
                )

            # Initialize user_prompt to handle None case
            user_prompt = ""
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

                {user_prompt}

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

            # Cache the result
            self._cached_prompt = prompt_text
            self._cached_user_prompt = self.user_prompt
            self._cached_document_type = self.document_type

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
        # Fast-path: when only counting data tokens (count_context=False), use direct tokenizer encoding
        if not count_context:
            # Try fast-path token counting (handles LLMDataModel, str, and other types)
            fast_count = count_tokens_fast(data, log_offset=True)
            if fast_count is not None:
                return fast_count

            # Fallback to original method if fast path fails
            logger.debug("Fast-path token counting failed, falling back to Function(preview=...)")

        # Original method: construct preview function for full context counting
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

    def _configure_torch_cpu(self):
        """Helper method to configure torch to use CPU operations.

        Called both before and after ChonkieChunker import to ensure
        torch is configured regardless of when it gets imported.
        """
        try:
            import torch
            if hasattr(torch, 'set_default_device'):
                # PyTorch 2.0+: set default device to CPU
                torch.set_default_device('cpu')
            elif hasattr(torch, 'set_default_tensor_type'):
                # Older PyTorch versions: use CPU tensor type
                torch.set_default_tensor_type('torch.FloatTensor')
        except (ImportError, Exception):
            # torch not available, already configured, or error - that's fine
            pass

    def _get_chunker(self):
        """Lazy load ChonkieChunker to avoid importing transformers/torch at module level.

        Attempts to configure transformers/torch to disable unused engines:
        - Sets environment variables to prefer CPU (before imports)
        - Configures torch to prefer CPU operations (before and after import)
        - Sets HuggingFace to prefer CPU (via environment variable)
        """
        if self._chunker is None:
            # Set environment variables BEFORE any imports to prevent GPU initialization
            # This must happen before ChonkieChunker is loaded, as it may import torch/transformers
            if "HF_DEVICE" not in os.environ:
                os.environ["HF_DEVICE"] = "cpu"

            # Attempt to configure torch BEFORE loading ChonkieChunker
            # If torch is already imported elsewhere, this will configure it
            self._configure_torch_cpu()

            # Load ChonkieChunker (may trigger transformers/torch imports)
            ChonkieChunker = Import.load_expression("ExtensityAI/chonkie-symai", "ChonkieChunker")

            # Configure torch again after loading (in case it was just imported)
            self._configure_torch_cpu()

            self._chunker = ChonkieChunker(tokenizer_name=self.tokenizer_name)

        return self._chunker

    def chunk_by_token_count(self, text, chunk_size, include_context=False):
        # prepare results
        chunker = self._get_chunker()
        chunks = chunker(data=Symbol(text), chunker_name=self.chunker_type, chunk_size=chunk_size)
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
            def worker():
                # Ensure DynamicEngine context is established in the executor thread
                if self.engine is not None:
                    from symai.components import DynamicEngine
                    with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                        return super(HierarchicalSummary, self).forward(
                            chunk,
                            preview=False,
                            response_format={"type": "json_object"},
                            **kwargs,
                        )
                else:
                    return super(HierarchicalSummary, self).forward(
                        chunk,
                        preview=False,
                        response_format={"type": "json_object"},
                        **kwargs,
                    )
            # Belt-and-suspenders: propagate ContextVar context to executor, but keep worker re-entry
            ctx = contextvars.copy_context()
            return await loop.run_in_executor(None, lambda: ctx.run(worker))
        """Summarize all chunks concurrently.

        Returns:
            (LLMDataModel, int): aggregated result instance and original number of chunks.
        """
        tasks = [summarize_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        # Aggregate raw results (concatenate strings / extend lists)
        aggregated = gather(results)
        final_res = self.data_model(**aggregated)

        return final_res, len(results)

    # -------------------------------------------------
    # Internal input augmentation
    # -------------------------------------------------
    def _augment_with_user_prompt(self, text: str) -> str:
        """Attach user/purpose prompt to model input body so that the ValidatedFunction sees it in data as well as in prompt."""
        if self.user_prompt:
            return f"{text}\n[[PURPOSE]]\n{self.user_prompt}\n"
        return text

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

        # Use DynamicEngine context if engine is provided
        if self.engine is not None:
            with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                res = doc_type_func(
                    content,
                    preview=False,
                    response_format={"type": "json_object"},
                    seed=self.seed,
                )
        else:
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

        if self.document_lang is not None:
            return self.document_lang

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

        # Use DynamicEngine context if engine is provided
        if self.engine is not None:
            with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                res = doc_lang_func(
                    content,
                    preview=False,
                    response_format={"type": "json_object"},
                    seed=self.seed,
                )
        else:
            res = doc_lang_func(
                content,
                preview=False,
                response_format={"type": "json_object"},
                seed=self.seed,
            )

        return res.language

    def forward(self, **kwargs) -> Summary:
        self.clear()

        # If an engine is provided, wrap all processing with DynamicEngine context
        if self.engine is not None:
            with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                return self._forward_with_engine(**kwargs)
        else:
            return self._forward_with_engine(**kwargs)

    def _forward_with_engine(self, **kwargs) -> Summary:
        # compute required tokens
        total_tokens = self.compute_required_tokens_graceful(self.content, count_context=False)
        if total_tokens is None:
            logger.warning("Total tokens could not be determined.")
            total_tokens = 1

        chunk_size = self.calculate_chunk_size(total_tokens)

        # Always perform a chunked first pass (even if it results in a single chunk)
        data = self.content
        doc_type = None

        chunks = self.chunk_by_token_count(str(self._augment_with_user_prompt(data)), chunk_size)
        if doc_type is None:
            doc_type = self.get_document_type(chunks[0])
            doc_lang = self.get_document_language(chunks[0])
            self.adapt("[[DOCUMENT TYPE]]\n" + doc_type.value)
            self.adapt("[[DOCUMENT LANGUAGE]]\n" + doc_lang)

        nest_asyncio.apply()
        loop = always_get_an_event_loop()
        logger.debug(f"Processing {len(chunks)} chunks (initial summarization)...")
        res, _orig_chunk_count = loop.run_until_complete(
            self.summarize_chunks(chunks, **kwargs)
        )
        logger.debug("Initial chunk processing completed")

        # --- Deduplicate list fields once ---
        res = self._deduplicate_list_fields(res)

        # --- Compression loop: only compress string fields ---
        def _string_field_names():
            for fname, finfo in res.model_fields.items():
                if getattr(finfo, "exclude", False):
                    continue
                val = getattr(res, fname)
                if isinstance(val, str):
                    yield fname

        summary_token_count = self.compute_required_tokens_graceful(res, count_context=False) or 0
        logger.debug(f"Initial aggregated token count: {summary_token_count}")

        compression_attempt = 0

        def _compress_pass(attempt: int):
            logger.debug(
                f"Compression attempt {attempt}: starting (current tokens={summary_token_count}, limit={self.max_output_tokens})"
            )
            improved = False
            for fname in _string_field_names():
                current_val = getattr(res, fname)
                if not current_val or not isinstance(current_val, str):
                    continue
                compressed = self._compress_string_field_recursive(
                    current_val, field_name=fname, attempt=attempt
                )
                if compressed and compressed != current_val:
                    setattr(res, fname, compressed)
                    improved = True
            return improved

        # Optional initial compression pass (can be disabled to avoid recursive inflation)
        if self.enable_initial_compression:
            compression_attempt += 1
            _compress_pass(compression_attempt)
            summary_token_count = self.compute_required_tokens_graceful(res, count_context=False) or 0
            logger.debug(
                f"Post-attempt {compression_attempt} token count: {summary_token_count} (initial pass)"
            )

            # Additional passes only if still above limit
            while summary_token_count > self.max_output_tokens and compression_attempt < 5:
                compression_attempt += 1
                _compress_pass(compression_attempt)
                summary_token_count = self.compute_required_tokens_graceful(
                    res, count_context=False
                ) or 0
                logger.debug(
                    f"Post-attempt {compression_attempt} token count: {summary_token_count}"
                )

        if summary_token_count > self.max_output_tokens:
            logger.warning(
                "Exceeded max_output_tokens after compression attempts; returning best-effort result."
            )

        # overwrite type with initially detected type
        if hasattr(res, "type"):
            res.type = doc_type

        # log compression ratio
        result_tokens = self.compute_required_tokens_graceful(res, count_context=False)
        if result_tokens is not None:
            logger.debug(
                f"Compression ratio: {total_tokens} -> {result_tokens} ({ result_tokens/total_tokens:.2f})"
            )

        return res

    def compute_required_tokens_graceful(self, data, count_context=True):
        try:
            return self.compute_required_tokens(data, count_context=count_context)
        except NotImplementedError:
            logger.debug(
                "compute_required_tokens is not implemented for this engine, returning None"
            )
            return None  # Gracefully handle NotImplementedError; any other exception will be raised

    # -------------------------------------------------
    # Helpers: list deduplication & string compression
    # -------------------------------------------------
    def _deduplicate_list(self, items: List[str]) -> List[str]:
        seen = set()
        result = []
        for it in items or []:
            if not isinstance(it, str):
                continue
            norm = re.sub(r"\s+", " ", it.strip().lower())
            if norm and norm not in seen:
                seen.add(norm)
                result.append(it.strip())
        # prune substrings (keep longer entries) to reduce redundancy
        pruned = []
        for i, x in enumerate(result):
            xl = x.lower()
            longer_exists = any(
                i != j and len(result[j]) > len(x) and xl in result[j].lower()
                for j in range(len(result))
            )
            if longer_exists:
                continue
            pruned.append(x)
        return pruned

    def _deduplicate_list_fields(self, res: LLMDataModel) -> LLMDataModel:
        """Deduplicate all list fields of a model instance in-place and return it."""
        for fname, finfo in res.model_fields.items():
            if getattr(finfo, "exclude", False):
                continue
            val = getattr(res, fname, None)
            if isinstance(val, list):
                deduped = self._deduplicate_list(val)
                if len(deduped) != len(val):
                    logger.debug(
                        f"Deduplicated list field '{fname}' from {len(val)} -> {len(deduped)} items"
                    )
                setattr(res, fname, deduped)
        return res

    def _compress_string_field_recursive(self, text: str, field_name: str, attempt: int) -> str:
        """Recursively invoke HierarchicalSummary on a single string field and return (potentially) shorter result.

        Only replaces the field if the recursive result is not longer than the original.
        """
        logger.debug(f"Attempting recursive compression for field '{field_name}' (len={len(text)}, attempt {attempt})")
        if not text or len(text) < 1024:  # Skip tiny strings to save cost
            return text
        try:
            # Create a nested summarizer instance using the same data model
            inner = HierarchicalSummary(
                content=text,
                document_name=f"{field_name}_compress_attempt_{attempt}.txt",
                data_model=self.data_model,
                min_num_chunks=self.min_num_chunks,
                min_chunk_size=int(self.min_chunk_size * (1 if attempt == 1 else 1 + (attempt - 1) * 0.2)),
                max_chunk_size=int(self.max_chunk_size * (1 if attempt == 1 else 1 + (attempt - 1) * 0.2)),
                max_output_tokens=self.max_output_tokens,
                user_prompt=self.user_prompt,
                include_quotes=self.include_quotes,
                tokenizer_name=self.tokenizer_name,
                chunker_name=self.chunker_type,
                seed=self.seed if self.seed else 42 + attempt,
                enable_initial_compression=False,  # prevent nested mandatory pass
                engine=self.engine,  # Pass the engine to nested instances
            )
            # Reuse already detected type / language to avoid re-detection cost
            if self.document_type:
                inner.document_type = self.document_type
                inner.adapt("[[DOCUMENT TYPE]]\n" + self.document_type.value)
            if self.document_lang:
                inner.adapt("[[DOCUMENT LANGUAGE]]\n" + self.document_lang)

            nested_res = inner.forward()
            # Extract same-named field
            if hasattr(nested_res, field_name):
                new_val = getattr(nested_res, field_name)
                if isinstance(new_val, str) and len(new_val) <= len(text):
                    logger.debug(
                        f"Recursive compression '{field_name}' attempt {attempt}: {len(text)} -> {len(new_val)} chars"
                    )
                    return new_val
        except Exception as e:
            logger.warning(
                f"Recursive compression failed for field '{field_name}' attempt {attempt}: {e}"
            )
        return text
