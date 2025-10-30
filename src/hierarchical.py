import asyncio
import functools
import os
import re
import tempfile
import urllib.request
from textwrap import dedent
from typing import List, Optional, Dict, Any
import contextvars

from .lazy_imports import (
    lazy_nest_asyncio, lazy_loguru, lazy_field, lazy_field_validator,
    lazy_import_class, lazy_symbol, lazy_file_reader, lazy_function,
    lazy_dynamic_engine, lazy_bind, lazy_llm_data_model,
    lazy_before_sleep_log, lazy_retry, lazy_retry_if_exception_type,
    lazy_stop_after_attempt, lazy_wait_exponential_jitter,
    lazy_encoding, lazy_tokenizers, lazy_chonkie_chunker,
    lazy_validated_function, lazy_type_specific_prompts, lazy_document_type,
    get_logger, lazy_symai, lazy_tiktoken
)
from .memoization import memoize_property, memoize_method, memoize, get_memoization_manager


def get_current_tokenizer():
    """Get the tokenizer from the current engine."""
    try:
        symai = lazy_symai()
        EngineRepository = symai.EngineRepository

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
        logger = get_logger()
        logger.debug(f"Could not get tokenizer from symai: {e}")
        return None



def create_summary_class():
    """Create Summary class with lazy imports."""
    LLMDataModel = lazy_llm_data_model()
    Field = lazy_field()

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

    return Summary

# Create the Summary class lazily
Summary = create_summary_class()


def gather(chunks: List):
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


class HierarchicalSummary:
    """HierarchicalSummary class with lazy inheritance."""

    def __init__(self, *args, **kwargs):
        # Initialize base class lazily
        ValidatedFunction = lazy_validated_function()
        # Create a temporary instance to get the base class methods
        self._base_instance = ValidatedFunction.__new__(ValidatedFunction)
        ValidatedFunction.__init__(self._base_instance, *args, **kwargs)

        # Copy base class attributes
        for attr_name in dir(self._base_instance):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self._base_instance, attr_name))

        # Now call our own initialization
        self._init_hierarchical_summary(*args, **kwargs)

    def _init_hierarchical_summary(self, *args, **kwargs):
        # Define the prompt types as class variables
        file_link = kwargs.get('file_link')
        content = kwargs.get('content')
        document_name = kwargs.get('document_name')
        document_lang = kwargs.get('document_lang')
        asset_name = kwargs.get('asset_name')
        data_model = kwargs.get('data_model')
        min_num_chunks = kwargs.get('min_num_chunks', 5)
        min_chunk_size = kwargs.get('min_chunk_size', 250)
        max_chunk_size = kwargs.get('max_chunk_size', 1000)
        max_output_tokens = kwargs.get('max_output_tokens', 10000)
        user_prompt = kwargs.get('user_prompt')
        include_quotes = kwargs.get('include_quotes', False)
        tokenizer_name = kwargs.get('tokenizer_name', "gpt2")
        chunker_name = kwargs.get('chunker_name', "RecursiveChunker")
        seed = kwargs.get('seed', 42)
        enable_initial_compression = kwargs.get('enable_initial_compression', True)
        plain_text_only = kwargs.get('plain_text_only', False)
        engine = kwargs.get('engine')
        # only allow file_link or content
        assert (file_link and not content) or (content and not file_link)

        if document_name is None and asset_name is not None:
            document_name = asset_name

        if content is not None:
            assert document_name is not None

        if data_model is None:
            data_model = Summary
        LLMDataModel = lazy_llm_data_model()
        assert issubclass(data_model, LLMDataModel)
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

        # init chunker
        ChonkieChunker = lazy_chonkie_chunker()
        self.chunker = ChonkieChunker(tokenizer_name=self.tokenizer_name)
        self.chunker_type = chunker_name

        # Content type is unknown at initialization
        self.document_type = None

        # Token tracking for logging
        self._token_offset = 0
        self._total_tokens_processed = 0

    def invalidate_cache(self, cache_type: Optional[str] = None):
        """
        Invalidate memoization cache for this instance.

        Args:
            cache_type: Specific cache to invalidate ('prompts', 'tokens', 'documents', 'models')
                       If None, invalidates all caches for this instance
        """
        manager = get_memoization_manager()
        instance_id = id(self)

        if cache_type:
            # Invalidate specific cache
            cache = manager.get_cache(cache_type)
            # Find and invalidate keys for this instance
            keys_to_remove = []
            for key in cache._cache.keys():
                if str(instance_id) in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                cache.invalidate(key)
        else:
            # Invalidate all caches for this instance
            for cache_name in ['prompts', 'tokens', 'documents', 'models']:
                cache = manager.get_cache(cache_name)
                keys_to_remove = []
                for key in cache._cache.keys():
                    if str(instance_id) in key:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    cache.invalidate(key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this instance."""
        manager = get_memoization_manager()
        instance_id = id(self)

        stats = {}
        for cache_name in ['prompts', 'tokens', 'documents', 'models']:
            cache = manager.get_cache(cache_name)
            cache_stats = cache.stats()

            # Count keys for this instance
            instance_keys = sum(1 for key in cache._cache.keys() if str(instance_id) in key)

            stats[cache_name] = {
                **cache_stats,
                'instance_keys': instance_keys
            }

        return stats

    def _log_token_usage(self, tokens_added: int, operation: str = "processing"):
        """Log token usage for monitoring and debugging."""
        self._total_tokens_processed += tokens_added
        logger = get_logger()
        logger.debug(f"Token usage - {operation}: +{tokens_added} tokens "
                    f"(total processed: {self._total_tokens_processed}, offset: {self._token_offset})")

    def get_token_stats(self) -> Dict[str, int]:
        """Get current token statistics."""
        return {
            'total_processed': self._total_tokens_processed,
            'offset': self._token_offset,
            'net_tokens': self._total_tokens_processed - self._token_offset
        }

    def reset_token_tracking(self):
        """Reset token tracking counters."""
        self._token_offset = 0
        self._total_tokens_processed = 0

    def __getattr__(self, name):
        """Delegate to base instance for missing attributes."""
        if name == '_base_instance':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if '_base_instance' in self.__dict__:
            return getattr(self.__dict__['_base_instance'], name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Delegate to base instance for certain attributes."""
        if name == '_base_instance':
            super().__setattr__(name, value)
            return
        if '_base_instance' in self.__dict__ and hasattr(self.__dict__['_base_instance'], name):
            setattr(self.__dict__['_base_instance'], name, value)
        else:
            super().__setattr__(name, value)

    def read_file(self, file_link: str):
        logger = get_logger()
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
            FileReader = lazy_file_reader()
            reader = FileReader()
            content = reader(file_link)
        file_name = os.path.basename(file_link)
        val = f"[[DOCUMENT::{file_name}]]: <<<\n{str(content)}\n>>>\n"
        return val, file_name

    def download_file(self, file_link: str):
        logger = get_logger()
        logger.info(f"Downloading file from {file_link}")

        with urllib.request.urlopen(file_link) as f:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(f.read())
                tmp_file.flush()
                tmp_file_name = tmp_file.name

        content, file_name = self.read_file(tmp_file_name)
        os.remove(tmp_file_name)
        return content, file_name

    @memoize_property('prompts', key_func=lambda self: f"prompt:{id(self)}:{self.user_prompt}:{getattr(self, 'document_type', None)}")
    def prompt(self):
        # Get type-specific prompt
        TYPE_SPECIFIC_PROMPTS = lazy_type_specific_prompts()
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

    @memoize_property('prompts', key_func=lambda self: f"static_context:{id(self)}")
    def static_context(self):
        return dedent(
            """
            Create a comprehensive summary of the provided text and extract important facts.
            The summary must be in the same language as the text.
            Return the summary in JSON format with the provided JSON schema.
        """
        )

    def _compute_required_tokens(self):
        bind = lazy_bind()
        return bind(engine="neurosymbolic", property="compute_required_tokens")(lambda: 0)

    def _max_context_tokens(self):
        bind = lazy_bind()
        return bind(engine="neurosymbolic", property="max_context_tokens")

    def _max_response_tokens(self):
        bind = lazy_bind()
        return bind(engine="neurosymbolic", property="max_response_tokens")

    def _compute_remaining_tokens(self):
        bind = lazy_bind()
        return bind(engine="neurosymbolic", property="compute_remaining_tokens")(lambda: 0)

    def _fast_path_token_count(self, text: str) -> Optional[int]:
        """
        Fast-path token counting that bypasses Function(preview=...) when only counting tokens.
        Uses direct tokenizer encoding for maximum performance.
        """
        try:
            # Try to get tokenizer from symai first
            tokenizer = get_current_tokenizer()
            if tokenizer:
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
                self._log_token_usage(token_count, "fast-path counting")
                return token_count

            # Fallback to tiktoken if available
            tiktoken = lazy_tiktoken()
            try:
                encoding = tiktoken.get_encoding("gpt2")  # Default encoding
                tokens = encoding.encode(text)
                token_count = len(tokens)
                self._log_token_usage(token_count, "fast-path counting (tiktoken)")
                return token_count
            except Exception:
                pass

            # Fallback to tokenizers library
            tokenizers = lazy_tokenizers()
            try:
                tokenizer = tokenizers.Tokenizer.from_pretrained(self.tokenizer_name)
                tokens = tokenizer.encode(text)
                token_count = len(tokens.tokens)
                self._log_token_usage(token_count, "fast-path counting (tokenizers)")
                return token_count
            except Exception:
                pass

            return None
        except Exception as e:
            logger = get_logger()
            logger.debug(f"Fast-path token counting failed: {e}")
            return None

    @memoize_method('tokens', key_func=lambda self, data, count_context: f"tokens:{id(self)}:{hash(str(data))}:{count_context}:{self.seed}")
    def compute_required_tokens(self, data, count_context=True):
        # Fast-path: if not counting context, try direct token counting first
        if not count_context:
            fast_count = self._fast_path_token_count(str(data))
            if fast_count is not None:
                logger = get_logger()
                logger.debug(f"Fast-path token count: {fast_count} tokens")
                return fast_count

        # Fallback to original method with Function preview
        # construct preview function
        Function = lazy_function()
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
        token_count = self._compute_required_tokens(preview.prop.prepared_input)
        if token_count is not None:
            self._log_token_usage(token_count, "Function preview counting")
        return token_count

    def split_words(self, text):
        return re.split(r"(\W+)", text)

    @memoize_method('documents', key_func=lambda self, text, chunk_size, include_context: f"chunks:{id(self)}:{hash(text)}:{chunk_size}:{include_context}:{self.chunker_type}")
    def chunk_by_token_count(self, text, chunk_size, include_context=False):
        # prepare results
        Symbol = lazy_symbol()
        chunks = self.chunker(data=Symbol(text), chunker_name=self.chunker_type, chunk_size=chunk_size)
        return chunks

    async def summarize_chunks(self, chunks, **kwargs):
        retry = lazy_retry()
        retry_if_exception_type = lazy_retry_if_exception_type()
        wait_exponential_jitter = lazy_wait_exponential_jitter()
        stop_after_attempt = lazy_stop_after_attempt()
        before_sleep_log = lazy_before_sleep_log()
        logger = get_logger()

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
                    DynamicEngine = lazy_dynamic_engine()
                    with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                        # Use object.__getattribute__ to avoid triggering property descriptors
                        base_instance = object.__getattribute__(self, '_base_instance')
                        return base_instance.forward(
                            chunk,
                            preview=False,
                            response_format={"type": "json_object"},
                            **kwargs,
                        )
                else:
                    # Use object.__getattribute__ to avoid triggering property descriptors
                    base_instance = object.__getattribute__(self, '_base_instance')
                    return base_instance.forward(
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

    @memoize_method('models', key_func=lambda self, content: f"doc_type:{id(self)}:{hash(content)}:{self.seed}")
    def get_document_type(self, content):
        # Lazy imports
        DocumentType = lazy_document_type()
        LLMDataModel = lazy_llm_data_model()
        field_validator = lazy_field_validator()
        ValidatedFunction = lazy_validated_function()
        DynamicEngine = lazy_dynamic_engine()

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

    @memoize_method('models', key_func=lambda self, content: f"doc_lang:{id(self)}:{hash(content)}:{self.document_lang}:{self.seed}")
    def get_document_language(self, content):
        # Lazy imports
        LLMDataModel = lazy_llm_data_model()
        ValidatedFunction = lazy_validated_function()
        DynamicEngine = lazy_dynamic_engine()

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

    def forward(self, **kwargs):
        self.clear()

        # If an engine is provided, wrap all processing with DynamicEngine context
        if self.engine is not None:
            DynamicEngine = lazy_dynamic_engine()
            with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                return self._forward_with_engine(**kwargs)
        else:
            return self._forward_with_engine(**kwargs)

    def _forward_with_engine(self, **kwargs):
        # compute required tokens
        total_tokens = self.compute_required_tokens_graceful(self.content, count_context=False)
        if total_tokens is None:
            logger = get_logger()
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

        nest_asyncio = lazy_nest_asyncio()
        nest_asyncio.apply()
        loop = always_get_an_event_loop()
        logger = get_logger()
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
            logger = get_logger()
            logger.debug(
                "compute_required_tokens is not implemented for this engine, returning None"
            )
            return # Gracefully handle NotImplementedError; any other exception will be raised

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

    def _deduplicate_list_fields(self, res) -> object:
        """Deduplicate all list fields of a model instance in-place and return it."""
        for fname, finfo in res.model_fields.items():
            if getattr(finfo, "exclude", False):
                continue
            val = getattr(res, fname, None)
            if isinstance(val, list):
                deduped = self._deduplicate_list(val)
                if len(deduped) != len(val):
                    logger = get_logger()
                    logger.debug(
                        f"Deduplicated list field '{fname}' from {len(val)} -> {len(deduped)} items"
                    )
                setattr(res, fname, deduped)
        return res

    def _compress_string_field_recursive(self, text: str, field_name: str, attempt: int) -> str:
        """Recursively invoke HierarchicalSummary on a single string field and return (potentially) shorter result.

        Only replaces the field if the recursive result is not longer than the original.
        """
        logger = get_logger()
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
