"""Tokenizer utilities for fast token counting and caching."""
import threading
from typing import Dict, Optional

from loguru import logger
from symai.models import LLMDataModel

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

