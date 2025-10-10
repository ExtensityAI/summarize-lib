"""
Lazy import utilities to avoid circular imports and improve startup performance.

This module provides lazy loading for heavy dependencies and classes that may cause
circular import issues. All imports are deferred until first access to improve
startup time and reduce memory usage.
"""

from typing import Type, Any, Optional
import functools


class LazyImport:
    """
    A descriptor that lazily imports modules or classes on first access.

    Usage:
        numpy = LazyImport('numpy')
        # Later: numpy.array([1, 2, 3])  # Import happens here
    """

    def __init__(self, module_name: str, attribute: Optional[str] = None):
        """
        Initialize lazy import.

        Args:
            module_name: The module to import (e.g., 'numpy', 'symai.components')
            attribute: Optional specific attribute to import (e.g., 'Function')
        """
        self.module_name = module_name
        self.attribute = attribute
        self._module = None
        self._attribute_obj = None

    def __get__(self, instance, owner):
        """Get the imported module or attribute."""
        if self._module is None:
            self._module = __import__(self.module_name, fromlist=[self.attribute] if self.attribute else [])

        if self.attribute:
            if self._attribute_obj is None:
                self._attribute_obj = getattr(self._module, self.attribute)
            return self._attribute_obj

        return self._module


def lazy_import(module_name: str, attribute: Optional[str] = None):
    """
    Create a lazy import function.

    Args:
        module_name: The module to import
        attribute: Optional specific attribute to import

    Returns:
        Function that returns the imported module/attribute on first call
    """
    @functools.lru_cache(maxsize=1)
    def _import():
        module = __import__(module_name, fromlist=[attribute] if attribute else [])
        if attribute:
            return getattr(module, attribute)
        return module

    return _import


# Heavy external dependencies
lazy_numpy = lazy_import('numpy')
lazy_loguru = lazy_import('loguru')
lazy_pydantic = lazy_import('pydantic')
lazy_symai = lazy_import('symai')
lazy_symai_components = lazy_import('symai.components')
lazy_symai_models = lazy_import('symai.models')
lazy_symai_core_ext = lazy_import('symai.core_ext')
lazy_tenacity = lazy_import('tenacity')
lazy_tiktoken = lazy_import('tiktoken')
lazy_tokenizers = lazy_import('tokenizers')
lazy_nest_asyncio = lazy_import('nest_asyncio')

# Specific classes and functions
lazy_field = lazy_import('pydantic', 'Field')
lazy_field_validator = lazy_import('pydantic', 'field_validator')
lazy_validation_error = lazy_import('pydantic', 'ValidationError')
lazy_llm_data_model = lazy_import('symai.models', 'LLMDataModel')
lazy_function = lazy_import('symai.components', 'Function')
lazy_file_reader = lazy_import('symai.components', 'FileReader')
lazy_dynamic_engine = lazy_import('symai.components', 'DynamicEngine')
lazy_symbol = lazy_import('symai', 'Symbol')
lazy_import_class = lazy_import('symai', 'Import')
lazy_bind = lazy_import('symai.core_ext', 'bind')

# Tenacity components
lazy_before_sleep_log = lazy_import('tenacity', 'before_sleep_log')
lazy_retry = lazy_import('tenacity', 'retry')
lazy_retry_if_exception_type = lazy_import('tenacity', 'retry_if_exception_type')
lazy_stop_after_attempt = lazy_import('tenacity', 'stop_after_attempt')
lazy_wait_exponential_jitter = lazy_import('tenacity', 'wait_exponential_jitter')

# Tokenizer components
lazy_encoding = lazy_import('tiktoken', 'Encoding')

# Internal modules
def lazy_hierarchical_summary() -> Type:
    """
    Lazy import of HierarchicalSummary class to avoid circular imports.

    Returns:
        The HierarchicalSummary class
    """
    from .hierarchical import HierarchicalSummary
    return HierarchicalSummary


def lazy_validated_function() -> Type:
    """
    Lazy import of ValidatedFunction class to avoid circular imports.

    Returns:
        The ValidatedFunction class
    """
    from .functions import ValidatedFunction
    return ValidatedFunction


def lazy_summary() -> Type:
    """
    Lazy import of Summary class to avoid circular imports.

    Returns:
        The Summary class
    """
    from .hierarchical import Summary
    return Summary


def lazy_document_type() -> Type:
    """
    Lazy import of DocumentType enum to avoid circular imports.

    Returns:
        The DocumentType enum
    """
    from .types import DocumentType
    return DocumentType


def lazy_type_specific_prompts() -> dict:
    """
    Lazy import of TYPE_SPECIFIC_PROMPTS to avoid circular imports.

    Returns:
        The TYPE_SPECIFIC_PROMPTS dictionary
    """
    from .types import TYPE_SPECIFIC_PROMPTS
    return TYPE_SPECIFIC_PROMPTS


def lazy_chonkie_chunker() -> Type:
    """
    Lazy import of ChonkieChunker to avoid heavy dependency loading.

    Returns:
        The ChonkieChunker class
    """
    Import = lazy_import_class()
    return Import.load_expression("ExtensityAI/chonkie-symai", "ChonkieChunker")


# Utility functions for common patterns
def get_logger():
    """Get the loguru logger lazily."""
    return lazy_loguru().logger


def get_random_state(seed: int = 42):
    """Get numpy RandomState lazily."""
    numpy = lazy_numpy()
    return numpy.random.RandomState(seed=seed)


def get_int16_max():
    """Get numpy int16 max value lazily."""
    numpy = lazy_numpy()
    return numpy.iinfo(numpy.int16).max
