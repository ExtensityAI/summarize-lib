"""
Summarize-lib: Hierarchical document summarization for SymbolicAI

This package provides lazy-loaded classes and utilities for document summarization.
All heavy dependencies are loaded on-demand to improve startup performance.
"""

# Import classes directly for symai Import.load_expression() compatibility
from .hierarchical import HierarchicalSummary, Summary
from .functions import ValidatedFunction
from .types import DocumentType, TYPE_SPECIFIC_PROMPTS

# Also export lazy accessors for internal use
from .lazy_imports import (
    lazy_hierarchical_summary,
    lazy_summary,
    lazy_document_type,
    lazy_type_specific_prompts,
    lazy_validated_function
)

# Create lazy accessors for backward compatibility
def get_hierarchical_summary():
    """Get HierarchicalSummary class lazily."""
    return lazy_hierarchical_summary()

def get_summary():
    """Get Summary class lazily."""
    return lazy_summary()

def get_document_type():
    """Get DocumentType enum lazily."""
    return lazy_document_type()

def get_type_specific_prompts():
    """Get TYPE_SPECIFIC_PROMPTS dictionary lazily."""
    return lazy_type_specific_prompts()

def get_validated_function():
    """Get ValidatedFunction class lazily."""
    return lazy_validated_function()

__all__ = [
    'HierarchicalSummary',
    'Summary',
    'ValidatedFunction',
    'DocumentType',
    'TYPE_SPECIFIC_PROMPTS',
    'get_hierarchical_summary',
    'get_summary',
    'get_document_type',
    'get_type_specific_prompts',
    'get_validated_function'
]
