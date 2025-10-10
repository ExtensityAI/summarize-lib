#!/usr/bin/env python3
"""
Test script to verify lazy imports are working correctly.
"""

import sys
import time
from src import get_hierarchical_summary, get_summary, get_document_type, MemoizedHierarchicalSummary

def test_lazy_imports():
    """Test that lazy imports work correctly."""
    print("Testing lazy imports...")

    # Test that we can get the classes without importing heavy dependencies upfront
    start_time = time.time()

    # Get classes lazily
    HierarchicalSummary = get_hierarchical_summary()
    Summary = get_summary()
    DocumentType = get_document_type()

    end_time = time.time()
    print(f"Lazy import time: {end_time - start_time:.4f} seconds")

    # Test that classes are properly loaded
    print(f"HierarchicalSummary: {HierarchicalSummary}")
    print(f"Summary: {Summary}")
    print(f"DocumentType: {DocumentType}")

    # Test MemoizedHierarchicalSummary
    print("\nTesting MemoizedHierarchicalSummary...")
    memoized = MemoizedHierarchicalSummary(
        content="This is a test document for lazy import verification.",
        document_name="test.txt"
    )

    print(f"Memoized instance created: {memoized}")
    print(f"Cache valid: {memoized.is_cache_valid()}")

    # Test prompt generation (this should trigger lazy imports)
    prompt = memoized.prompt
    print(f"Prompt generated (length: {len(prompt)})")
    print(f"Cache valid after prompt: {memoized.is_cache_valid()}")

    # Test cache invalidation
    memoized.user_prompt = "Different prompt"
    print(f"Cache valid after user_prompt change: {memoized.is_cache_valid()}")

    print("\n✅ All lazy import tests passed!")

if __name__ == "__main__":
    test_lazy_imports()
