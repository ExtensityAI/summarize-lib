#!/usr/bin/env python3
"""
Test script demonstrating the new token counting optimizations in summarize-lib.

This script shows:
1. Fast-path token counting that bypasses Function(preview=...) when only counting
2. Tokenizer cache size monitoring and warnings
3. Token usage logging and tracking
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the modules properly
try:
    from src.hierarchical import HierarchicalSummary, get_current_tokenizer
    from src.memoization import get_memoization_manager
except ImportError:
    # Fallback for direct execution
    import hierarchical
    import memoization
    HierarchicalSummary = hierarchical.HierarchicalSummary
    get_current_tokenizer = hierarchical.get_current_tokenizer
    get_memoization_manager = memoization.get_memoization_manager

def test_fast_path_token_counting():
    """Test the fast-path token counting functionality."""
    print("=== Testing Fast-Path Token Counting ===")

    # Create a simple summarizer instance
    summarizer = HierarchicalSummary(
        content="This is a test document for token counting optimization.",
        tokenizer_name="gpt2"
    )

    # Test fast-path counting (count_context=False)
    print("Testing fast-path token counting (count_context=False)...")
    tokens_fast = summarizer.compute_required_tokens("Hello world!", count_context=False)
    print(f"Fast-path token count: {tokens_fast}")

    # Test regular counting (count_context=True) - will use Function preview
    print("Testing regular token counting (count_context=True)...")
    tokens_regular = summarizer.compute_required_tokens("Hello world!", count_context=True)
    print(f"Regular token count: {tokens_regular}")

    # Show token statistics
    stats = summarizer.get_token_stats()
    print(f"Token statistics: {stats}")

    return summarizer

def test_tokenizer_cache_monitoring():
    """Test tokenizer cache size monitoring."""
    print("\n=== Testing Tokenizer Cache Monitoring ===")

    manager = get_memoization_manager()
    tokenizer_cache = manager.get_cache('tokenizers')

    print(f"Initial tokenizer cache size: {len(tokenizer_cache._cache)}")

    # Simulate adding items to trigger warnings
    print("Adding items to tokenizer cache...")
    for i in range(5):
        tokenizer_cache.put(f"tokenizer_{i}", f"tokenizer_instance_{i}")
        print(f"Cache size after adding item {i}: {len(tokenizer_cache._cache)}")

    # Show cache statistics
    stats = tokenizer_cache.stats()
    print(f"Tokenizer cache statistics: {stats}")

def test_symai_tokenizer_access():
    """Test symai tokenizer access."""
    print("\n=== Testing SymAI Tokenizer Access ===")

    tokenizer = get_current_tokenizer()
    if tokenizer:
        print("Successfully retrieved tokenizer from symai")
        try:
            # Test tokenization
            text = "This is a test sentence for tokenization."
            tokens = tokenizer.encode(text)
            print(f"Text: '{text}'")
            print(f"Tokens: {tokens}")
            print(f"Token count: {len(tokens)}")

            # Test decoding
            decoded = tokenizer.decode(tokens)
            print(f"Decoded: '{decoded}'")
        except Exception as e:
            print(f"Error during tokenization: {e}")
    else:
        print("No tokenizer available from symai (this is normal if symai is not configured)")

def main():
    """Run all tests."""
    print("Testing Token Counting Optimizations in Summarize-Lib")
    print("=" * 60)

    try:
        # Test fast-path token counting
        summarizer = test_fast_path_token_counting()

        # Test tokenizer cache monitoring
        test_tokenizer_cache_monitoring()

        # Test symai tokenizer access
        test_symai_tokenizer_access()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
