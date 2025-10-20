#!/usr/bin/env python3
"""
Simple test script for the new token counting optimizations.
"""

from .hierarchical import get_current_tokenizer
from .memoization import get_memoization_manager

def test_tokenizer_cache_monitoring():
    """Test tokenizer cache size monitoring."""
    print("=== Testing Tokenizer Cache Monitoring ===")

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

def test_fast_path_token_counting():
    """Test the fast-path token counting functionality."""
    print("\n=== Testing Fast-Path Token Counting ===")

    # Test the fast-path function directly
    from .hierarchical import HierarchicalSummary

    # Create a minimal instance for testing
    class TestSummarizer:
        def __init__(self):
            self.tokenizer_name = "gpt2"
            self._token_offset = 0
            self._total_tokens_processed = 0

        def _log_token_usage(self, tokens_added, operation="processing"):
            self._total_tokens_processed += tokens_added
            print(f"Token usage - {operation}: +{tokens_added} tokens "
                  f"(total processed: {self._total_tokens_processed}, offset: {self._token_offset})")

        def get_token_stats(self):
            return {
                'total_processed': self._total_tokens_processed,
                'offset': self._token_offset,
                'net_tokens': self._total_tokens_processed - self._token_offset
            }

    # Add the fast-path method to the test class
    test_instance = TestSummarizer()
    test_instance._fast_path_token_count = HierarchicalSummary._fast_path_token_count.__get__(test_instance, TestSummarizer)

    # Test fast-path counting
    text = "Hello world! This is a test sentence."
    token_count = test_instance._fast_path_token_count(text)
    print(f"Fast-path token count for '{text}': {token_count}")

    # Show token statistics
    stats = test_instance.get_token_stats()
    print(f"Token statistics: {stats}")

def main():
    """Run all tests."""
    print("Testing Token Counting Optimizations in Summarize-Lib")
    print("=" * 60)

    try:
        # Test tokenizer cache monitoring
        test_tokenizer_cache_monitoring()

        # Test symai tokenizer access
        test_symai_tokenizer_access()

        # Test fast-path token counting
        test_fast_path_token_counting()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()