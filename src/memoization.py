"""
Thread-safe memoization system for the summarize-lib codebase.

This module provides a comprehensive memoization framework that ensures
thread safety while caching expensive operations like prompt generation,
token computation, and document processing.
"""

import threading
import time
import hashlib
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import OrderedDict
from weakref import WeakKeyDictionary


class ThreadSafeLRUCache:
    """
    Thread-safe Least Recently Used (LRU) cache implementation.

    Features:
    - Thread-safe operations using locks
    - LRU eviction policy
    - Configurable size limits
    - Memory-efficient with weak references
    - TTL (Time To Live) support
    """

    def __init__(self, max_size: int = 128, ttl: Optional[float] = None):
        """
        Initialize the thread-safe LRU cache.

        Args:
            max_size: Maximum number of items to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache."""
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._miss_count += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hit_count += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Put an item in the cache."""
        with self._lock:
            current_time = time.time()

            # Remove if already exists
            if key in self._cache:
                del self._cache[key]

            # Add new item
            self._cache[key] = (value, current_time)

            # Evict if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._hit_count = 0
            self._miss_count = 0

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class MemoizationManager:
    """
    Central manager for all memoization in the codebase.

    Provides thread-safe caching for various operations with
    automatic cache invalidation and memory management.
    """

    def __init__(self):
        """Initialize the memoization manager."""
        self._caches: Dict[str, ThreadSafeLRUCache] = {}
        self._lock = threading.RLock()

        # Create default caches
        self._create_default_caches()

    def _create_default_caches(self):
        """Create default caches for common operations."""
        # Prompt cache - large size, no TTL (prompts don't change often)
        self._caches['prompts'] = ThreadSafeLRUCache(max_size=256, ttl=None)

        # Token computation cache - medium size, short TTL
        self._caches['tokens'] = ThreadSafeLRUCache(max_size=128, ttl=300)  # 5 minutes

        # Document processing cache - small size, medium TTL
        self._caches['documents'] = ThreadSafeLRUCache(max_size=64, ttl=600)  # 10 minutes

        # Model computation cache - medium size, long TTL
        self._caches['models'] = ThreadSafeLRUCache(max_size=96, ttl=1800)  # 30 minutes

    def get_cache(self, name: str) -> ThreadSafeLRUCache:
        """Get a cache by name."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = ThreadSafeLRUCache()
            return self._caches[name]

    def clear_cache(self, name: Optional[str] = None):
        """Clear a specific cache or all caches."""
        with self._lock:
            if name is None:
                for cache in self._caches.values():
                    cache.clear()
            elif name in self._caches:
                self._caches[name].clear()

    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching a pattern."""
        with self._lock:
            for cache in self._caches.values():
                # Acquire cache lock once to avoid lock contention
                with cache._lock:
                    # Collect keys to remove while holding the lock
                    keys_to_remove = [key for key in cache._cache.keys() if pattern in key]

                    # Remove keys directly to avoid calling invalidate() which would acquire lock again
                    for key in keys_to_remove:
                        if key in cache._cache:  # Double-check inside lock
                            del cache._cache[key]

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        with self._lock:
            return {name: cache.stats() for name, cache in self._caches.items()}


# Global memoization manager instance
_memoization_manager = MemoizationManager()


def get_memoization_manager() -> MemoizationManager:
    """Get the global memoization manager."""
    return _memoization_manager


def _generate_cache_key(args, kwargs, key_func: Optional[Callable] = None) -> str:
    """Helper function to generate cache key consistently."""
    if key_func:
        return key_func(*args, **kwargs)
    else:
        # Default key generation
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()


def memoize(cache_name: str, key_func: Optional[Callable] = None, ttl: Optional[float] = None):
    """
    Decorator for memoizing function results.

    Args:
        cache_name: Name of the cache to use
        key_func: Function to generate cache key from arguments
        ttl: Time to live override for this function

    Returns:
        Decorated function with memoization
    """
    def decorator(func: Callable) -> Callable:
        cache = get_memoization_manager().get_cache(cache_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = _generate_cache_key(args, kwargs, key_func)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute result WITHIN the cache's lock to prevent race conditions
            with cache._lock:
                # Double-check after acquiring lock
                result = cache.get(key)
                if result is not None:
                    return result

                # Compute result
                result = func(*args, **kwargs)

                # Store in cache
                cache.put(key, result)

            return result

        # Add cache management methods to the wrapper
        wrapper._cache = cache
        wrapper._cache_name = cache_name

        def invalidate_cache(*args, **kwargs):
            """Invalidate cache for specific arguments."""
            key = _generate_cache_key(args, kwargs, key_func)
            cache.invalidate(key)

        wrapper.invalidate_cache = invalidate_cache
        wrapper.clear_cache = cache.clear

        return wrapper

    return decorator


def _generate_method_cache_key(self, args, kwargs, key_func: Optional[Callable] = None) -> str:
    """Helper function to generate cache key for methods consistently."""
    if key_func:
        return key_func(self, *args, **kwargs)
    else:
        # Default key generation including instance
        instance_id = id(self)
        key_data = f"{instance_id}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()


def memoize_method(cache_name: str, key_func: Optional[Callable] = None):
    """
    Decorator for memoizing instance methods.

    Args:
        cache_name: Name of the cache to use
        key_func: Function to generate cache key from instance and arguments

    Returns:
        Decorated method with memoization
    """
    def decorator(func: Callable) -> Callable:
        cache = get_memoization_manager().get_cache(cache_name)

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key including instance ID
            key = _generate_method_cache_key(self, args, kwargs, key_func)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute result WITHIN the cache's lock to prevent race conditions
            with cache._lock:
                # Double-check after acquiring lock
                result = cache.get(key)
                if result is not None:
                    return result

                # Compute result
                result = func(self, *args, **kwargs)

                # Store in cache
                cache.put(key, result)

            return result

        # Add cache management methods
        wrapper._cache = cache
        wrapper._cache_name = cache_name

        def invalidate_instance_cache(self, *args, **kwargs):
            """Invalidate cache for this instance."""
            key = _generate_method_cache_key(self, args, kwargs, key_func)
            cache.invalidate(key)

        wrapper.invalidate_instance_cache = invalidate_instance_cache

        return wrapper

    return decorator


def _generate_property_cache_key(self, func_name: str, key_func: Optional[Callable] = None) -> str:
    """Helper function to generate cache key for properties consistently."""
    if key_func:
        return key_func(self)
    else:
        return f"{id(self)}:{func_name}"


def memoize_property(cache_name: str, key_func: Optional[Callable] = None):
    """
    Decorator for memoizing properties.

    Args:
        cache_name: Name of the cache to use
        key_func: Function to generate cache key from instance

    Returns:
        Decorated property with memoization
    """
    def decorator(func: Callable) -> property:
        cache = get_memoization_manager().get_cache(cache_name)

        def getter(self):
            # Generate cache key
            key = _generate_property_cache_key(self, func.__name__, key_func)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute result WITHIN the cache's lock to prevent race conditions
            with cache._lock:
                # Double-check after acquiring lock
                result = cache.get(key)
                if result is not None:
                    return result

                # Compute result
                result = func(self)

                # Store in cache
                cache.put(key, result)

            return result

        def setter(self, value):
            # Invalidate cache when property is set
            key = _generate_property_cache_key(self, func.__name__, key_func)
            cache.invalidate(key)

            # Set the actual value
            if hasattr(self, f"_{func.__name__}"):
                setattr(self, f"_{func.__name__}", value)

        def deleter(self):
            # Invalidate cache when property is deleted
            key = _generate_property_cache_key(self, func.__name__, key_func)
            cache.invalidate(key)

            # Delete the actual value
            if hasattr(self, f"_{func.__name__}"):
                delattr(self, f"_{func.__name__}")

        return property(getter, setter, deleter)

    return decorator


# Utility functions for cache management
def clear_all_caches():
    """Clear all memoization caches."""
    get_memoization_manager().clear_cache()


def clear_cache(cache_name: str):
    """Clear a specific cache."""
    get_memoization_manager().clear_cache(cache_name)


def invalidate_pattern(pattern: str):
    """Invalidate all keys matching a pattern."""
    get_memoization_manager().invalidate_pattern(pattern)


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return get_memoization_manager().get_stats()


# Thread-safe singleton pattern for expensive computations
class ThreadSafeSingleton:
    """
    Thread-safe singleton pattern for expensive computations.
    """

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get_instance(self, key: str, factory: Callable[[], Any]) -> Any:
        """Get or create a singleton instance."""
        with self._lock:
            if key not in self._instances:
                self._instances[key] = factory()
            return self._instances[key]

    def clear(self):
        """Clear all instances."""
        with self._lock:
            self._instances.clear()


# Global singleton manager
_singleton_manager = ThreadSafeSingleton()


def get_singleton(key: str, factory: Callable[[], Any]) -> Any:
    """Get or create a singleton instance."""
    return _singleton_manager.get_instance(key, factory)
