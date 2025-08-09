from typing import Any, Callable, TypeVar, Optional, Union, Awaitable
import functools
import inspect
import time

T = TypeVar('T')

def fail_safe(default: Optional[Any] = None) -> Callable[[Callable[..., T]], Callable[..., Union[T, Any]]]:

    """
    Universal safety decorator that catches all exceptions in:
    - Regular functions
    - Generator functions
    - Async functions
    - Async generator functions
    
    Args:
        default: Value to return on any exception
        
    Returns:
        Decorated function that never raises exceptions
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Error in {func.__name__}: {str(e)}")
                return default

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Error in {func.__name__}: {str(e)}")
                return default

        @functools.wraps(func)
        def generator_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                yield from func(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Error in {func.__name__}: {str(e)}")
                yield default

        @functools.wraps(func)
        async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                async for item in func(*args, **kwargs):
                    yield item
            except Exception as e:
                print(f"⚠️ Error in {func.__name__}: {str(e)}")
                yield default

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        elif inspect.isasyncgenfunction(func):
            return async_gen_wrapper
        elif inspect.isgeneratorfunction(func):
            return generator_wrapper
        return sync_wrapper

    return decorator


def time_profile(print_output=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if print_output:
                print(f"⏱️ {func.__name__} processed by {elapsed:.6f} sec.")
            return result
        return wrapper
    return decorator

