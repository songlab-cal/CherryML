import hashlib
import logging
import os
import sys
from inspect import signature
from typing import List


def _init_logger():
    logger = logging.getLogger("caching")
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()
logger = logging.getLogger("caching")

_CACHE_DIR = None
_USE_HASH = True
_HASH_LEN = None


def set_cache_dir(cache_dir: str):
    logger.info(f"Setting cache directory to: {cache_dir}")
    global _CACHE_DIR
    _CACHE_DIR = cache_dir


def get_cache_dir():
    global _CACHE_DIR
    return _CACHE_DIR


def set_log_level(log_level: int):
    logger = logging.getLogger("caching")
    logger.setLevel(level=log_level)


def set_use_hash(use_hash: bool):
    logger.info(f"Setting cache to use use_hash: {use_hash}")
    global _USE_HASH
    _USE_HASH = use_hash


def get_use_hash():
    global _USE_HASH
    return _USE_HASH


def set_hash_len(hash_len: int):
    if hash_len > 128:
        raise ValueError(
            "The maximum allowed hash length is 128. "
            f"You requested: {hash_len}"
        )
    logger.info(f"Setting cache to use hash length: {hash_len}")
    global _HASH_LEN
    _HASH_LEN = hash_len


def get_hash_len():
    global _HASH_LEN
    return _HASH_LEN


class CacheUsageError(Exception):
    pass


def _hash_all(xs: List[str]) -> str:
    hash_len = get_hash_len()
    hashes = [hashlib.sha512(x.encode("utf-8")).hexdigest() for x in xs]
    res = hashlib.sha512("".join(hashes).encode("utf-8")).hexdigest()
    if hash_len is None:
        return res
    else:
        return res[:hash_len]


def _get_func_caching_dir(
    func, unhashed_args: List[str], args, kwargs, cache_dir: str, use_hash: bool
) -> str:
    """
    Get caching directory for the given *function call*.

    The arguments in unhashed_args are not included in the cache key.
    """
    # Get the binding
    s = signature(func)
    binding = s.bind(*args, **kwargs)
    binding.apply_defaults()

    # Compute the cache key
    if not use_hash:
        path = (
            [cache_dir]
            + [f"{func.__name__}"]
            + [
                f"{key}_{val}"
                for (key, val) in binding.arguments.items()
                if key not in unhashed_args
            ]
        )
    else:
        path = (
            [cache_dir]
            + [f"{func.__name__}"]
            + [
                _hash_all(
                    sum(
                        [
                            [f"{key}", f"{val}"]
                            for (key, val) in binding.arguments.items()
                            if (key not in unhashed_args)
                        ],
                        [],
                    )
                )
            ]
        )
    func_caching_dir = os.path.join(*path)
    return func_caching_dir


def _validate_decorator_args(
    func,
    decorator_args: List[str],
) -> None:
    """
    Validate that the decorator arguments makes sense.

    Raises:
        CacheUsageError is any error is detected.
    """
    # Check that all arguments specified in the decorator are arguments of the
    # wrapped function - the user might have made a typo!
    func_parameters = list(signature(func).parameters.keys())
    for arg in decorator_args:
        if arg not in func_parameters:
            raise CacheUsageError(
                f"{arg} is not an argument to '{func.__name__}'. Fix the "
                f"arguments of the caching decorator."
            )

    # No argument can be repeated in the decorator
    if len(set(decorator_args)) != len(decorator_args):
        raise CacheUsageError(
            "All the function arguments specified in the caching decorator for "
            f"'{func.__name__}' should be distinct. You provided: "
            f"{decorator_args} "
        )


def _get_mode(path):
    """
    Get mode of a file (e.g. '664', '555')
    """
    return oct(os.stat(path).st_mode)[-3:]
