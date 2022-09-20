import logging
import os
import pickle
from functools import wraps
from inspect import signature
from typing import List, Optional

from ._common import CacheUsageError, _hash_all, get_cache_dir, get_use_hash

logger = logging.getLogger("caching")


def cached(
    exclude: Optional[List] = None,
    exclude_if_default: Optional[List] = None,
):
    """
    Cache the wrapped function.
    The outputs of the wrapped function are cached by pickling them into a
    file whose path is determined by the function's name and it's arguments.
    By default all arguments are used to define the cache key, but they can be
    excluded via the `exclude` list. Moreover, an argument can be excluded
    *only* if it has default values via the `exclude_if_default` list.
    Args:
        exclude: What arguments to exclude from the hash key. E.g.
            n_processes, which does not affect the result of the function.
            If None, nothing will be excluded.
        exclude_if_default: What arguments to exclude from the hash key *if
            default*. E.g. new arguments introduced to a function as it is
            extended with new functionality, while preserving the old
            behavior when it has default values.
    """

    def f_res(func):
        @wraps(func)
        def r_res_res(*args, **kwargs):
            cache_dir = get_cache_dir()
            if cache_dir is None:
                return func(*args, **kwargs)
            use_hash = get_use_hash()

            s = signature(func)
            # Check that all excluded args are present -
            # user might have a typo!
            if exclude is not None:
                for arg in exclude:
                    if arg not in s.parameters:
                        raise CacheUsageError(
                            f"{arg} is not an argument to {func.__name__}. "
                            "Fix the arguments in `exclude`."
                        )
            # Check that all exclude_if_default args are present -
            # user might have a typo!
            if exclude_if_default is not None:
                for arg in exclude_if_default:
                    if arg not in s.parameters:
                        raise CacheUsageError(
                            f"{arg} is not an argument to {func.__name__}. "
                            "Fix the arguments in `exclude_if_default`."
                        )

            # None of the exclude_if_default args can be a prefix of another one
            # (otherwise adversarial hash collisions can be easily crafted due
            # to how args are concatenated and hashed)
            if exclude_if_default is not None:
                for arg1 in exclude_if_default:
                    for arg2 in exclude_if_default:
                        if arg1 != arg2 and arg2.startswith(arg1):
                            raise CacheUsageError(
                                "None of the exclude_if_default arguments can "
                                "be a prefix of another exclude_if_default "
                                "argument. This ensures that it is not "
                                "possible to craft adversarial hash collisions."
                            )

            def excluded(arg, val) -> bool:
                if exclude is not None and arg in exclude:
                    return True
                if exclude_if_default is not None and arg in exclude_if_default:
                    p = s.parameters[arg]
                    default = p.default
                    if val == default:
                        return True
                return False

            binding = s.bind(*args, **kwargs)
            binding.apply_defaults()
            if not use_hash:
                path = (
                    [cache_dir]
                    + [f"{func.__name__}"]
                    + [
                        f"{arg}_{val}"
                        for (arg, val) in binding.arguments.items()
                        if (not excluded(arg, val))
                    ]
                    + ["result"]
                )
            else:
                path = (
                    [cache_dir]
                    + [f"{func.__name__}"]
                    + [
                        _hash_all(
                            sum(
                                [
                                    [f"{arg}", f"{val}"]
                                    for (arg, val) in binding.arguments.items()
                                    if not excluded(arg, val)
                                ],
                                [],
                            )
                        )
                    ]
                    + ["result"]
                )
            success_token_filename = os.path.join(*path) + ".success"
            filename = os.path.join(*path) + ".pickle"
            if os.path.isfile(success_token_filename):
                if not os.path.isfile(filename):
                    raise Exception(
                        "Success token is present but file is missing!: "
                        f"{filename}"
                    )
                with open(filename, "rb") as f:
                    try:
                        return pickle.load(f)
                    except Exception:
                        raise Exception(
                            "Corrupt cache file due to unpickling error, even "
                            f"though success token was present!: {filename}"
                        )
            else:
                if os.path.isfile(filename):
                    assert not os.path.isfile(
                        success_token_filename
                    )  # Should be true because we are in the else statement
                    logger.info(
                        "Success token missing but pickle file is present. "
                        "Thus pickle file is most likely corrupt. "
                        f"Will have to recompute: {filename}"
                    )
                logger.debug(f"Calling {func.__name__}")
                res = func(*args, **kwargs)
                os.makedirs(os.path.join(*path[:-1]), exist_ok=True)
                with open(filename, "wb") as f:
                    pickle.dump(res, f)
                    f.flush()
                with open(success_token_filename, "w") as f:
                    f.write("SUCCESS\n")
                    f.flush()
                return res

        return r_res_res

    return f_res
