import logging
import os
from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import List

from ._common import (CacheUsageError, _get_func_caching_dir, _get_mode,
                      _validate_decorator_args, get_cache_dir, get_read_only,
                      get_use_hash)

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))


def _get_func_binding(
    func,
    exclude_args: List[str],
    output_dirs: List[str],
    args,
    kwargs,
):
    """
    Note: does not exclude exclude_args_if_default, which is fine since this is
    used just for logging purposes and nothing else.
    """
    args = deepcopy(args)
    kwargs = deepcopy(kwargs)
    unhashed_args = exclude_args + output_dirs
    for unhashed_arg in unhashed_args:
        kwargs[unhashed_arg] = None
    s = signature(func)
    binding = s.bind(*args, **kwargs)
    binding.apply_defaults()
    return binding


def _get_func_caching_dir_aux(
    func,
    exclude_args: List[str],
    exclude_args_if_default: List[str],
    output_dirs: List[str],
    args,
    kwargs,
    cache_dir: str,
    use_hash: bool,
) -> str:
    """
    Get caching directory for the given *function call*.
    """
    args = deepcopy(args)
    kwargs = deepcopy(kwargs)

    # To bind the func call, we need to have all arguments specified. Since the
    # output dirs can be omitted, we just hackily assign them None. TODO: If
    # the output dirs are passed as args, this fails due to a `TypeError:
    # multiple values for argument`. The good news is that it fails noisily,
    # instead of causing a silent bug. So really not critical to address right
    # now.
    for output_dir in output_dirs:
        kwargs[output_dir] = None

    unhashed_args = exclude_args + output_dirs
    # Now add the exclude_args_if_default
    s = signature(func)
    binding = s.bind(*args, **kwargs)
    binding.apply_defaults()
    for (arg, val) in binding.arguments.items():
        if arg in exclude_args_if_default:
            p = s.parameters[arg]
            default = p.default
            if val == default:
                logger.info(f"Excluding {arg} because it has default value {val}.")
                unhashed_args.append(arg)

    return _get_func_caching_dir(
        func=func,
        unhashed_args=unhashed_args,
        args=args,
        kwargs=kwargs,
        cache_dir=cache_dir,
        use_hash=use_hash,
    )


def _maybe_write_usefull_stuff_cached_computation(
    func,
    exclude_args,
    exclude_args_if_default,
    output_dirs,
    args,
    kwargs,
    cache_dir,
    output_dir,
    write_extra_log_files,
):
    """
    Creates the logfiles _unhashed_output_dir.log and
    _function_binding.log which contain useful information
    about the exact function call (in case hashing was used
    to compute the output directory).

    The files are written if they already exist or their mode
    is not 444.
    """
    if not write_extra_log_files:
        return

    unhashed_func_caching_dir = _get_func_caching_dir_aux(
        func,
        exclude_args,
        exclude_args_if_default,
        output_dirs,
        args,
        kwargs,
        cache_dir,
        use_hash=False,
    )
    unhashed_func_caching_dir_logfile = os.path.join(
        kwargs[output_dir],
        "_unhashed_output_dir.log",
    )
    if not (
        os.path.exists(unhashed_func_caching_dir_logfile)
        and _get_mode(unhashed_func_caching_dir_logfile) == "444"
    ):
        open(unhashed_func_caching_dir_logfile, "w").write(
            unhashed_func_caching_dir
        )
        os.system(f'chmod 444 "{unhashed_func_caching_dir_logfile}"')

    binding = _get_func_binding(
        func,
        exclude_args,
        output_dirs,
        args,
        kwargs,
    )
    func_binding_logfile = os.path.join(
        kwargs[output_dir],
        "_function_binding.log",
    )
    if not (
        os.path.exists(func_binding_logfile)
        and _get_mode(func_binding_logfile) == "444"
    ):
        open(func_binding_logfile, "w").write(str(binding))
        os.system(f'chmod 444 "{func_binding_logfile}"')


def cached_computation(
    exclude_args: List = [],
    exclude_args_if_default: List = [],
    output_dirs: List = [],
    write_extra_log_files: bool = False,
):
    """
    Cache a function's outputs.

    The function must write its outputs to `output_dirs[i]/result.txt` for each
    i. The files `output_dirs[i]/result.txt` will be changed to mode 444 by the
    decorator to avoid data corruption, and upon successful execution a file
    `output_dirs[i]/result.success` will be created. This caching
    scheme ensures (in two redundant ways) that data is never corrupted. If an
    output file exists but is found to not have mode 444, or not have a
    corresponding success token, then the output will be considered corrupted
    and the computation will be performed.

    The computation performed by the function should be a function of all
    arguments to the function except for the provided `exclude_args`,
    and the `output_dirs`. This tells the decorator
    how to set the `output_dirs` to look for previously computed results.
    The only argument in the decorator that has some wiggle room is the
    `exclude_args`. Arguments that we usually want to specify here include
    things like the total number of CPU cores to use; whether to use CPU or GPU,
    how many OpenMP threads to use; how many MPI ranks to use; the verbosity
    level. The latter are all arguments related to parallelization or logging
    that the function does not depend on, so we exclude them. If they are not
    excluded, then e.g. running the same function with more CPU cores will
    result in all the computation being redone, which is ridiculous!

    Because the output directories are determined automatically by the
    decorator, there is not way to know what they are. Because of this, the
    decorated function will *return the output directories* when called. This
    allows seamlessly piecing together the computation performed by
    cached parallel functions. The return value is, more precisely, a dictionary
    mapping each element of `output_dirs` to its path.

    Args:
        exclude_args: Arguments which should be excluded when computing the
            cache key. For example, the number of processors to use.
        exclude_args_if_default: Arguments which should be excluded when
            computing the cache key, ONLY IF THEY TAKE THEIR DEFAULT VALUE.
            Good for backwards compatibility when extending the cached function
            with new arguments.
        output_dirs: The list of output directories of the wrapped function.
            Each value in the argument specified by parallel_arg creates one
            output in each directory specified by output_dirs.
        write_extra_log_files: If to write extra log files indicating e.g. the
            full function call. For functions that are called many times (e.g.
            metrics) with many arguments, the extra log files can clog up space,
            so it if useful to set this argument to `False`.

    Returns:
        The concrete caching decorator.
    """

    def cached_computation_decorator(func):
        """
        Concrete caching decorator.

        The concrete decorator resulting from the provided exclude_args,
        output_dirs.

        Args:
            func: The wrapped function.

        Returns:
            The wrapped function.
        """
        _validate_decorator_args(
            func, decorator_args=exclude_args + exclude_args_if_default + output_dirs
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only allow calling the function with kwargs for simplicity. TODO:
            # If I find a way to put all the args into kwargs, I can remove
            # this restriction from the user.
            if len(args) > 0:
                raise CacheUsageError(
                    f"Please call {func.__name__} with keyword arguments only. "
                    f"Positional arguments are not allowed for caching reasons."
                )

            # Get caching hyperparameters
            cache_dir = get_cache_dir()
            if cache_dir is None:
                return func(*args, **kwargs)
            use_hash = get_use_hash()

            # Compute function caching directory.
            func_caching_dir = _get_func_caching_dir_aux(
                func,
                exclude_args,
                exclude_args_if_default,
                output_dirs,
                args,
                kwargs,
                cache_dir,
                use_hash,
            )

            # Set the output dirs in kwargs based on the caching directory if
            # not already provided.
            for output_dir in output_dirs:
                if output_dir not in kwargs or kwargs[output_dir] is None:
                    kwargs[output_dir] = os.path.join(
                        func_caching_dir, output_dir
                    )

            # We will return the output directories. We get them right now in
            # case these get modified by the call.
            res = {output_dir: kwargs[output_dir] for output_dir in output_dirs}

            def computed():
                # Check that each of the output files exists
                for output_dir in output_dirs:
                    output_filepath = os.path.join(
                        kwargs[output_dir], "result.txt"
                    )

                    if not os.path.exists(output_filepath):
                        return False

                    # COMMENTED OUT: Don't be so picky about the file mode.
                    # mode = _get_mode(output_filepath)
                    # if mode != "444":
                    #     return False

                    output_success_token_filepath = os.path.join(
                        kwargs[output_dir], "result.success"
                    )
                    if not os.path.exists(output_success_token_filepath):
                        return False
                return True

            def clear_previous_outputs():
                for output_dir in output_dirs:
                    output_filepath = os.path.join(
                        kwargs[output_dir], "result.txt"
                    )
                    if os.path.exists(output_filepath):
                        logger.info(
                            f"Removing possibly corrupted {output_filepath}"
                        )
                        os.system(f'chmod 666 "{output_filepath}"')
                        os.remove(output_filepath)

                    output_success_token_filepath = os.path.join(
                        kwargs[output_dir], "result.success"
                    )
                    if os.path.exists(output_success_token_filepath):
                        logger.info(f"Removing {output_success_token_filepath}")
                        os.system(
                            f'chmod 666 "{output_success_token_filepath}"'
                        )
                        os.remove(output_success_token_filepath)

            # Make sure that all the output directories exist.
            for output_dir in output_dirs:
                if not os.path.exists(kwargs[output_dir]):
                    os.umask(0)  # To ensure all collaborators can access cache
                    os.makedirs(kwargs[output_dir], mode=0o777)
                    # Let's write some useful stuff to the directory
                _maybe_write_usefull_stuff_cached_computation(
                    func,
                    exclude_args,
                    exclude_args_if_default,
                    output_dirs,
                    args,
                    kwargs,
                    cache_dir,
                    output_dir,
                    write_extra_log_files,
                )

            # Only call the function if there is any work to do at all.
            if not computed():
                # Now call the wrapped function
                logger.debug(
                    f"Calling {func.__name__} . Output location: {func_caching_dir}"
                )
                if get_read_only():
                    raise CacheUsageError(
                        "Cache is in read only mode! Will not call function."
                    )
                clear_previous_outputs()
                func(*args, **kwargs)

                # Now verify that all outputs are there.
                for output_dir in output_dirs:
                    output_filepath = os.path.join(
                        kwargs[output_dir], "result.txt"
                    )
                    if not os.path.exists(output_filepath):
                        raise CacheUsageError(
                            f"function {func.__name__} should have created "
                            f"and written output to {output_filepath} but "
                            "the file does not exist."
                        )

                # Now chmod 444 all the output files and create the success
                # tokens for as redundant verification.
                for output_dir in output_dirs:
                    output_filepath = os.path.join(
                        kwargs[output_dir], "result.txt"
                    )
                    os.system(f'chmod 444 "{output_filepath}"')
                    output_success_token_filepath = os.path.join(
                        kwargs[output_dir], "result.success"
                    )
                    with open(output_success_token_filepath, "w") as f:
                        f.write("SUCCESS\n")
                        f.flush()
            return res

        return wrapper

    return cached_computation_decorator
