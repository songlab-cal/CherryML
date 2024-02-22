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


def secure_parallel_output(output_dir: str, parallel_arg: str) -> None:
    os.system(f"chmod 444 {os.path.join(output_dir, parallel_arg + '.txt')}")
    open(os.path.join(output_dir, parallel_arg + ".success"), "w").write(
        "SUCCESS\n"
    )


def _get_parallel_func_binding(
    func,
    exclude_args: List[str],
    parallel_arg: str,
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
    unhashed_args = exclude_args + [parallel_arg] + output_dirs
    for unhashed_arg in unhashed_args:
        kwargs[unhashed_arg] = None
    s = signature(func)
    binding = s.bind(*args, **kwargs)
    binding.apply_defaults()
    return binding


def _get_parallel_func_caching_dir_aux(
    func,
    exclude_args: List[str],
    exclude_args_if_default: List[str],
    parallel_arg: str,
    output_dirs: List[str],
    args,
    kwargs,
    cache_dir: str,
    use_hash: bool,
) -> str:
    """
    Get caching directory for the given *parallel function call*.
    """
    args = deepcopy(args)
    kwargs = deepcopy(kwargs)

    # To bind the parallel func call, we need to have all arguments specified.
    # Since the output dirs can be omitted, we just hackily assign them None.
    # TODO: If the output dirs are passed as args, this fails due to a
    # `TypeError: multiple values for argument`. The good news is that it fails
    # noisily, instead of causing a silent bug. So really not critical to
    # address right now.
    for output_dir in output_dirs:
        kwargs[output_dir] = None

    unhashed_args = exclude_args + [parallel_arg] + output_dirs
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


def _maybe_write_usefull_stuff_cached_parallel_computation(
    func,
    exclude_args,
    exclude_args_if_default,
    parallel_arg,
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

    unhashed_func_caching_dir = _get_parallel_func_caching_dir_aux(
        func,
        exclude_args,
        exclude_args_if_default,
        parallel_arg,
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

    binding = _get_parallel_func_binding(
        func,
        exclude_args,
        parallel_arg,
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


def cached_parallel_computation(
    parallel_arg: str,
    exclude_args: List = [],
    exclude_args_if_default: List = [],
    output_dirs: List = [],
    write_extra_log_files: bool = False,
):
    """
    Cache a parallel function's outputs.

    A 'parallel function' is one that does the same computation for different
    data, where the different data is specified by the `parallel_arg`. The API
    that such a function much respect is that for each value
    'parallel_arg_value' of the `parallel_arg`, the function writes its outputs
    to `output_dirs[i]/{parallel_arg_value}.txt` for each i.
    If this API is satisfied, the function can be decorated with
    this caching decorator and called *without specifying the `output_dirs`*.
    When this is done, the `output_dirs` will be determined *by the decorator*
    abd subsequent calls to the function will avoid
    re-doing any computation that has already been done. This is achieved
    by calling the wrapped function only for those values of the `parallel_arg`
    that have not already been computed. The files
    `output_dirs[i]/{parallel_arg_value}.txt` will be changed to mode 444 by the
    decorator to avoid data corruption, and upon successful execution a file
    `output_dirs[i]/{parallel_arg_value}.success` will be created. This caching
    scheme ensures (in two redundant ways) that data is never corrupted. If an
    output file exists but is found to not have mode 444, or not have a
    corresponding success token, then the output will be considered corrupted
    and the computation will be run for this 'parallel_arg_value'.
    The decorated function can also change the file's mode to 444 and create the
    success token if it wants, which might be useful when the decorated function
    is doing very large amounts of computation and we don't want it to finish
    generating all its outputs before marking the output files as successfully
    generated.

    The computation performed by the function should be a function of all
    arguments to the function except for the provided `exclude_args`,
    the `parallel_arg`, and the `output_dirs`. This tells the decorator
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

    The parallel_arg list can contain repeated values. In this case, the
    duplicated values (beyond the first) will be ignored. This is convenient
    when e.g. boostrapping, which involves sampling with replacement; we don't
    want to run into race conditions because of the parallel_arg containing
    repeated values!

    Args:
        exclude_args: Arguments which should be excluded when computing the
            cache key. For example, the number of processors to use.
         exclude_args_if_default: Arguments which should be excluded when
            computing the cache key, ONLY IF THEY TAKE THEIR DEFAULT VALUE.
            Good for backwards compatibility when extending the cached function
            with new arguments.
        parallel_arg: The argument over which computation is parallelized.
            This argument should be a list, and for each value in this list,
            an output is generated in each output directory.
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

    def cached_parallel_computation_decorator(func):
        """
        Concrete caching decorator.

        The concrete decorator resulting from the provided exclude_args,
        parallel_arg, output_dirs.

        Args:
            func: The wrapped function.

        Returns:
            The wrapped function.
        """
        _validate_decorator_args(
            func, decorator_args=exclude_args + exclude_args_if_default + [parallel_arg] + output_dirs
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

            # Remove duplicated values of the parallel arg
            kwargs[parallel_arg] = sorted(list(set(kwargs[parallel_arg])))

            # Get caching hyperparameters
            cache_dir = get_cache_dir()
            if cache_dir is None:
                return func(*args, **kwargs)
            use_hash = get_use_hash()

            # Compute function caching directory.
            func_caching_dir = _get_parallel_func_caching_dir_aux(
                func,
                exclude_args,
                exclude_args_if_default,
                parallel_arg,
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

            def computed(parallel_arg_value):
                # Check that each of the output files exists
                for output_dir in output_dirs:
                    output_filepath = os.path.join(
                        kwargs[output_dir], parallel_arg_value + ".txt"
                    )

                    if not os.path.exists(output_filepath):
                        return False

                    # COMMENTED OUT: Don't be so picky about the file mode.
                    # mode = _get_mode(output_filepath)
                    # if mode != "444":
                    #     return False

                    output_success_token_filepath = os.path.join(
                        kwargs[output_dir], parallel_arg_value + ".success"
                    )
                    if not os.path.exists(output_success_token_filepath):
                        return False
                return True

            def clear_previous_outputs(parallel_arg_value):
                for output_dir in output_dirs:
                    output_filepath = os.path.join(
                        kwargs[output_dir], parallel_arg_value + ".txt"
                    )
                    if os.path.exists(output_filepath):
                        logger.info(
                            f"Removing possibly corrupted {output_filepath}"
                        )
                        os.system(f'chmod 666 "{output_filepath}"')
                        os.remove(output_filepath)

                    output_success_token_filepath = os.path.join(
                        kwargs[output_dir], parallel_arg_value + ".success"
                    )
                    if os.path.exists(output_success_token_filepath):
                        logger.info(f"Removing {output_success_token_filepath}")
                        os.system(
                            f'chmod 666 "{output_success_token_filepath}"'
                        )
                        os.remove(output_success_token_filepath)

            # We will only call the function on the values that have not
            # already been computed: these are the 'new_parallel_args'
            new_parallel_args = []
            for parallel_arg_value in kwargs[parallel_arg]:
                if not computed(parallel_arg_value):
                    new_parallel_args.append(parallel_arg_value)
            # Replace the parallel_arg by the new_parallel_args
            kwargs[parallel_arg] = new_parallel_args

            # Make sure that all the output directories exist.
            for output_dir in output_dirs:
                if not os.path.exists(kwargs[output_dir]):
                    os.umask(0)  # To ensure all collaborators can access cache
                    os.makedirs(kwargs[output_dir], mode=0o777)
                    # Let's write some useful stuff to the directory
                _maybe_write_usefull_stuff_cached_parallel_computation(
                    func,
                    exclude_args,
                    exclude_args_if_default,
                    parallel_arg,
                    output_dirs,
                    args,
                    kwargs,
                    cache_dir,
                    output_dir,
                    write_extra_log_files,
                )

            # Only call the function if there is any work to do at all.
            if len(new_parallel_args):
                # Now call the wrapped function
                logger.debug(
                    f"Calling {func.__name__} . Output location: {func_caching_dir}"
                )
                if get_read_only():
                    raise CacheUsageError(
                        "Cache is in read only mode! Will not call function."
                    )
                for parallel_arg_value in new_parallel_args:
                    # Need to clear up all the previous partial outputs!
                    clear_previous_outputs(parallel_arg_value)
                func(*args, **kwargs)

                # Now verify that all outputs are there.
                for output_dir in output_dirs:
                    for parallel_arg_value in kwargs[parallel_arg]:
                        output_filepath = os.path.join(
                            kwargs[output_dir], parallel_arg_value + ".txt"
                        )
                        if not os.path.exists(output_filepath):
                            raise CacheUsageError(
                                f"function {func.__name__} should have created "
                                f"and written output to {output_filepath} but "
                                "the file does not exist."
                            )

                # Now chmod 444 all the output files and create the success
                # tokens for as redundant verification.
                # Note that the decorated function might have done this
                # already for more efficient crash recovery, but it doesn't
                # matter: we just re-do it anyway.
                for output_dir in output_dirs:
                    for parallel_arg_value in kwargs[parallel_arg]:
                        output_filepath = os.path.join(
                            kwargs[output_dir], parallel_arg_value + ".txt"
                        )
                        os.system(f'chmod 444 "{output_filepath}"')
                        output_success_token_filepath = os.path.join(
                            kwargs[output_dir], parallel_arg_value + ".success"
                        )
                        with open(output_success_token_filepath, "w") as f:
                            f.write("SUCCESS\n")
                            f.flush()
            return res

        return wrapper

    return cached_parallel_computation_decorator
