"""
Caching decorator module.

The caching decorator module provides decorators to wrap functions such that
any previously done computation is avoided in subsequent function calls. The
caching is performed directly in the filesystem (as opposed to in-memory),
enabling persistent caching. This is fundamental for compute-intensive
applications.
"""
from ._cached import cached
from ._cached_computation import cached_computation
from ._cached_parallel_computation import (cached_parallel_computation,
                                           secure_parallel_output)
from ._common import (set_cache_dir, set_dir_levels, set_hash_len,
                      set_log_level, set_read_only, set_use_hash)
