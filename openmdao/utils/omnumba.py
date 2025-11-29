"""
Optionally import numba and provide a dummy jit decorator if it is not available.

If numba is not available, this module provides a dummy jit decorator that simply returns the
original function.
"""

try:
    from openmdao.utils.lazy_imports import numba  # noqa: F401
except ImportError:
    numba = None
    prange = range

    # If numba is not available, just write a dummy jit wrapper.
    # Code will still run at a significant performance hit.
    def jit(f, *args, **kwargs):
        """
        Return original function.

        Parameters
        ----------
        f : function
            The function to be decorated.
        """
        return f
