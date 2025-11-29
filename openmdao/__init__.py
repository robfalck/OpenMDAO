import os
import sys


__version__ = '3.41.1-dev'

INF_BOUND = 1.0E30


if 'jax' not in sys.modules:
    # Safe to set env var - JAX not loaded yet
    os.environ.setdefault('JAX_ENABLE_X64', 'true')
elif os.environ.get('JAX_ENABLE_X64', '').lower() not in ('1', 'true'):
    # JAX already loaded without x64 - warn user
    import warnings
    warnings.warn(
        "JAX was imported before OpenMDAO without x64 precision enabled. "
        "For full precision, set JAX_ENABLE_X64=True before importing any packages, "
        "or import OpenMDAO before other JAX-dependent packages.",
        UserWarning
    )
