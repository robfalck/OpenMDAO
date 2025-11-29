"""
Provide a lazy importing capability to improve import speeds of the OpenMDAO package.

In general, one should delay the import of specialized packages and modules until
they are needed, like IPython or pyDOE3.

This typical approach to lazy loading breaks down when a module is needed within
a function in a performance sensitive area, or due to legacy usages such as testing
for the availbility of MPI with `if MPI is not None`.

In these cases, this module lets those packages be lazily loaded at a module-level.

For instance:

    from openmdao.utils.lazy_imports import IPython

This will return an instance of LazyImport and will not actually perform the
import of IPython until an attribute of IPython is accessed:

    display = IPython.display

The LazyImport object will efficiently test for the availability of the given
package/module using importlib.util.find_spec(module_name) rather than
actually importing it. Therefore, if the package is not available,
the import will quickly raise ModuleNotFoundError, preserving the existing
behavior.
"""

import importlib.util


class LazyImport:
    """
    Lazy loader for a module that only imports on attribute access.

    Checks if the module is available without actually importing it.
    Raises ImportError immediately if the module is not installed,
    allowing try/except ImportError patterns to work as expected.

    Parameters
    ----------
    module_name : str
        Name of the module to import (e.g., 'jax', 'mpi4py.MPI').

    Attributes
    ----------
    _module_name : str
        The name of the module to be lazily loaded.
    _module : Python module or None
        The lazily imported module if it has been loaded, otherwise None.

    Raises
    ------
    ModuleNotFoundError
        If the module is not available for import.
    """

    def __init__(self, module_name):
        """Initialize the LazyImport wrapper."""
        self._module_name = module_name
        base_package = module_name.split('.')[0]
        self._module = None

        if importlib.util.find_spec(base_package) is None:
            raise ModuleNotFoundError(f"No module named '{base_package}'")

    def _load(self):
        """
        Actually import the module if not already imported.

        This is called lazily on first attribute access, not at
        initialization time, which avoids the import overhead until
        the module is actually used.

        Returns
        -------
        module
            The imported module.
        """
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name):
        """
        Intercept attribute access to trigger lazy import.

        This is called whenever an attribute is accessed on the lazy
        import object. It ensures the module is loaded and then
        delegates the attribute lookup to the actual module.

        Parameters
        ----------
        name : str
            Name of the attribute being accessed.

        Returns
        -------
        any
            The requested attribute from the imported module.

        Raises
        ------
        AttributeError
            If the attribute doesn't exist in the imported module.
        """
        return getattr(self._load(), name)

    def __dir__(self):
        """
        Return list of attributes for tab completion and dir().

        This ensures that IDEs and interactive environments can provide
        proper tab completion and introspection by loading the module
        and returning its attributes.

        Returns
        -------
        list of str
            List of attribute names from the imported module.
        """
        return dir(self._load())

    def __repr__(self):
        """
        Return string representation of the lazy import.

        Loads the module to provide its actual repr, falling back to
        a placeholder if something goes wrong.

        Returns
        -------
        str
            String representation of the module or a placeholder.
        """
        module = self._load()
        return repr(module) if module else f"<{self._module_name} (not loaded)>"


def __getattr__(name):
    """
    Lazily create LazyImport instances when attributes are accessed.

    For example, this allows 'from openmdao.utils.lazy_imports import jax'
    to raise ImportError if jax is not available, matching standard import
    behavior.

    This provides a special case for mpi4py.MPI, since in that case we
    want the MPI module to be lazily loaded, not the mpi4py package.

    Any attribute name will attempt to be lazily imported as a module.

    Parameters
    ----------
    name : str
        The name of a module to be lazily loaded.
    """
    # Ignore private/dunder attributes - these are Python internals
    # The import system will handle this.
    if name.startswith('_'):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == 'MPI':
        return LazyImport('mpi4py.MPI')
    elif name == 'PETSc':
        return LazyImport('petsc4py.PETSc')

    return LazyImport(name)
