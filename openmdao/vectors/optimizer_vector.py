"""Lightweight vector wrapper for driver-level design variables and responses."""

import numpy as np


class OptimizerVector(object):
    """
    Provides name-based indexing over optimizer-level design variable or response vectors.

    This is a lightweight dict-like wrapper, not a System Vector. It provides convenient
    access for optimization algorithms without the System dependencies of DefaultVector.

    Parameters
    ----------
    data : ndarray
        Flat numpy array containing variable values.
    metadata : dict
        Metadata dict mapping variable names to index information. Each entry should contain
        'start_idx', 'end_idx', and 'size' keys.

    Examples
    --------
    >>> vec = OpimizerVector(numpy_array, metadata)
    >>> x_value = vec['x']  # Get design variable by name
    >>> vec['x'] = 2.5  # Set design variable by name
    >>> for name, value in vec.items():  # Iterate over variables
    ...     print(f"{name}: {value}")
    """

    def __init__(self, data=None, metadata=None):
        """Initialize OpimizerVector with data array and metadata."""
        self._data = data
        self._meta = metadata

    def __getitem__(self, name):
        """
        Get variable value by name.

        Parameters
        ----------
        name : str
            Variable name (promoted or alias).

        Returns
        -------
        ndarray
            1D array of variable values.

        Raises
        ------
        KeyError
            If variable name not found.
        """
        if name not in self._meta:
            raise KeyError(f"Variable '{name}' not found in OpimizerVector")
        info = self._meta[name]
        return self._data[info['start_idx']:info['end_idx']].reshape(-1)

    def __setitem__(self, name, value):
        """
        Set variable value by name.

        Parameters
        ----------
        name : str
            Variable name (promoted or alias).
        value : float or ndarray
            New value for the variable.

        Raises
        ------
        KeyError
            If variable name not found.
        """
        if name not in self._meta:
            raise KeyError(f"Variable '{name}' not found in OpimizerVector")
        info = self._meta[name]
        self._data[info['start_idx']:info['end_idx']] = np.asarray(value).flat

    def __contains__(self, name):
        """
        Check if variable name exists.

        Parameters
        ----------
        name : str
            Variable name to check.

        Returns
        -------
        bool
            True if variable exists, False otherwise.
        """
        return name in self._meta

    def __len__(self):
        """
        Return number of variables.

        Returns
        -------
        int
            Number of variables in this vector.
        """
        return len(self._meta)

    def __iter__(self):
        """
        Iterate over variable names.

        Yields
        ------
        str
            Variable names in iteration order.
        """
        return iter(self._meta)

    def keys(self):
        """
        Return variable names.

        Returns
        -------
        dict_keys
            View of variable names.
        """
        return self._meta.keys()

    def values(self):
        """
        Iterate over variable values.

        Yields
        ------
        ndarray
            Variable values in iteration order.
        """
        for name in self._meta:
            yield self[name]

    def items(self):
        """
        Iterate over (name, value) pairs.

        Yields
        ------
        tuple
            (variable_name, variable_value) pairs in iteration order.
        """
        for name in self._meta:
            yield name, self[name]

    def asarray(self):
        """
        Return underlying flat numpy array.

        Returns
        -------
        ndarray
            The underlying data array (not a copy).
        """
        return self._data

    def get_metadata(self, name=None):
        """
        Get metadata for variable(s).

        Parameters
        ----------
        name : str or None, optional
            Variable name to get metadata for. If None, returns all metadata.
            Default is None.

        Returns
        -------
        dict
            If name is provided, returns metadata dict for that variable.
            If name is None, returns entire metadata dict.

        Raises
        ------
        KeyError
            If name is provided and not found.
        """
        if name is None:
            return self._meta
        if name not in self._meta:
            raise KeyError(f"Variable '{name}' not found in OpimizerVector")
        return self._meta[name]

