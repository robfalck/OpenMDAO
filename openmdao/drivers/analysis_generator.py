"""
Provide generators for use with AnalysisDriver.

These generators are pythonic, lazy generators which, when provided with a dictionary
of variables and values to be tested, produce some set of sample values to be evaluated.

"""

from collections.abc import Iterator
import csv
import itertools

import numpy as np


class AnalysisGenerator(Iterator):
    """
    Provide a generator which provides case data for AnalysisDriver.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val`.

    Attributes
    ----------
    _iter : Iterator
        The underlying iterator for variable values.
    _run_count : int
        A running count of the samples obtained from the iterator.
    _var_dict : dict
        An internal copy of the var_dict used to create the generator.
    """

    def __init__(self, var_dict):
        """
        Instantiate the base class for AnalysisGenerators.

        Parameters
        ----------
        var_dict : dict
            A dictionary mapping a variable name to values to be assumed, as well as optional
            units and indices specifications.
        """
        super().__init__()
        self._run_count = 0
        self._var_dict = var_dict
        self._iter = None

    def _setup(self, problem=None):
        """
        Reset the run counter and instantiate the internal Iterator.

        Subclasses of AnalysisGenerator should override this method
        to define self._iter.

        This method is called by AnalysisDriver at the start of a run.
        """
        self._run_count = 0

    def _get_sampled_vars(self):
        """
        Return the set of variable names whose value are provided by this generator.
        """
        return self._var_dict.keys()

    def __next__(self):
        """
        Provide a dictionary of the next point to be analyzed.

        The key of each entry is the promoted path of var whose values are to be set.
        The associated value is the values to set (required), units (optional),
        and indices (optional).

        Raises
        ------
        StopIteration
            When all analysis var_dict have been exhausted.

        Returns
        -------
        dict
            A dictionary containing the promoted paths of variables to
            be set by the AnalysisDriver
        """
        d = {}
        vals = next(self._iter)

        for i, name in enumerate(self._var_dict.keys()):
            d[name] = {'val': vals[i],
                       'units': self._var_dict[name].get('units', None),
                       'indices': self._var_dict[name].get('indices', None)}
        self._run_count += 1
        return d


class ZipGenerator(AnalysisGenerator):
    """
    A generator which provides case data for AnalysisDriver by zipping values of each factor.

    Parameters
    ----------
    var_dict : dict
        A dictionary which maps promoted path names of variables to be
        set in each itearation with their values to be assumed (required),
        units (optional), and indices (optional).
    """

    def _setup(self, problem=None):
        """
        Set up the iterator which provides each case.

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        super()._setup()
        sampler = (c['val'] for c in self._var_dict.values())
        _lens = {name: len(meta['val']) for name, meta in self._var_dict.items()}
        if len(set([_l for _l in _lens.values()])) != 1:
            raise ValueError('ZipGenerator requires that val '
                             f'for all var_dict have the same length:\n{_lens}')
        sampler = (c['val'] for c in self._var_dict.values())
        self._iter = zip(*sampler)


class ProductGenerator(AnalysisGenerator):
    """
    A generator which provides full-factorial case data for AnalysisDriver.

    Parameters
    ----------
    var_dict : dict
        A dictionary which maps promoted path names of variables to be
        set in each itearation with their values to be assumed (required),
        units (optional), and indices (optional).
    """

    def _setup(self, problem=None):
        """
        Set up the iterator which provides each case.

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        super()._setup()
        sampler = (c['val'] for c in self._var_dict.values())
        self._iter = itertools.product(*sampler)


class CSVGenerator(AnalysisGenerator):
    """
    A generator which provides cases for AnalysisDriver by pulling rows from a CSV file.

    Parameters
    ----------
    filename : str
        The filename for the CSV file containing the variable data.
    has_units : bool
        If True, the second line of the CSV contains the units of each variable.
    has_indices : bool
        If True, the line after units (if present) contains the indices being set.

    Attributes
    ----------
    _filename : str
        The filename of the CSV file providing the samples.
    _has_units : bool
        True if the CSV file contains a row of the units for each variable.
    _has_indices : bool
        True if the CSV file contains a row of indices being provided for each variable.
        If units are present, indices will be on the line following units.
    _csv_file : file
        The file object for the CSV file.
    _csv_reader : DictReader
        The reader object for the CSV file.
    _var_names : set of str
        The set of variable names provided by this CSVGenerator.
    _ret_val : dict
        The dict which is returned by each call to __next__.
    """

    def __init__(self, filename, has_units=False, has_indices=False):
        """
        Instantiate CSVGenerator.

        Parameters
        ----------
        filename : str
            The filename for the CSV file containing the variable data.
        has_units : bool
            If True, the second line of the CSV contains the units of each variable.
        has_indices : bool
            If True, the line after units (if present) contains the indices being set.
        """
        self._filename = filename
        self._has_units = has_units
        self._has_indices = has_indices

        self._csv_file = open(self._filename, 'r')
        self._csv_reader = csv.DictReader(self._csv_file)

        self._var_names = set(self._csv_reader.fieldnames)

        self._ret_val = {var: {'units': None, 'indices': None}
                         for var in self._csv_reader.fieldnames}

        if self._has_units:
            var_units_dict = next(self._csv_reader)
            for var, units in var_units_dict.items():
                self._ret_val[var]['units'] = None if not units else units

        if self._has_indices:
            var_idxs_dict = next(self._csv_reader)
            for var, idxs in var_idxs_dict.items():
                idxs = eval(idxs, {'__builtins__': {}})  # nosec: scope limited
                self._ret_val[var]['indices'] = idxs

    def _get_sampled_vars(self):
        return self._var_names

    def __next__(self):
        """
        Provide the data from the next row of the csv file.
        """
        try:
            var_val_dict = next(self._csv_reader)
            for var, val in var_val_dict.items():
                self._ret_val[var]['val'] = val
            return self._ret_val
        except StopIteration:
            # Close the file and propagate the exception
            self._csv_file.close()
            raise

    def __del__(self):
        """
        Ensure the file is closed if we don't exhaust the iterator.
        """
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()


class SequenceGenerator(AnalysisGenerator):
    """
    A generator which provides samples from python lists or tuples.

    Internally this generator converts the list or tuple to a deque and then consumes it
    as it iterates over it.

    Parameters
    ----------
    container : container
        A python container, excluding strings, bytes, or bytearray.

    Attributes
    ----------
    _sampled_vars : list(str)
        A list of the variables in the model being sampled.
    _iter : Iterator
        The internal iterator over the users case data.

    Raises
    ------
    StopIteration
        When given list or tuple is exhausted.
    """

    def __init__(self, container):
        """
        Instantiate a SequenceGenerator with the given container of samples.
        """
        self._sampled_vars = [k for k in list(container)[0].keys()]
        self._iter = iter(container)

    def __iter__(self):
        """
        Provide the python iterator for this instance.
        """
        return self

    def __next__(self):
        """
        Provide the next values for the variables in the generator.
        """
        return next(self._iter)

    def _get_sampled_vars(self):
        """
        Return the set of variable names whose value are provided by this generator.
        """
        return self._sampled_vars


# def _inverse_affine(x, lower, upper):
#     """
#     Maps x from the range [-1, 1] to [lower, upper].

#     The generators in pyDOE3 yield values on [-1, 1], which we remap to [lower, upper]

#     Parameters
#     ----------
#     x : float or array-like
#         Value(s) in the range [-1, 1].
#     lower : float
#         Lower bound of the target range.
#     upper : float
#         Upper bound of the target range.

#     Returns
#     -------
#     float or array-like
#         Transformed value(s) in the range [lower, upper].
#     """
#     return (upper - lower) * (x + 1) / 2 + lower


class BaseDOEGenerator(AnalysisGenerator):
    
    def _setup(self, problem):
        """
        If self._var_data is not provided, use the problems design variables.

        Parameters
        ----------
        problem : Problem
            The OpenMDAO problem associated with the generator.
        """
        if self._var_dict is not None:
            return
        
        self._var_dict = problem.driver._designvars
    
    def _get_size(self):
        for meta in self._var_dict.values():
            if 'size' not in meta:
                if 'shape' in meta:
                    meta['size'] = np.prod(meta['shape'])
                elif 'val' in meta:
                    meta['size'] = np.prod(np.asarray(meta['val']).shape)
                else:
                    meta['size'] = 1

        return sum([meta['size'] for meta in self._var_dict.values()])

    def _get_sample_vals(self, doe):
        # yield desvar values for doe samples
        for row in doe:
            retval = []
            col = 0
            for name, meta in self._var_dict.items():
                size = meta['size']
                doe_cols = row[col:col + size]

                lower = meta['lower']
                if not isinstance(lower, np.ndarray):
                    lower = lower * np.ones(size)

                upper = meta['upper']
                if not isinstance(upper, np.ndarray):
                    upper = upper * np.ones(size)

                val = lower + doe_cols * (upper - lower)

                retval.append(val)
                col += size

            yield retval


class BoxBehnkenGenerator(BaseDOEGenerator):

    def __init__(self, var_dict=None, centers=1):
        """
        Instantiate the base class for AnalysisGenerators.

        Parameters
        ----------
        var_dict : dict
            A dictionary mapping a variable name to their lower and upper bounds to be tested.
            For each varaible, dictionaries lower and upper are required. Entry 'units', 'shape',
            and 'indices' may be optionally provided.

            If var_dict is not provided, use the problem's defined design variables.
        centers : int
            The number of center points in the design.
        """
        super().__init__(var_dict)
        self._centers = centers

    def _setup(self, problem=None):
        """
        Set up the iterator which provides each case.

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        try:
            from pyDOE3 import bbdesign
        except ImportError as e:
            raise ImportError(
                f'{self.__class__.__name__} requires pyDOE3\n'
                'Use `python -m pip install pyDOE3` to install it.'
            ) from e

        super()._setup(problem)

        bb_designs = bbdesign(n=self._get_size(), center=self._centers)

        # def _iter_rows():
        #     num_designs = bb_designs.shape[0]
        #     for i in range(num_designs):
        #         bb = bb_designs[i, ...]
        #         vals = []
        #         for j, (_, meta) in enumerate(self._var_dict.items()):
        #             lower = meta['lower']
        #             upper = meta['upper']
        #             vals.append(_inverse_affine(bb[j], lower, upper))
        #         yield vals

        self._iter = self._get_sample_vals(bb_designs)


class LHSGenerator(BaseDOEGenerator):
    """
    Parameters
    ----------
    var_dict : dict, optional
        A dictionary mapping a variable name to their lower and upper bounds to be tested.
        For each varaible, dictionaries lower and upper are required. Entry 'units', 'shape',
        and 'indices' may be optionally provided.

        If var_dict is not provided, use the problem's defined design variables.
    samples : int, optional
        The number of samples to generate for each factor (Defaults to n).
    criterion : str, optional
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int, optional
        The number of iterations in the maximin and correlations algorithms
        (Defaults to 5).
    seed : int, optional
        Random seed to use if design is randomized. Defaults to None.
    """
    def __init__(self, var_dict=None, num_samples=None, criterion=None, iterations=5, random_state=None):
        super().__init__(var_dict)
        self._num_samples = num_samples
        self._criterion = criterion
        self._iterations = iterations
        self._random_seed = random_state
            
    def _setup(self, problem=None):
        try:
            from pyDOE3 import lhs
        except ImportError as e:
            raise ImportError(
                f'{self.__class__.__name__} requires pyDOE3\n'
                'Use `python -m pip install pyDOE3` to install it.'
            ) from e
        super()._setup(problem)

        lhs_designs = lhs(self._get_size(),
                          samples=self._num_samples,
                          criterion=self._criterion,
                          iterations=self._iterations,
                          random_state=self._random_seed)

        self._iter = self._get_sample_vals(lhs_designs)