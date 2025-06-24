from collections.abc import Iterable
from openmdao.core.constants import _DEFAULT_OUT_STREAM

import numpy as np


def _create_arg_or_return_dict(args=None, arg_type='args'):
    """
    Create args or returns dictionaries for the function interface.

    These dictionaries should include indices and units at a minimum,
    and are updated with metadata that might be provided by the user.

    Parameters
    ----------
    args : dict, Iterable, or str
        The variables in the system which the user wants to be arguments
        or return values. If given as a dict, the user can specify indices
        and units for each variable, otherwise they will default to None.

    Raises
    ------
    TypeError
        Raised if args is not of an acceptable type

    Returns
    -------
    dict
        A dictionary of argument or return variables and their relevant metadata.
    """
    if isinstance(args, dict):
        _args = {key : {'indices': None, 'units': None} for key in args}
        _args.update(args)
    elif isinstance(args, Iterable):
        _args = {arg: {'indices': None, 'units': None} for arg in args}
    elif isinstance(args, str):
        _args = {args: {'indices': None, 'units': None}}
    elif args is None:
        _args = {}
    else:
        raise TypeError('`{arg_type}` must be a dictionary of variable, '
                        'metadata pairs, a list of strings, or a string.')
    return _args


class Function:
    """
    A callable that provides a function interface to a Problem.

    This callable serves as both a function itself and also provides
    a `derivs` method which returns

    Parameters
    ----------
    problem
        The OpenMDAO problem for which this function interface is being provided.
    args
        Promoted variable paths to be used as arguments of the function.
    returns
        Promoted variable paths to be used as returned by the function.
    design_vars
        If True, include declared design variables in the arguments.
    constraints
        If True, include declared constraints in the returned values.
    objectives
        If True, include declared objectives in the returned values.
    driver_scaling
        If True, accept design variables and return outputs with driver
        scaling applied.
    implicit_outputs
        If True, include impicit outputs in the returned values.
    residuals
        If True, include residuals in the returned values.
    solver_scaling
        If True, include solver scaling on implicit outputs and residuals.
    flatten_args
        If True, args shall be provided as a single, C-order-flattened vector.
    flatten_returns
        If True, returns shall be returned as a single, C-order-flattened vector.
    """
    def __init__(self, problem, args=None, returns=None,
                 design_vars=True, constraints=True, objectives=True, driver_scaling=False,
                 implicit_outputs=False, residuals=False, solver_scaling=False,
                 flatten_args=True, flatten_returns=True):

        self._problem = problem
        self._flatten_args = flatten_args
        self._flatten_returns = flatten_returns

        responses = problem.model.get_responses(recurse=True, use_prom_ivc=True)
        cons = {k: v for k, v in responses.items() if v['type'] == 'con'}
        objs = {k: v for k, v in responses.items() if v['type'] == 'obj'}
        desvars = problem.model.get_design_vars(recurse=True, use_prom_ivc=True)

        # Initialize args and returns with the user requested values.
        self._args = _create_arg_or_return_dict(args, arg_type='args')
        self._returns = _create_arg_or_return_dict(args, arg_type='returns')

        # Add design vars to the args
        dvs = { v['name']: {'indices': v.get('indices', None),
                            'units': v['units']} for abs_name, v in desvars.items()}

        if design_vars:
            self._args.update(dvs)
        if constraints:
            self._returns.update(cons)
        if objectives:
            self._returns.update(objs)

        self._x_size = 0

        for arg, meta in self._args.items():
            meta['size'] = np.size(self._problem.get_val(arg)[meta['indices']].ravel())
            self._x_size += meta['size']

    def _set_vals(self, *args):
        """

        """

    def list_args(self, out_stream=_DEFAULT_OUT_STREAM):
        """
        Return a
        """

    def get_x(self):
        """
        Return the current value of the arguments.
        """
        x0 = []
        for arg, meta in self._args.items():
            x_val = self._problem.get_val(arg, units=meta['units'])[meta['indices']]
            x0.append(x_val)

        if self._flatten_args:
            return np.concatenate([x.ravel() for x in x0]).ravel()
        return x0

    def set_x(self, x):
        """
        Set new values in the model from a single 1D vector of arguments.
        """
        i = 0
        for arg, meta in self._args.items():
            indices=meta['indices']
            num_x = len(np.ravel(indices))
            units=meta['units']
            self._problem.set_val(arg, x[i:i + num_x], indices=indices, units=units)
            i += num_x

    def get_f(self):
        """
        Return the current outputs of the Function.
        """
        ret_val = []
        for ret, meta in self._returns.items():
            indices=meta['indices']
            units=meta['units']
            ret_val.append(self._problem.get_val(ret, indices=indices, units=units))
        if self._flatten_returns:
            return np.concatenate([f.ravel() for f in ret_val]).ravel()
        return ret_val

    def __call__(self, *args):
        if self._flatten_args:
            x = np.asarray(args[0])
            if np.size(x) != self._x_size:
                raise ValueError(f'Function expected x to contain {self._x_size} elements but it contains {np.size(x)}.')
            self.set_x(x)

        self._problem.run_model()

        return self.get_f()


    def derivs(self, *args):
        if self._flatten_args:
            x = np.asarray(args[0])
            if np.size(x) != self._x_size:
                raise ValueError(f'Function expected x to contain {self._x_size} elements but it contains {np.size(x)}.')
            self.set_x(x)

        self._problem.run_model()

        totals = self._problem.compute_totals()
        return(totals)
