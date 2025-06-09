from collections.abc import Iterable


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
        _args = {key : {'indices': None, 'units': None, 'val': None} for key in args}
        _args.update(args)
    elif isinstance(args, Iterable):
        _args = {arg: {'indices': None, 'units': None, 'val': None} for arg in args}
    elif isinstance(args, str):
        _args = {args: {'indices': None, 'units': None, 'val': None}}
    elif args is None:
        _args = {}
    else:
        raise TypeError('`args` must be a dictionary of variable, '
                        'metadata pairs, a list of strings, or a string.')
    return _args


class Function():
    """
    A callable that provides a function interface to a Problem.

    This callable serves as both a function itself and also provides
    a `derivs` method which returns
    """

    def __init__(self, problem, args=None, returns=None,
                 design_vars=True, constraints=True, objectives=True, driver_scaling=False,
                 implicit_outputs=False, residuals=False, solver_scaling=False,
                 derivs=True):

        self._problem = problem

        driver_vars = problem.list_driver_vars(out_stream=None)

        desvars = driver_vars['design_vars']
        # cons = driver_vars['constraints']
        # objs = driver_vars['objectives']

        # other_args = args if args is None else {}
        # other_returns = returns if returns is None else {}

        # Initialize args and returns with the user requested values.
        self._args = _create_arg_or_return_dict(args, arg_type='args')
        self._returns = _create_arg_or_return_dict(args, arg_type='returns')

        print(problem.model._resolver.keys())

        # Add design vars to the args
        dvs = { v['name']: {'indices': v['indices'],
                            'units': problem.model._var_abs2meta['output'][abs_name]['units'],
                            'val': v['val']} for abs_name, v in desvars}
        self._args.update(dvs)


    def __call__(self, *args):
        print('hi there')
        # print(func_self._args)
        print(self._args)

    def derivs(self, *args):
        print('hi from the derivs func')
