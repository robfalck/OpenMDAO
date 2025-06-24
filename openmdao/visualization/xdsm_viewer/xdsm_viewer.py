from collections import deque

import openmdao.api as om

# from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
# from openmdao.solvers.solver import Solver
# from openmdao.core.driver import Driver

try:
    from pyxdsm.XDSM import XDSM, SOLVER, FUNC, GROUP
except ImportError:
    XDSM = None


# def _all_subsys_iter(sys, include_self=False, recurse=True, typ=None,
#                      recurse_exceptions=None, solvers_active=True,
#                      include_nl_run_once=False):
#     """
#     Yield a generator of local subsystems of this system.

#     Parameters
#     ----------
#     include_self : bool
#         If True, include this system in the iteration.
#     recurse : bool
#         If True, iterate over the whole tree under this system.
#     typ : type
#         If not None, only yield Systems that match that are instances of the
#         given type.
#     recurse_exceptions : set
#         A set of system pathnames that should not be recursed, if recurse is True.
#     solvers_active : bool
#         If True, solvers are active and should be included in the hierarchy
#         as we proceed down through the model. Otherwise, they are ignored.
#         When a NewtonSolver with solve_subsystems=False is encountered, this
#         option is set to false so that the XDSM conveys that the inner solvers are not active.

#     Yields
#     ------
#     type or None
#     """
#     from openmdao.core.group import Group
#     from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce

#     _recurse_excptions = recurse_exceptions or []

#     if include_self and (typ is None or isinstance(sys, typ)):
#         print(f'yielding {sys} line 50')
#         yield sys

#     if isinstance(sys, Group):
#         if recurse:
#             ignore_solver = False
#             if not include_nl_run_once and isinstance(sys.nonlinear_solver, NonlinearRunOnce):
#                 ignore_solver = True
#             if isinstance(sys.nonlinear_solver, om.NewtonSolver) and not sys.nonlinear_solver.options['solve_subsystems']:
#                 solvers_active = False
#                 # If Newton is encountered without solve subsystems, yield this solver but no inner solvers for the diagram.
#             if not ignore_solver:
#                 yield sys.nonlinear_solver
#         else:
#             print(f'yielding {sys} line 64')
#             yield sys

#     for subsys_name in sys._subsystems_allprocs:
#         subsys = sys._get_subsystem(subsys_name)
#         if typ is None or isinstance(subsys, typ):
#             print(f'yielding {sys} line 70')
#             yield subsys

#         if recurse and subsys.pathname not in _recurse_excptions:
#             for sub in _all_subsys_iter(subsys, include_self=False, recurse=True, typ=typ,
#                                         recurse_exceptions=recurse_exceptions):
#                 print(f'yielding {sub} line 50')
#                 yield sub


def _make_legal_node_name(s):
    """
    Create a legal tikz node name by replacing '.', ',', ':', and ';' with a dash.

    Parameters
    ----------
    s : str
        The node name.

    Returns
    -------
    str
        The legal tikz node name.
    """
    for _char in '.', ',', ':', ';':
        s = s.replace(_char, '_')
    return s

def _detokenize(s):
    r"""
    Wrap the given string in the LaTeX \detokenize command.

    Parameters
    ----------
    s : str
        The string to be detokenized.

    Returns
    -------
    str
        The string wrapped in \detokenize{}.
    """
    return rf'\detokenize{{{s}}}'

def _collect_connections(system):
    """
    Collect all of the variables between two systems, returning a dictionary
    keyed by a tuple of the system paths (source, target), and mapped to a list of
    the output variable names being connected to the target system.

    Parameters
    ----------
    system : System
        The OpenMDAO system whose connections are being gathered.

    Returns
    -------
    dict[str, str]
        A dictionary that maps the source system name and target system name to a set of variable names
        connected between the two.
    """
    global_abs_in2out = system._conn_global_abs_in2out
    conns = {}
    for inp, outp in global_abs_in2out.items():
        source_sys, source_var = outp.rsplit('.', maxsplit=1)
        source_var = source_var
        tgt_sys = inp.rsplit('.', maxsplit=1)[0]
        if (source_sys, tgt_sys) in conns:
            conns[source_sys, tgt_sys].add(source_var)
        else:
            conns[source_sys, tgt_sys] = {source_var}
    return conns

def _convert_varname_to_type(source_sys, var_name, abs2prom_out, var_names='local', ):
    if var_names in ('promoted', 'local'):
        name = abs2prom_out(f'{source_sys}.{var_name}')
        if var_names == 'local':
            name = name.split('.')[-1]
        return name
    elif var_names == 'absolute':
        return f'{source_sys}.{var_name}'
    else:
        raise ValueError('var_names must be one of "absolute", "promoted", or "local".')

def create_xdsm(problem_or_group, recurse=True, recurse_exceptions=None, use_full_path=False,
                show_autoivc=False, var_map=None, var_names='local', include_nl_run_once=False):
    """
    Create an XDSM PDF file using pyXDSM.

    Parameters
    ----------
    problem_or_group : _type_
        _description_
    recurse : bool, optional
        _description_, by default True
    recurse_exceptions : _type_, optional
        _description_, by default None
    use_full_path : bool, optional
        _description_, by default False
    show_autoivc : bool, optional
        _description_, by default False
    var_map : _type_, optional
        _description_, by default None
    var_names : str, optional
        _description_, by default 'local'
    """
    if XDSM is None:
        raise RuntimeError('create_xdsm requires the pyxdsm package. Try `pip install pyxdsm')

    _var_map = var_map or {}

    if isinstance(problem_or_group, om.Problem):
        _model = problem_or_group.model
    else:
        _model = problem_or_group

    _top_pathname = _model.pathname if _model.pathname else 'Problem.model'

    abs2prom_out = _model._resolver.abs2prom

    # Change `use_sfmath` to False to use computer modern
    xdsm = XDSM(use_sfmath=True)
    sys_outputs = dict()

    conns = _collect_connections(_model)

    solvers = deque()

    for sys in _model.system_iter(include_self=False, recurse=recurse):
        kwargs = {'stack': False, 'faded': False, 'label_width': None, 'spec_name': None}
        # depth = sys.pathname.count('.')

        # keyword arguments that are common regardless of system type.
        name = _make_legal_node_name(sys.pathname)
        label = sys.pathname if use_full_path else sys.pathname.split('.')[-1]
        label = _detokenize(label)

        if isinstance(sys, om.Group):
            if recurse:
                kind = SOLVER
                solver = sys.nonlinear_solver
                name = _make_legal_node_name(f'{solver}_{sys.pathname}')
                solvers.append(name)
                label = [_detokenize(str(solver)), label]
            else:
                kind = GROUP
                sys_outputs[name] = list(sys.list_outputs(out_stream=None, return_format='dict').keys())

        elif isinstance(sys, Component):
            if sys.pathname == '_auto_ivc' and not show_autoivc:
                continue
            kind = FUNC

            if isinstance(sys, ImplicitComponent):
                imp_outputs = sys.list_outputs(residuals=True, out_stream=None, return_format='dict')
                for imp_output in imp_outputs.keys():
                    for solver_node_name in solvers:
                        xdsm.connect(solver_node_name,
                                     _make_legal_node_name(sys.pathname),
                                     sorted({_detokenize(v) for v in imp_outputs.keys()}))
                        xdsm.connect(_make_legal_node_name(sys.pathname),
                                     solver_node_name,
                                     sorted({rf'\mathcal{{R}}\left({_detokenize(v)}\right)' for v in imp_outputs.keys()}))
            else:
                sys_outputs[name] = list(sys.list_outputs(out_stream=None, return_format='dict').keys())
        else:
            raise RuntimeError('Unexpected system type', sys)

        xdsm.add_system(name, kind, label=label, **kwargs)


    #
    # Render the connections
    #
    for (source_sys, tgt_sys), output_vars in conns.items():
        if source_sys == '_auto_ivc' and not show_autoivc:
            from_auto_ivc = set()
            for var in output_vars:
                var_label = _convert_varname_to_type(source_sys, var, abs2prom_out, var_names)
                from_auto_ivc.add(var_label)
            xdsm.add_input(_make_legal_node_name(tgt_sys), sorted({_detokenize(v) for v in from_auto_ivc}))
        else:
            if not recurse:
                # If we're not recursing into groups, then connect the groups rather than
                # their internal components.
                source_sys = source_sys.split('.')[0]
                tgt_sys = tgt_sys.split('.')[0]
            if source_sys != tgt_sys:
                xdsm.connect(_make_legal_node_name(source_sys),
                            _make_legal_node_name(tgt_sys),
                            sorted({_detokenize(v) for v in output_vars}))

    for sys_name, outputs in sys_outputs.items():
        for (source_sys, tgt_sys), output_vars in conns.items():
            if not recurse:
                source_sys = source_sys.split('.')[0]


    print(sys_outputs)
    print(conns)
    # elif outputs:
    #     xdsm.add_output(name, sorted({_detokenize(v) for v in outputs.keys()}), side='right')

    xdsm.write('xdsm')