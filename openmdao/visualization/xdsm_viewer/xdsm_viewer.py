from collections import deque

import openmdao.api as om

from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.solvers.solver import Solver
from openmdao.core.driver import Driver


def _all_subsys_iter(sys, include_self=True, recurse=True, typ=None,
                     recurse_exceptions=None, solvers_active=True):
    """
    Yield a generator of local subsystems of this system.

    Parameters
    ----------
    include_self : bool
        If True, include this system in the iteration.
    recurse : bool
        If True, iterate over the whole tree under this system.
    typ : type
        If not None, only yield Systems that match that are instances of the
        given type.
    recurse_exceptions : set
        A set of system pathnames that should not be recursed, if recurse is True.
    solvers_active : bool
        If True, solvers are active and should be included in the hierarchy 
        as we proceed down through the model. Otherwise, they are ignored.
        When a NewtonSolver with solve_subsystems=False is encountered, this
        option is set to false so that the XDSM conveys that the inner solvers are not active.

    Yields
    ------
    type or None
    """
    from openmdao.core.group import Group
    from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce

    _recurse_excptions = recurse_exceptions or []

    if include_self and (typ is None or isinstance(sys, typ)):
        yield sys

    if solvers_active and isinstance(sys, Group) and not isinstance(sys.nonlinear_solver, NonlinearRunOnce):
        if isinstance(sys.nonlinear_solver, om.NewtonSolver) and not sys.nonlinear_solver.options['solve_subsystems']:
            solvers_active = False
            # If Newton is encountered without solve subsystems, yield this solver but no inner solvers for the diagram.
        yield sys.nonlinear_solver

    for subsys_name in sys._subsystems_allprocs:
        subsys = sys._get_subsystem(subsys_name)
        if typ is None or isinstance(subsys, typ):
            yield subsys

        if recurse and subsys.pathname not in _recurse_excptions:
            for sub in _all_subsys_iter(subsys, include_self=False, recurse=True, typ=typ,
                                        recurse_exceptions=recurse_exceptions):
                yield sub


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
    """
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
    return f'\detokenize{{{s}}}'

def _collect_connections(system):
    """
    Collect all of the variables between two systems, returning a dictionary
    keyed by a tuple of the system paths (source, target), and mapped to a list of
    the output variable names being connected to the target system.

    Parameters
    ----------
    system : System
        The OpenMDAO system whose connections are being gathered.
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
        name = abs2prom_out[f'{source_sys}.{var_name}']
        if var_names == 'local':
            name = name.split('.')[-1]
        return name
    elif var_names == 'absolute':
        return f'{source_sys}.{var_name}'
    else:
        raise ValueError('var_names must be one of "absolute", "promoted", or "local".')

def create_xdsm(problem_or_group, recurse=True, recurse_exceptions=None, use_full_path=False,
                show_autoivc=False, var_map=None, var_names='local'):
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

    _var_map = var_map or {}

    if isinstance(problem_or_group, om.Problem):
        _model = problem_or_group.model
    else:
        _model = problem_or_group

    abs2prom_out = _model._var_allprocs_abs2prom['output']

    from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC

    # Change `use_sfmath` to False to use computer modern
    xdsm = XDSM(use_sfmath=True)

    # Add systems, solvers, and optimizers to the XDSM
    solvers = deque()
    for item in _all_subsys_iter(_model, include_self=False,
                                recurse=recurse,
                                recurse_exceptions=recurse_exceptions,
                                typ=Component):
        if isinstance(item, System):
            if item.pathname == '_auto_ivc' and not show_autoivc:
                continue
            name = _make_legal_node_name(item.pathname)
            label = item.pathname if use_full_path else item.pathname.split('.')[-1]
            label = _detokenize(label)
            kind = FUNC

            if isinstance(item, ImplicitComponent):
                imp_outputs = item.list_outputs(residuals=True, out_stream=None, return_format='dict')
                for imp_output in imp_outputs.keys():
                    for solver in solvers:
                        xdsm.connect(_make_legal_node_name(solver),
                                    _make_legal_node_name(item.pathname),
                                    sorted({_detokenize(v) for v in imp_outputs.keys()}))
                        xdsm.connect(_make_legal_node_name(item.pathname),
                                    _make_legal_node_name(solver),
                                    sorted({rf'\mathcal{{R}}\left({_detokenize(v)}\right)' for v in imp_outputs.keys()}))

        elif isinstance(item, Solver):
            name = _make_legal_node_name(f'{item._system().pathname}_{str(item)}')
            label = [str(item), item._system().pathname]
            kind = SOLVER
            solvers.append(name)
        elif isinstance(item, Driver):
            # TODO: Check if opt driver
            name = _make_legal_node_name(item.__class__)
            label = name
            kind = OPT

        # Remove illegal punctuation from names
        xdsm.add_system(name, kind, label=label)

    conns = _collect_connections(_model)

    for (source_sys, tgt_sys), output_vars in conns.items():
        if source_sys == '_auto_ivc' and not show_autoivc:
            from_auto_ivc = set()
            for var in output_vars:
                var_label = _convert_varname_to_type(source_sys, var, abs2prom_out, var_names)
                from_auto_ivc.add(var_label)
            xdsm.add_input(_make_legal_node_name(tgt_sys), sorted({_detokenize(v) for v in from_auto_ivc}))

        else:
            xdsm.connect(_make_legal_node_name(source_sys),
                         _make_legal_node_name(tgt_sys),
                         sorted({_detokenize(v) for v in output_vars}))
                        # _var_map.get(source_var, _detokenize(source_var)))

    # # Add the connections
    # for inp, outp in global_abs_in2out.items():
    #     source_sys = _make_legal_node_name(outp.rsplit('.', maxsplit=1)[0])
    #     source_var = outp.rsplit('.', maxsplit=1)[-1]
    #     tgt_sys = _make_legal_node_name(inp.rsplit('.', maxsplit=1)[0])
    #     if source_sys == '_auto_ivc' and not show_autoivc:
    #         xdsm.add_input(tgt_sys, '')
    #     else:
    #         xdsm.connect(source_sys, tgt_sys,
    #                      _var_map.get(source_var, _detokenize(source_var)))

    xdsm.write('xdsm')