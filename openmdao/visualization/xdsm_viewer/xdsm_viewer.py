
import openmdao.api as om

from typing import Sequence

from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent

try:
    from pyxdsm.XDSM import XDSM, SOLVER, OPT, FUNC, GROUP
except ImportError:
    XDSM = None


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


def _make_label(label, font='textsc'):
    if isinstance(label, str):
        label = [label]
    ret = []
    for i, s in enumerate(label):
        # ds = _detokenize(s)
        # ret.append(rf'\textnormal{{{ds}}}')
        ret.append(_format_label_line(s, font=font))
    return ret


def _format_label_line(s: str, font: str = None) -> str:
    """
    Wraps a string in LaTeX commands for styling, escaping special characters as needed.

    Parameters:
    - s: The input string (e.g., "group.path_name")
    - font: The LaTeX font command to use (e.g., "textbf", "textsc")

    Returns:
    - A LaTeX-formatted string like \\texttt{\\textbf{a\\_b\\_c}}
    """
    # Escape LaTeX special characters that are commonly problematic in text mode
    escape_map = {
        # '\\': r'\\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '_': r'\_',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }

    escaped = ''.join(escape_map.get(char, char) for char in s)
    if font is not None:
        return rf"\{font}{{{escaped}}}"
    else:
        return escaped


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


def _find_connection_system(pathname: str, systems: Sequence[str]) -> str:
    """
    Find the pathname in the known list of systems.

    If the pathname does not exist, try removing the part after and including the final '.'
    and search again. If we did not recurse into certain groups, then this will effectively
    treat the group as the source or target of a given connection.

    Parameters
    ----------
    pathname
        The pathname that is either the source or target of a connection.
    systems
        The systems we've added to the XDSM.

    Raises
    ------
    ValueError
        If no compatible system is found.

    Returns
    -------
    str
        The pathname in the system that matches the given pathname.  This is either an identical string,
        or the "most recent parent" system.
    """
    if pathname in systems.keys():
        return pathname
    elif '.' in pathname:
        pathname, _, _ = pathname.rpartition('.')
        return _find_connection_system(pathname, systems)
    else:
        raise ValueError(f'Could not find {pathname} in known systems.')


def _parent_groups(pathname: str) -> list[str]:
    parts = pathname.split('.')
    return [''] + ['.'.join(parts[:i+1]) for i in range(len(parts))]


def create_xdsm(problem_or_group, recurse=True, recurse_exceptions=None, use_full_path=True,
                var_map=None, var_names='local', include_nl_run_once=False):
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
    var_map : _type_, optional
        _description_, by default None
    var_names : str, optional
        _description_, by default 'local'
    """
    if XDSM is None:
        raise RuntimeError('create_xdsm requires the pyxdsm package. Try `pip install pyxdsm')

    if isinstance(problem_or_group, om.Problem):
        _model = problem_or_group.model
        driver = problem_or_group.driver
    else:
        _model = problem_or_group
        driver = None

    abs2prom_out = _model._resolver.abs2prom

    # Change `use_sfmath` to False to use computer modern
    xdsm = XDSM(use_sfmath=True)

    conns = _collect_connections(_model)
    indep_vars = list(_model.get_io_metadata(is_indep_var=True).keys())
    des_vars = list(_model.get_io_metadata(is_design_var=True).keys())

    solvers = []
    solver_conns = {}
    constraint_conns = {}
    obj_conns = {}
    systems = {}

    if recurse_exceptions is None:
        recurse_exceptions = []

    if driver is not None:
        xdsm.add_system('driver', OPT, label=_make_label(['Optimizer', driver.__class__.__name__], font='textbf'))

    for sys in _model.system_iter(include_self=False, recurse=recurse):
        kwargs = {'stack': False, 'faded': False, 'label_width': None, 'spec_name': None}


        # keyword arguments that are common regardless of system type.
        name = _make_legal_node_name(sys.pathname)
        label = [sys.pathname]
        if not use_full_path:
            label = label.split('.')[-1:]

        if isinstance(sys, om.Group):
            if not recurse:
                recurse_exceptions.append(sys.pathname)
            if sys.pathname not in recurse_exceptions:
                kind = SOLVER
                solver = sys.nonlinear_solver
                if not include_nl_run_once and isinstance(solver, om.NonlinearRunOnce):
                    continue
                name = _make_legal_node_name(sys.pathname)
                solvers.append(sys.pathname)
                label = [str(solver)] + label
            else:
                kind = GROUP

        elif isinstance(sys, Component):
            if isinstance(sys, om.IndepVarComp):
                # Don't show IVCs
                continue

            kind = FUNC

            if isinstance(sys, ImplicitComponent):
                imp_outputs = sys.list_outputs(residuals=True, out_stream=None, return_format='dict')
                parent_groups = _parent_groups(sys.pathname)
                for solver_pathname in solvers:
                    if solver_pathname in parent_groups:
                        solver_conns[solver_pathname, sys.pathname] = list(imp_outputs.keys()), False
                        solver_conns[sys.pathname, solver_pathname] = list(imp_outputs.keys()), True
        else:
            raise RuntimeError('Unexpected system type', sys)

        for ex in recurse_exceptions:
            if sys.pathname.startswith(f'{ex}.'):
                break
        else:
            systems[sys.pathname] = kind
            xdsm.add_system(name, kind, label=_make_label(label, font='textbf' if kind in (SOLVER,) else 'textsc'), **kwargs)

    #
    # Render the system connections
    #
    for (source_sys, tgt_sys), output_vars in conns.items():
        if isinstance(_model._get_subsystem(source_sys), om.IndepVarComp):
            from_inputs = set()
            from_driver = set()
            for var in output_vars:
                var_label = _convert_varname_to_type(source_sys, var, abs2prom_out, var_names)
                tgt_abs_path = f'{tgt_sys}.{var_label}'
                if tgt_abs_path in des_vars:
                    from_driver.add(var_label)
                elif tgt_abs_path in indep_vars:
                    from_inputs.add(var_label)
            if from_inputs:
                tgt_sys = _find_connection_system(tgt_sys, systems)
                xdsm.add_input(_make_legal_node_name(tgt_sys), _make_label(sorted(from_inputs), font=None))
            if from_driver:
                tgt_sys = _find_connection_system(tgt_sys, systems)
                xdsm.connect(_make_legal_node_name('driver'),
                             _make_legal_node_name(tgt_sys),
                             _make_label(sorted(from_driver), font=None))
        else:
            source_sys = _find_connection_system(source_sys, systems)
            tgt_sys = _find_connection_system(tgt_sys, systems)

            src_idx = list(systems.keys()).index(source_sys)
            tgt_idx = list(systems.keys()).index(tgt_sys)

            if source_sys != tgt_sys:
                xdsm.connect(_make_legal_node_name(source_sys),
                            _make_legal_node_name(tgt_sys),
                            sorted({rf'\detokenize{{{v}}}' for v in output_vars}))

                if src_idx > tgt_idx:
                    # This is a feedback connection, and should include connections to the parent solver.
                    parent_groups = _parent_groups(source_sys)
                    for solver_pathname in solvers:
                        # continue
                        if solver_pathname in parent_groups:
                            solver_conns[solver_pathname, tgt_sys] = sorted(output_vars), False
                            solver_conns[source_sys, solver_pathname] = sorted(output_vars), True

    #
    # Render the solver connections
    #

    for (source_sys, tgt_sys), (output_vars, is_resid) in solver_conns.items():
        source_sys = _find_connection_system(source_sys, systems)
        tgt_sys = _find_connection_system(tgt_sys, systems)

        if source_sys != tgt_sys:
            if is_resid:
                conn_vars = sorted({rf'\mathcal{{R}}(\detokenize{{{v}}})' for v in output_vars})
            else:
                conn_vars = sorted({rf'\detokenize{{{v}}}' for v in output_vars})
            xdsm.connect(_make_legal_node_name(source_sys),
                         _make_legal_node_name(tgt_sys),
                         conn_vars)

    #
    # Render the Driver connections
    #
    if driver is not None:
        for con_meta in driver._cons.values():
            source_sys, _, output = con_meta['source'].rpartition('.')
            source_sys = _find_connection_system(source_sys, systems)
            if source_sys not in constraint_conns:
                constraint_conns[source_sys] = [output]
            else:
                constraint_conns[source_sys].append(output)
        for con_source, con_vars in constraint_conns.items():
            xdsm.connect(_make_legal_node_name(con_source),
                         _make_legal_node_name('driver'),
                         [f'g({s})' for s in con_vars])

        for obj_meta in driver._objs.values():
            source_sys, _, output = obj_meta['source'].rpartition('.')
            source_sys = _find_connection_system(source_sys, systems)
            if source_sys not in obj_conns:
                obj_conns[source_sys] = [output]
            else:
                obj_conns[source_sys].append(output)
        for obj_source, obj_vars in obj_conns.items():
            xdsm.connect(_make_legal_node_name(obj_source),
                         _make_legal_node_name('driver'),
                         [f'f({s})' for s in obj_vars])

    xdsm.write('xdsm')