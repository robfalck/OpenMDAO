import openmdao.api as om

from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.solvers.solver import Solver
from openmdao.core.driver import Driver


def _all_subsys_iter(sys, include_self=True, recurse=True, typ=None,
                     recurse_exceptions=None):
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

    Yields
    ------
    type or None
    """
    from openmdao.core.group import Group
    from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce

    _recurse_excptions = recurse_exceptions or []

    if include_self and (typ is None or isinstance(sys, typ)):
        yield sys

    if isinstance(sys, Group) and not isinstance(sys.nonlinear_solver, NonlinearRunOnce):
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
    system : _type_
        _description_
    """
    global_abs_in2out = system._conn_global_abs_in2out
    conns = {}
    for inp, outp in global_abs_in2out.items():
        source_sys, source_var = outp.rsplit('.', maxsplit=1)
        source_var = _detokenize(source_var)
        tgt_sys = inp.rsplit('.', maxsplit=1)[0]
        if (source_sys, tgt_sys) in conns:
            conns[source_sys, tgt_sys].add(source_var)
        else:
            conns[source_sys, tgt_sys] = {source_var}
    return conns

def create_xdsm(problem_or_group, recurse=True, recurse_exceptions=None, use_full_path=False,
                show_autoivc=False, var_map=None):

    _var_map = var_map or {}

    if isinstance(problem_or_group, om.Problem):
        _model = problem_or_group.model
    else:
        _model = problem_or_group

    from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC

    # Change `use_sfmath` to False to use computer modern
    xdsm = XDSM(use_sfmath=True)

    # x.add_system("opt", OPT, r"\text{Optimizer}")
    # x.add_system("solver", SOLVER, r"\text{Newton}")
    # x.add_system("D1", FUNC, "D_1")
    # x.add_system("D2", FUNC, "D_2")
    # x.add_system("F", FUNC, "F")
    # x.add_system("G", FUNC, "G")

    # Add systems, solvers, and optimizers to the XDSM
    for item in _all_subsys_iter(_model, include_self=False,
                                recurse=recurse,
                                recurse_exceptions=recurse_exceptions,
                                typ=Component):
        if isinstance(item, System):
            if item.pathname == '_auto_ivc' and not show_autoivc:
                continue
            name = item.pathname
            label = item.pathname if use_full_path else item.pathname.split('.')[-1]
            label = _detokenize(label)
            kind = FUNC
        elif isinstance(item, Solver):
            name = str(item)
            label = name
            kind = SOLVER
        elif isinstance(item, Driver):
            # TODO: Check if opt driver
            name = item.__class__
            label = name
            kind = OPT

        # Remove illegal punctuation from names
        name = _make_legal_node_name(name)
        xdsm.add_system(name, kind, label=label)

    conns = _collect_connections(_model)

    for (source_sys, tgt_sys), output_vars in conns.items():
        print(source_sys)
        print(tgt_sys)
        print(output_vars)

        if source_sys == '_auto_ivc' and not show_autoivc:
            xdsm.add_input(_make_legal_node_name(tgt_sys), '')
        else:
            xdsm.connect(_make_legal_node_name(source_sys),
                         _make_legal_node_name(tgt_sys),
                         sorted(output_vars))
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