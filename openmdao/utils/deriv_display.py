"""
Functions used for the display of derivatives matrices.
"""
from enum import Enum
import textwrap
from io import StringIO

import numpy as np

try:
    import rich
    from rich.console import Console
except ImportError:
    rich is None

from openmdao.utils.array_utils import get_errors
from openmdao.utils.general_utils import add_border, _UNDEFINED
from openmdao.utils.mpi import MPI
from openmdao.visualization.tables.table_builder import generate_table


class _Style(Enum):
    """
    Styles tags used in formatting output with rich.
    """
    ABS_ERR = 'bright_red'
    REL_ERR = 'orange1'
    OUT_SPARSITY = 'dim'
    IN_SPARSITY = 'bold'
    WARN = 'orange1'
    SYSTEM = 'bold'
    VAR = 'bold'


def _rich_wrap(s, *tags):
    """
    If rich is available, wrap the given string in the provided tags.
    If rich is not available, just return the string.

    Parameters
    ----------
    s : str
        The string to be wrapped in rich tags.
    *tags : str
        The rich tags to be wrapped around s. These can either be
        strings or elements of the _Style enumeration.

    Returns
    -------
    str
        The given string wrapped in the provided rich tags.
    """
    if rich is None or not tags or not tags[0]:
        return s
    cmds = sorted([t if isinstance(t, str) else t.value for t in tags])
    on = ' '.join(cmds)
    off = '/' + ' '.join(reversed(cmds))
    return f'[{on}]{s}[{off}]'


def _deriv_display(system, err_iter, derivatives, rel_error_tol, abs_error_tol, out_stream,
                   fd_opts, totals=False, show_only_incorrect=False, lcons=None, console=None):
    """
    Print derivative error info to out_stream.

    Parameters
    ----------
    system : System
        The system for which derivatives are being displayed.
    err_iter : iterator
        Iterator that yields tuples of the form (key, fd_norm, fd_opts, directional, above_abs,
        above_rel, inconsistent) for each subjac.
    derivatives : dict
        Dictionary containing derivative information keyed by (of, wrt).
    rel_error_tol : float
        Relative error tolerance.
    abs_error_tol : float
        Absolute error tolerance.
    out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
    fd_opts : dict
        Dictionary containing options for the finite difference.
    totals : bool
        True if derivatives are totals.
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    lcons : list or None
        For total derivatives only, list of outputs that are actually linear constraints.
    sort : bool
        If True, sort subjacobian keys alphabetically.
    console : Console
        The rich.console.Console to which derivative information is
        displayed, if available. If None, _deriv_display will instantiate
        a new Console (if rich is available) and also save it as an html report.
        If provided, assume the caller will save the report.
    """
    from openmdao.core.component import Component

    if out_stream is None:
        return

    # Match header to appropriate type.
    if isinstance(system, Component):
        sys_type = 'Component'
    else:
        sys_type = 'Group'

    sys_name = system.pathname
    sys_class_name = type(system).__name__

    if totals:
        sys_name = 'Full Model'

    num_bad_jacs = 0  # Keep track of number of bad derivative values for each component

    # Need to capture the output of a component's derivative
    # info so that it can be used if that component is the
    # worst subjac. That info is printed at the bottom of all the output
    sys_buffer = StringIO()

    if totals:
        title = "Total Derivatives"
    else:
        title = f"{sys_type}: {sys_class_name} '{sys_name}'"

    print(f"{add_border(title, '-')}\n", file=sys_buffer)
    parts = []

    for key, fd_opts, directional, above_abs, above_rel, inconsistent in err_iter:

        if above_abs or above_rel or inconsistent:
            num_bad_jacs += 1

        of, wrt = key
        derivative_info = derivatives[key]

        # Informative output for responses that were declared with an index.
        indices = derivative_info.get('indices')
        if indices is not None:
            of = f'{of} (index size: {indices})'

        # need this check because if directional may be list
        if isinstance(wrt, str):
            wrt = f"'{wrt}'"
        if isinstance(of, str):
            of = f"'{of}'"

        if directional:
            wrt = f"(d){wrt}"

        abs_errs = derivative_info['abs error']
        rel_errs = derivative_info['rel error']
        abs_vals = derivative_info['vals_at_max_abs']
        rel_vals = derivative_info['vals_at_max_rel']
        denom_idxs = derivative_info['denom_idx']
        steps = derivative_info['steps']

        Jfwd = derivative_info.get('J_fwd')
        Jrev = derivative_info.get('J_rev')

        if len(steps) > 1:
            stepstrs = [f", step={step}" for step in steps]
        else:
            stepstrs = [""]

        fd_desc = f"{fd_opts['method']}:{fd_opts['form']}"
        parts.append(f"  {_rich_wrap(sys_name, _Style.SYSTEM)}:"
                     f" {_rich_wrap(of, _Style.VAR)} "
                     f"wrt {_rich_wrap(wrt, _Style.VAR)}")
        if not isinstance(of, tuple) and lcons and of.strip("'") in lcons:
            parts[-1] += " (Linear constraint)"
        parts.append('')

        for i in range(len(abs_errs)):
            # Absolute Errors
            if directional:
                if totals and abs_errs[i].forward is not None:
                    err = _format_error(abs_errs[i].forward, abs_error_tol)
                    parts.append(f'    Max Absolute Error (Jfwd - Jfd){stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {abs_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {abs_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if ('directional_fd_rev' in derivative_info and
                        derivative_info['directional_fd_rev'][i]):
                    err = _format_error(abs_errs[i].reverse, abs_error_tol)
                    parts.append('    Max Absolute Error ([rev, fd] Dot Product Test)'
                                 f'{stepstrs[i]} : {err}')
                    fd, rev = derivative_info['directional_fd_rev'][i]
                    parts.append(f'      rev value: {rev:.6e}')
                    parts.append(f'      fd value: {fd:.6e} ({fd_desc}{stepstrs[i]})\n')
            else:
                if abs_errs[i].forward is not None:
                    err = _format_error(abs_errs[i].forward, abs_error_tol)
                    parts.append(f'    Max Absolute Error (Jfwd - Jfd){stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {abs_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {abs_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if abs_errs[i].reverse is not None:
                    err = _format_error(abs_errs[i].reverse, abs_error_tol)
                    parts.append(f'    Max Absolute Error (Jrev - Jfd){stepstrs[i]} : {err}')
                    parts.append(f'      rev value: {abs_vals[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {abs_vals[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

        if directional:
            if ('directional_fwd_rev' in derivative_info and
                    derivative_info['directional_fwd_rev']):
                err = _format_error(abs_errs[0].fwd_rev, abs_error_tol)
                parts.append(f'    Max Absolute Error ([rev, fwd] Dot Product Test) : {err}')
                fwd, rev = derivative_info['directional_fwd_rev']
                parts.append(f'      rev value: {rev:.6e}')
                parts.append(f'      fwd value: {fwd:.6e}\n')
        elif abs_errs[0].fwd_rev is not None:
            err = _format_error(abs_errs[0].fwd_rev, abs_error_tol)
            parts.append(f'    Max Absolute Error (Jrev - Jfwd) : {err}')
            parts.append(f'      rev value: {abs_vals[0].fwd_rev[0]:.6e}')
            parts.append(f'      fwd value: {abs_vals[0].fwd_rev[1]:.6e}\n')

        divname = {
            'fwd': ['Jfwd', 'Jfd'],
            'rev': ['Jrev', 'Jfd'],
            'fwd_rev': ['Jrev', 'Jfwd']
        }

        for i in range(len(abs_errs)):
            didxs = denom_idxs[i]
            divname_fwd = divname['fwd'][didxs['fwd']]
            divname_rev = divname['rev'][didxs['rev']]
            divname_fwd_rev = divname['fwd_rev'][didxs['fwd_rev']]

            # Relative Errors
            if directional:
                if totals and rel_errs[i].forward is not None:
                    err = _format_error(rel_errs[i].forward, rel_error_tol)
                    parts.append(f'    Max Relative Error (Jfwd - Jfd) / {divname_fwd}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {rel_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if ('directional_fd_rev' in derivative_info and
                        derivative_info['directional_fd_rev'][i]):
                    err = _format_error(rel_errs[i].reverse, rel_error_tol)
                    parts.append(f'    Max Relative Error ([rev, fd] Dot Product Test) '
                                 f'/ {divname_rev}{stepstrs[i]} : {err}')
                    parts.append(f'      rev value: {rel_vals[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')
            else:
                if rel_errs[i].forward is not None:
                    err = _format_error(rel_errs[i].forward, rel_error_tol)
                    parts.append(f'    Max Relative Error (Jfwd - Jfd) / {divname_fwd}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {rel_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if rel_errs[i].reverse is not None:
                    err = _format_error(rel_errs[i].reverse, rel_error_tol)
                    parts.append(f'    Max Relative Error (Jrev - Jfd) / {divname_rev}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      rev value: {rel_vals[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

        if rel_errs[0].fwd_rev is not None:
            if directional:
                err = _format_error(rel_errs[0].fwd_rev, rel_error_tol)
                parts.append(f'    Max Relative Error ([rev, fwd] Dot Product Test) / '
                             f'{divname_fwd_rev} : {err}')
                rev, fwd = derivative_info['directional_fwd_rev']
                parts.append(f'      rev value: {rev:.6e}')
                parts.append(f'      fwd value: {fwd:.6e}\n')
            else:
                err = _format_error(rel_errs[0].fwd_rev, rel_error_tol)
                parts.append(f'    Max Relative Error (Jrev - Jfwd) / {divname_fwd_rev} : '
                             f'{err}')
                parts.append(f'      rev value: {rel_vals[0].fwd_rev[0]:.6e}')
                parts.append(f'      fwd value: {rel_vals[0].fwd_rev[1]:.6e}\n')

        if inconsistent:
            msg = '* Inconsistent value across ranks *'
            parts.append(f'\n    {_rich_wrap(msg, _Style.ABS_ERR)}\n')

        comm = system._problem_meta['comm']
        if MPI and comm.size > 1:
            parts.append(f'\n    MPI Rank {comm.rank}\n')

        uncovered_nz = derivative_info.get('uncovered_nz', None)
        if uncovered_nz is not None:
            uncovered_threshold = derivative_info['uncovered_threshold']
            rs = np.array([r for r, _ in uncovered_nz], dtype=int)
            cs = np.array([c for _, c in uncovered_nz])
            msg = (f'    Sparsity excludes {len(uncovered_nz)} entries which'
                   f' appear to be non-zero. (Magnitudes exceed {uncovered_threshold}) *')
            msg = _rich_wrap(msg, _Style.ABS_ERR)
            parts.append(textwrap.indent(msg, '    '))
            with np.printoptions(linewidth=1000, formatter={'int': lambda i: f'{i:>4d}'}):
                parts.append(textwrap.indent(f'Rows: {rs}', '      '))
                parts.append(textwrap.indent(f'Cols: {cs}\n', '      '))

        try:
            fds = derivative_info['J_fd']
        except KeyError:
            fds = [0.]

        jac_fmt = _JacFormatter(Jfwd.shape,
                                nzrows=derivative_info['rows'],
                                nzcols=derivative_info['cols'],
                                uncovered=uncovered_nz,
                                Jref=fds[0],
                                abs_err_tol=abs_error_tol,
                                rel_err_tol=rel_error_tol)

        with np.printoptions(linewidth=max(np.get_printoptions()['linewidth'], 10000),
                             formatter={'all': jac_fmt}):
            # Raw Derivatives

            if abs_errs[0].forward is not None:
                if directional:
                    parts.append('    Directional Derivative (Jfwd)')
                else:
                    parts.append('    Raw Forward Derivative (Jfwd)')
                    ss = StringIO()
                    print(Jfwd, file=ss)
                    Jstr = textwrap.indent(ss.getvalue(), '    ')
                    parts.append(f"{Jstr}\n")
                    jac_fmt.reset()

            fdtype = fd_opts['method'].upper()

            if abs_errs[0].reverse is not None:
                if directional:
                    if totals:
                        parts.append('    Directional Derivative (Jrev) Dot Product')
                    else:
                        parts.append('    Directional Derivative (Jrev)')
                else:
                    parts.append('    Raw Reverse Derivative (Jrev)')
                    ss = StringIO()
                    print(Jrev, file=ss)
                    Jstr = textwrap.indent(ss.getvalue(), '    ')
                    jac_fmt.reset()
                    parts.append(f"{Jstr}\n")

            for i in range(len(abs_errs)):
                fd = fds[i]

                if directional:
                    if totals and abs_errs[i].reverse is not None:
                        parts.append(f'    Directional {fdtype} Derivative (Jfd) '
                                     f'Dot Product{stepstrs[i]}\n{Jstr}\n')
                    else:
                        parts.append(f"    Directional {fdtype} Derivative (Jfd)"
                                     f"{stepstrs[i]}\n{Jstr}\n")
                else:
                    parts.append(f"    Raw {fdtype} Derivative (Jfd){stepstrs[i]}")
                    ss = StringIO()
                    print(fd, file=ss)
                    Jstr = textwrap.indent(ss.getvalue(), '    ')
                    parts.append(f"{Jstr}\n")

        parts.append(' -' * 30)
        parts.append('')

    sys_buffer.write('\n'.join(parts))

    if not show_only_incorrect or num_bad_jacs > 0:

        if rich is not None:
            c = console or Console(file=out_stream, force_terminal=True, record=True)
            c.print('\n'.join(parts), highlight=False)
            if console and system.get_reports_dir().is_dir():
                report = system.get_reports_dir() / f'{system.pathname}_derivs.html'
                c.save_html(report)
        else:
            out_stream.write(sys_buffer.getvalue())


def _deriv_display_compact(system, err_iter, derivatives, out_stream, totals=False,
                           show_only_incorrect=False, show_worst=False):
    """
    Print derivative error info to out_stream in a compact tabular format.

    Parameters
    ----------
    system : System
        The system for which derivatives are being displayed.
    err_iter : iterator
        Iterator that yields tuples of the form (key, fd_norm, fd_opts, directional, above_abs,
        above_rel, inconsistent) for each subjac.
    derivatives : dict
        Dictionary containing derivative information keyed by (of, wrt).
    out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
    totals : bool
        True if derivatives are totals.
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    show_worst : bool
        Set to True to show the worst subjac.

    Returns
    -------
    tuple or None
        Tuple contains the worst relative error, corresponding table row, and table header.
    """
    if out_stream is None:
        return

    from openmdao.core.component import Component

    # Match header to appropriate type.
    if isinstance(system, Component):
        sys_type = 'Component'
    else:
        sys_type = 'Group'

    sys_name = system.pathname
    sys_class_name = type(system).__name__
    matrix_free = system.matrix_free and not totals

    if totals:
        sys_name = 'Full Model'

    num_bad_jacs = 0  # Keep track of number of bad derivative values for each component

    # Need to capture the output of a component's derivative
    # info so that it can be used if that component is the
    # worst subjac. That info is printed at the bottom of all the output
    sys_buffer = StringIO()

    if totals:
        title = "Total Derivatives"
    else:
        title = f"{sys_type}: {sys_class_name} '{sys_name}'"

    print(f"{add_border(title, '-')}\n", file=sys_buffer)

    table_data = []
    worst_subjac = None

    for key, _, directional, above_abs, above_rel, inconsistent in err_iter:

        if above_abs or above_rel or inconsistent:
            num_bad_jacs += 1

        of, wrt = key
        derivative_info = derivatives[key]

        # Informative output for responses that were declared with an index.
        indices = derivative_info.get('indices')
        if indices is not None:
            of = f'{of} (index size: {indices})'

        if directional:
            wrt = f"(d) {wrt}"

        err_desc = []
        if above_abs:
            err_desc.append(' >ABS_TOL')
        if above_rel:
            err_desc.append(' >REL_TOL')
        if inconsistent:
            err_desc.append(' <RANK INCONSISTENT>')
        if 'uncovered_nz' in derivative_info:
            err_desc.append(' <BAD SPARSITY>')
        err_desc = ''.join(err_desc)

        abs_errs = derivative_info['abs error']
        rel_errs = derivative_info['rel error']
        abs_vals = derivative_info['vals_at_max_abs']
        rel_vals = derivative_info['vals_at_max_rel']
        steps = derivative_info['steps']

        # loop over different fd step sizes
        for abs_err, rel_err, abs_val, rel_val, step in zip(abs_errs, rel_errs,
                                                            abs_vals, rel_vals,
                                                            steps):

            # use forward even if both fwd and rev are defined
            if abs_err.forward is not None:
                calc_abs = abs_err.forward
                calc_rel = rel_err.forward
                calc_abs_val_fd = abs_val.forward[1]
                calc_rel_val_fd = rel_val.forward[1]
                calc_abs_val = abs_val.forward[0]
                calc_rel_val = rel_val.forward[0]
            elif abs_err.reverse is not None:
                calc_abs = abs_err.reverse
                calc_rel = rel_err.reverse
                calc_abs_val_fd = abs_val.reverse[1]
                calc_rel_val_fd = rel_val.reverse[1]
                calc_abs_val = abs_val.reverse[0]
                calc_rel_val = rel_val.reverse[0]

            start = [of, wrt, step] if len(steps) > 1 else [of, wrt]

            if totals:
                table_data.append(start +
                                  [calc_abs_val, calc_abs_val_fd, calc_abs,
                                   calc_rel_val, calc_rel_val_fd, calc_rel,
                                   err_desc])
            else:  # partials
                if matrix_free:
                    table_data.append(start +
                                      [abs_val.forward[0], abs_val.forward[1],
                                       abs_err.forward,
                                       abs_val.reverse[0], abs_val.reverse[1],
                                       abs_err.reverse,
                                       abs_val.fwd_rev[0], abs_val.fwd_rev[1],
                                       abs_err.fwd_rev,
                                       rel_val.forward[0], rel_val.forward[1],
                                       rel_err.forward,
                                       rel_val.reverse[0], rel_val.reverse[1],
                                       rel_err.reverse,
                                       rel_val.fwd_rev[0], rel_val.fwd_rev[1],
                                       rel_err.fwd_rev,
                                       err_desc])
                else:
                    if abs_val.forward is not None:
                        table_data.append(start +
                                          [abs_val.forward[0], abs_val.forward[1],
                                           abs_err.forward,
                                           rel_val.forward[0], rel_val.forward[1],
                                           rel_err.forward,
                                           err_desc])
                    else:
                        table_data.append(start +
                                          [abs_val.reverse[0], abs_val.reverse[1],
                                           abs_err.reverse,
                                           rel_val.reverse[0], rel_val.reverse[1],
                                           rel_err.reverse,
                                           err_desc])

                    assert abs_err.fwd_rev is None
                    assert rel_err.fwd_rev is None

                # See if this subjacobian has the greater error in the derivative computation
                # compared to the other subjacobians so far
                if worst_subjac is None or rel_err.max() > worst_subjac[0]:
                    worst_subjac = (rel_err.max(), table_data[-1])

    headers = []
    if table_data:
        headers = ["'of' variable", "'wrt' variable"]
        if len(steps) > 1:
            headers.append('step')

        if matrix_free:
            headers.extend(['a(fwd val)', 'a(fd val)', 'a(fwd-fd)',
                            'a(rev val)', 'a(rchk val)', 'a(rev-fd)',
                            'a(fwd val)', 'a(rev val)', 'a(fwd-rev)',
                            'r(fwd val)', 'r(fd val)', 'r(fwd-fd)',
                            'r(rev val)', 'r(rchk val)', 'r(rev-fd)',
                            'r(fwd val)', 'r(rev val)', 'r(fwd-rev)',
                            'error desc'])
        else:
            headers.extend(['a(calc val)', 'a(fd val)', 'a(calc-fd)',
                            'r(calc val)', 'r(fd val)', 'r(calc-fd)',
                            'error desc'])

        _print_deriv_table(table_data, headers, sys_buffer)

        if show_worst and worst_subjac is not None:
            print(f"\nWorst Sub-Jacobian (relative error): {worst_subjac[0]}\n",
                  file=sys_buffer)
            _print_deriv_table([worst_subjac[1]], headers, sys_buffer)

    if not show_only_incorrect or num_bad_jacs > 0:
        out_stream.write(sys_buffer.getvalue())

    if worst_subjac is None:
        return None

    return worst_subjac + (headers,)


def _format_error(error, tol):
    """
    Format the error, flagging if necessary.

    Parameters
    ----------
    error : float
        The absolute or relative error.
    tol : float
        Tolerance above which errors are flagged

    Returns
    -------
    str
        Formatted and possibly flagged error.
    """
    s = f'{error:.6e}'
    if rich is None:
        if np.isnan(error) or error < tol:
            return s
        return f'{s} *'
    else:
        if error > tol:
            s = f'{s} *'
        wrap = 'bright_red' if error > tol else ''
        return _rich_wrap(s, wrap)


def _print_deriv_table(table_data, headers, out_stream, tablefmt='grid'):
    """
    Print a table of derivatives.

    Parameters
    ----------
    table_data : list
        List of lists containing the table data.
    headers : list
        List of column headers.
    out_stream : file-like object
        Where to send human readable output.
        Set to None to suppress.
    tablefmt : str
        The table format to use.
    """
    if table_data and out_stream is not None:
        num_col_meta = {'format': '{: 1.4e}'}
        column_meta = [{}, {}]
        column_meta.extend([num_col_meta.copy() for _ in range(len(headers) - 3)])
        column_meta.append({})
        print(generate_table(table_data, headers=headers, tablefmt=tablefmt,
                             column_meta=column_meta, missing_val='n/a'), file=out_stream)


class _JacFormatter:
    """
    A class

    Parameters
    ----------
    shape : tuple
        The shape of the jacobian matrix being printed.
    nzrows : array-like or None
        The nonzero rows in the sparsity pattern.
    nzcols : array-like or None
        The nonzero columns in the sparsity pattern.
    Jref : array-like or None
        A reference jacobian with which any values are checked for error.
    abs_err_tol : float
        The absolute error tolerance to signify errors in the element being printed.
    rel_err_tol : float
        The relative error tolerance to signify errors in the element being printed.
    show_uncovered : bool
        If True, highlight nonzero elements outside of the given sparsity pattern
        as erroneous.

    Attributes
    ----------
    _shape : tuple[int]
        Thes hape of the jacobian matrix being printed.
    _nonzero : array-like or None
        The nonzero rows and columns in the sparsity pattern.
    _Jref : array-like or None
        A reference jacobian with which any values are checked for error.
    _abs_err_tol : float
        The absolute error tolerance to signify errors in the element being printed.
    _rel_err_tol : float
        The relative error tolerance to signify errors in the element being printed.
    _show_uncovered : bool
        If True, highlight nonzero elements outside of the given sparsity pattern
        as erroneous.
    _uncovered_nz : list or None
        If given, the coordinates of the uncovered nonzeros in the sparsity pattern.
    _i : int
        An internal counter used to track the current row being printed.
    _j : int
        An internal counter used to track the current column being printed.
    """
    def __init__(self, shape, nzrows=None, nzcols=None, Jref=None,
                 abs_err_tol=1.0E-8, rel_err_tol=1.0E-8, uncovered=None):
        self._shape = shape

        if nzrows is not None and nzcols is not None:
            self._nonzero = list(zip(nzrows, nzcols))
        else:
            self._nonzero = None

        self._Jref = Jref

        self._abs_err_tol = abs_err_tol
        self._rel_err_tol = rel_err_tol

        self._uncovered = uncovered

        # _i and _j are used to track the current row/col being printed.
        self._i = 0
        self._j = 0

    def reset(self, Jref=_UNDEFINED):
        """
        Reset the row/column counters, and optionally provide
        a new reference jacobian for error calculation.

        Parameters
        ----------
        Jref : array-like
            A reference jacobian against any values are checked for error.
        """
        self._i = 0
        self._j = 0
        if Jref != _UNDEFINED:
            self._Jref = Jref

    def __call__(self, x):
        i, j = self._i, self._j
        Jref = self._Jref
        atol = self._abs_err_tol
        rtol = self._rel_err_tol

        has_sparsity = self._nonzero is not None

        # Default output, no format.
        s = f'{x: .6e}'

        if rich is not None:
            rich_fmt = set()
            if (Jref is not None and  atol is not None and  rtol is not None):
                abs_err, _, rel_err, _, _ = get_errors(x, Jref[i, j])
            else:
                abs_err = 0.0
                rel_err = 0.0

            if has_sparsity:
                if (i, j) in self._nonzero:
                    rich_fmt |= {_Style.IN_SPARSITY}
                    if abs_err > atol:
                        rich_fmt |= {_Style.ABS_ERR}
                    elif np.abs(x) == 0:
                        rich_fmt |= {_Style.WARN}
                else:
                    rich_fmt |= {_Style.OUT_SPARSITY}
                    if abs_err > atol:
                        rich_fmt |= {_Style.ABS_ERR}
                    elif self._uncovered is not None and (i, j) in self._uncovered:
                        rich_fmt |= {_Style.WARN}
            else:
                if abs_err > atol:
                    rich_fmt |= {_Style.ABS_ERR}
                elif rel_err > rtol:
                    rich_fmt |= {_Style.REL_ERR}

            s = _rich_wrap(s, *rich_fmt)

        # Increment the row and column being printed.
        self._j += 1
        if self._j >= self._shape[1]:
            self._j = 0
            self._i += 1
        return s