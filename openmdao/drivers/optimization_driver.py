"""Define a base class for all Drivers in OpenMDAO."""
import os

import numpy as np

from openmdao.core.driver import Driver
from openmdao.utils.om_warnings import issue_warning, DriverWarning
from openmdao.core.constants import _SetupStatus


class OptimizationDriver(Driver):
    """
    Base-class for all optimizing drivers.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    iter_count : int
        Keep track of iterations for case recording.
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    recording_options : <OptionsDictionary>
        Dictionary with driver recording options.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    _problem : weakref to <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistent way for drivers to declare what features they support.
    _designvars : dict
        Contains all design variable info.
    _designvars_discrete : list
        List of design variables that are discrete.
    _dist_driver_vars : dict
        Dict of constraints that are distributed outputs. Key is a 'user' variable name,
        typically promoted name or an alias. Values are (local indices, local sizes).
    _cons : dict
        Contains all constraint info.
    _objs : dict
        Contains all objective info.
    _responses : dict
        Contains all response info.
    _lin_dvs : dict
        Contains design variables relevant to linear constraints.
    _nl_dvs : dict
        Contains design variables relevant to nonlinear constraints.
    _remote_dvs : dict
        Dict of design variables that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_cons : dict
        Dict of constraints that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_objs : dict
        Dict of objectives that are remote on at least one proc. Values are
        (owning rank, size).
    _coloring_info : dict
        Metadata pertaining to total coloring.
    _total_jac_format : str
        Specifies the format of the total jacobian. Allowed values are 'flat_dict', 'dict', and
        'array'.
    _con_subjacs : dict
        Dict of sparse subjacobians for use with certain optimizers, e.g. pyOptSparseDriver.
        Keyed by sources and aliases.
    _total_jac : _TotalJacInfo or None
        Cached total jacobian handling object.
    _total_jac_linear : _TotalJacInfo or None
        Cached linear total jacobian handling object.
    result : DriverResult
        DriverResult object containing information for use in the optimization report.
    _has_scaling : bool
        If True, scaling has been set for this driver.
    """
    def __init__(self, **kwargs):
        """
        Initialize the driver.
        """
        super().__init__(**kwargs)

        # What the driver supports.
        self.supports._read_only = False
        self.supports['optimization'] = True
        self.supports._read_only = True

    def _get_inst_id(self):
        if self._problem is None:
            return None
        return f"{self._problem()._get_inst_id()}.driver"

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Driver.
        """
        default_desvar_behavior = os.environ.get('OPENMDAO_INVALID_DESVAR_BEHAVIOR', 'warn').lower()

        self.options.declare('invalid_desvar_behavior', values=('warn', 'raise', 'ignore'),
                             desc='Behavior of driver if the initial value of a design '
                                  'variable exceeds its bounds. The default value may be'
                                  'set using the `OPENMDAO_INVALID_DESVAR_BEHAVIOR` environment '
                                  'variable to one of the valid options.',
                             default=default_desvar_behavior)

    def _check_for_missing_objective(self):
        """
        Check for missing objective and raise error if no objectives found.
        """
        if len(self._objs) == 0:
            msg = "Driver requires objective to be declared"
            raise RuntimeError(msg)

    def _check_for_invalid_desvar_values(self):
        """
        Check for design variable values that exceed their bounds.

        This method's behavior is controlled by the OPENMDAO_INVALID_DESVAR environment variable,
        which may take on values 'ignore', 'raise'', 'warn'.
        - 'ignore' : Proceed without checking desvar bounds.
        - 'warn' : Issue a warning if one or more desvar values exceed bounds.
        - 'raise' : Raise an exception if one or more desvar values exceed bounds.

        These options are case insensitive.
        """
        if self.options['invalid_desvar_behavior'] != 'ignore':
            invalid_desvar_data = []
            for var, meta in self._designvars.items():
                _val = self._problem().get_val(var, units=meta['units'], get_remote=True)
                val = np.array([_val]) if np.ndim(_val) == 0 else _val  # Handle discrete desvars
                idxs = meta['indices']() if meta['indices'] else None
                flat_idxs = meta['flat_indices']
                scaler = meta['scaler'] if meta['scaler'] is not None else 1.
                adder = meta['adder'] if meta['adder'] is not None else 0.
                lower = meta['lower'] / scaler - adder
                upper = meta['upper'] / scaler - adder
                flat_val = val.ravel()[idxs] if flat_idxs else val[idxs].ravel()

                if (flat_val < lower).any() or (flat_val > upper).any():
                    invalid_desvar_data.append((var, val, lower, upper))
            if invalid_desvar_data:
                s = 'The following design variable initial conditions are out of their ' \
                    'specified bounds:'
                for var, val, lower, upper in invalid_desvar_data:
                    s += f'\n  {var}\n    val: {val.ravel()}' \
                         f'\n    lower: {lower}\n    upper: {upper}'
                s += '\nSet the initial value of the design variable to a valid value or set ' \
                     'the driver option[\'invalid_desvar_behavior\'] to \'ignore\'.'
                if self.options['invalid_desvar_behavior'] == 'raise':
                    raise ValueError(s)
                else:
                    issue_warning(s, category=DriverWarning)

    def get_exit_status(self):
        """
        Return exit status of driver run.

        Returns
        -------
        str
            String indicating result of driver run.
        """
        return 'SUCCESS' if self.result.success else 'FAIL'

    def scaling_report(self, outfile='driver_scaling_report.html', title=None, show_browser=True,
                        jac=True):
            """
            Generate a self-contained html file containing a detailed connection viewer.

            Optionally pops up a web browser to view the file.

            Parameters
            ----------
            outfile : str, optional
                The name of the output html file.  Defaults to 'driver_scaling_report.html'.
            title : str, optional
                Sets the title of the web page.
            show_browser : bool, optional
                If True, pop up a browser to view the generated html file. Defaults to True.
            jac : bool
                If True, show jacobian information.

            Returns
            -------
            dict
                Data used to create html file.
            """
            from openmdao.visualization.scaling_viewer.scaling_report import view_driver_scaling

            # Run the model if it hasn't been run yet.
            status = -1 if self._problem is None else self._problem()._metadata['setup_status']
            if status < _SetupStatus.POST_FINAL_SETUP:
                raise RuntimeError("Either 'run_model' or 'final_setup' must be called before the "
                                "scaling report can be generated.")

            prob = self._problem()
            if prob._run_counter < 0:
                prob.run_model()

            return view_driver_scaling(self, outfile=outfile, show_browser=show_browser, jac=jac,
                                    title=title)

    def _get_active_cons_and_dvs(self, feas_atol=1.e-6, feas_rtol=1.e-6, assume_dvs_active=False):
        """
        Obtain the constraints and design varaibles which are active.

        Active means the constraint or design variable is on the bound (or close enough
        that it satisfies np.isclose(val, bound, atol=feas_atol, rtol=feas_rtol))

        Parameters
        ----------
        feas_atol : float
            Feasibility absolute tolerance
        feas_rtol : float
            Feasibility relative tolerance
        assume_dvs_active : bool
            Override to force design variables to be treated as active.

        Returns:
        active_cons : dict[str: dict
            The names of the active constraints. For each active constraint,
            a dict of the active indices and the active bound (0='equals',
            -1='lower', 1='upper') is provided.
        active_dvs : list[str]
            The names of the active design variables. For each active design
            variable, a dict of the active indices and the active bound
            (0='equals', -1='lower', 1='upper') is provided. An active
            design variable bound of 'equal' is only possible when
            assume_dvs_active is True, and the design variables are
            returned as if they are on an active equality constraint.
        """
        active_cons = {}
        active_dvs = {}
        prob = self._problem()
        des_vars = self._designvars
        constraints = self._cons

        for constraint, con_options in constraints.items():
            constraint_value = np.copy(prob.get_val(constraint)).ravel()
            con_size = con_options['size']
            if con_options.get('equals', None) is not None:
                active_cons[constraint] = {'indices': np.arange(con_size, dtype=int),
                                           'active_bounds': np.zeros(con_size, dtype=int)}
            else:
                constraint_upper = con_options.get("upper", np.inf)
                constraint_lower = con_options.get("lower", -np.inf)

                upper_mask = np.logical_or(constraint_value > constraint_upper,
                                           np.isclose(constraint_value, constraint_upper,
                                                      atol=feas_atol, rtol=feas_rtol))
                upper_idxs = np.where(upper_mask)[0]
                lower_mask = np.logical_or(constraint_value < constraint_lower,
                                           np.isclose(constraint_value, constraint_lower,
                                                      atol=feas_atol, rtol=feas_rtol))
                lower_idxs = np.where(lower_mask)[0]

                active_idxs = sorted(np.concatenate((upper_idxs, lower_idxs)))
                active_bounds = [1 if idx in upper_idxs else -1 for idx in active_idxs]
                if active_idxs:
                    active_cons[constraint] = {'indices': active_idxs,
                                               'active_bounds': active_bounds}

        for des_var, des_var_options in des_vars.items():
            des_var_value = np.copy(prob.get_val(des_var)).ravel()
            des_var_upper = np.ravel(des_var_options.get("upper", np.inf))
            des_var_lower = np.ravel(des_var_options.get("lower", -np.inf))
            des_var_size = des_var_options['size']
            if assume_dvs_active:
                active_dvs[des_var] = {'indices': np.arange(des_var_size, dtype=int),
                                       'active_bounds': np.zeros(des_var_size, dtype=int)}
            else:
                upper_mask = np.logical_or(des_var_value > des_var_upper,
                                           np.isclose(des_var_value, des_var_upper,
                                                      atol=feas_atol, rtol=feas_rtol))
                upper_idxs = np.where(upper_mask)[0]
                lower_mask = np.logical_or(des_var_value < des_var_lower,
                                           np.isclose(des_var_value, des_var_lower,
                                                      atol=feas_atol, rtol=feas_rtol))
                lower_idxs = np.where(lower_mask)[0]

                active_idxs = sorted(np.concatenate((upper_idxs, lower_idxs)))
                active_bounds = [1 if idx in upper_idxs else -1 for idx in active_idxs]
                if active_idxs:
                    active_dvs[des_var] = {'indices': np.asarray(active_idxs, dtype=int),
                                           'active_bounds': np.asarray(active_bounds, dtype=int)}

        return active_cons, active_dvs

    def _unscale_lagrange_multipliers(self, multipliers):
        """
        Unscale the Lagrange multipliers from optimizer scaling to physical/model scaling.

        This method assumes that the optimizer is in a converged state, satisfying both the
        primal constraints as well as the optimality conditions.

        Parameters
        ----------
        active_constraints : Sequence[str]
            Active constraints/dvs in the optimization, determined using the
            get_active_cons_and_dvs method.
        multipliers : dict[str: ArrayLike]
            The Lagrange multipliers, in Driver-scaled units.

        Returns
        -------
        dict
            The Lagrange multipliers in model/physical units.
        """
        if len(self._objs) != 1:
            raise ValueError('Lagrange Multplier estimation requires that there '
                             f'be a single objective, but there are {len(self._objs)}.')

        obj_meta = list(self._objs.values())[0]
        obj_ref = obj_meta['ref']
        obj_ref0 = obj_meta['ref0']

        if obj_ref is None:
            obj_ref = 1.0
        if obj_ref0 is None:
            obj_ref0 = 0.0

        unscaled_multipliers = {}

        for name, val in multipliers.items():
            if name in self._designvars:
                ref = self._designvars[name]['ref']
                ref0 = self._designvars[name]['ref0']
            else:
                for response in self._responses.values():
                    if response['name'] == name:
                        ref = response['ref']
                        ref0 = response['ref0']

            if ref is None:
                ref = 1.0
            if ref0 is None:
                ref0 = 0.0

            unscaled_multipliers[name] = val * (obj_ref - obj_ref0) / (ref - ref0)

        return unscaled_multipliers

    def compute_lagrange_multipliers(self, driver_scaling=False, feas_tol=1.0E-6):
        """
        Get the approximated Lagrange multipliers of one or more constraints.

        This method assumes that the optimizer is in a converged state, satisfying both the
        primal constraints as well as the optimality conditions.

        The estimation of which constraints are active depends upon the feasibility tolerance
        specified. This applies to the driver-scaled values of the constraints, and should be
        the same as that used by the optimizer, if available.

        Optimizers which provide their Lagrange multipliers may override this method.

        Parameters
        ----------
        driver_scaling : bool
            If False, return the Lagrange multipliers estimates in their physical units.
            If True, return the Lagrange multiplier estimates in a driver-scaled state.
        feas_tol : float or None
            The feasibility tolerance under which the optimization was run. If None, attempt
            to determine this automatically based on the specified optimizer settings.

        Returns
        -------
        multipliers : dict[str: ArrayLike]
            A dictionary with an entry for each active constraint and the
            associated Lagrange multiplier value.
        active_info : dict[str: dict
            A dictionary with an entry for each active constraint and its
            active indices and bounds.
        """
        prob = self._problem()

        objective = list(self._objs.keys())[0]
        constraints = self._cons
        des_vars = self._designvars

        of_totals = {objective, *constraints.keys()}
        wrt_totals = {*des_vars.keys()}

        active_cons, active_dvs = self._get_active_cons_and_dvs(feas_atol=feas_tol,
                                                                feas_rtol=feas_tol)

        totals = prob.compute_totals(list(of_totals),
                                     list(wrt_totals),
                                     driver_scaling=True)

        grad_f = {inp: totals[objective, inp] for inp in des_vars.keys()}

        n = 0
        for inp in grad_f.keys():
            n += grad_f[inp].size

        grad_f_vec = np.zeros((n))
        offset = 0
        for inp in grad_f.keys():
            inp_size = grad_f[inp].size
            grad_f_vec[offset:offset + inp_size] = grad_f[inp]
            offset += inp_size

        actives = active_cons | active_dvs
        n_active = np.sum(np.fromiter((len(c['indices']) for c in actives.values()), dtype=int))
        active_cons_mat = np.zeros((n, n_active))
        active_jac_blocks = []

        if n_active > 0:
            # TODO: Convert this to a sparse nonlinear least squares.
            for i, (con_name, active_meta) in enumerate(actives.items()):
                # If the constraint is a design variable, the constraint gradient
                # wrt des vars is just an identity matrix sized by the number of
                # active elements in the design variable.
                active_idxs = active_meta['indices']
                if con_name in des_vars.keys():
                    size = des_vars[con_name]['size']
                    con_grad = {(con_name, inp): np.eye(size)[active_idxs, ...] if inp == con_name
                                else np.zeros((size, size))[active_idxs, ...]
                                for inp in des_vars.keys()}
                else:
                    con_grad = {(con_name, inp): totals[con_name, inp][active_idxs, ...]
                                for inp in des_vars.keys()}
                active_jac_blocks.append(list(con_grad.values()))

            active_cons_mat = np.block(active_jac_blocks)
        else:
            return {}, actives

        multipliers_vec, optimality_squared, rank, singular_vals = \
            np.linalg.lstsq(active_cons_mat.T, -grad_f_vec, rcond=None)

        multipliers = dict()
        offset = 0

        opts = ['indices', 'ref0', 'ref', 'units']
        driver_vars = prob.list_driver_vars(out_stream=None,
                                            desvar_opts=opts + ['flat_indices'],
                                            cons_opts=opts + ['flat_indices'],
                                            objs_opts=opts,
                                            return_format='dict')
        driver_vars = driver_vars['design_vars'] | driver_vars['constraints']

        for constraint, act_info in actives.items():
            act_idxs = act_info['indices']
            active_size = len(act_idxs)
            mult_vals = multipliers_vec[offset:offset + active_size]

            shape = prob.get_val(constraint).shape
            opt_idxs = driver_vars[constraint]['indices']
            opt_flat_idxs = driver_vars[constraint]['indices']
            if opt_idxs is not None:
                if opt_flat_idxs:
                    flat_opt_idxs = opt_idxs
                else:
                    flat_opt_idxs = np.ravel_multi_index(opt_idxs)
                unraveled_idxs = np.unravel_index(flat_opt_idxs[act_idxs], shape)
            else:
                unraveled_idxs = np.unravel_index(act_idxs, shape)
            multipliers[constraint] = np.zeros(shape)
            multipliers[constraint][unraveled_idxs] = mult_vals
            offset += active_size

        if not driver_scaling:
            self._unscale_lagrange_multipliers(multipliers)

        return multipliers, actives
