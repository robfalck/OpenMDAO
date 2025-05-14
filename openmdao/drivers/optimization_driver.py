"""Define a base class for all Drivers in OpenMDAO."""
from itertools import chain
import pprint
import sys
import os
import weakref

import numpy as np

from openmdao.core.driver import Driver, DriverResult, SaveDriverResult, filter_by_meta, \
    record_iteration, RecordingDebugging
from openmdao.core.group import Group
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.core.constants import INT_DTYPE, _SetupStatus
from openmdao.utils.record_util import create_local_meta, check_path, has_match
from openmdao.utils.general_utils import _src_name_iter
from openmdao.utils.mpi import MPI
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.array_utils import sizes2offsets
from openmdao.vectors.vector import _full_slice, _flat_full_indexer
from openmdao.utils.indexer import indexer
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning, \
    DriverWarning, warn_deprecation


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

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        return type(self).__name__

    def add_recorder(self, recorder):
        """
        Add a recorder to the driver.

        Parameters
        ----------
        recorder : CaseRecorder
           A recorder instance.
        """
        self._rec_mgr.append(recorder)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        # shut down all recorders
        self._rec_mgr.shutdown()

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

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        return comm

    def _set_problem(self, problem):
        """
        Set a reference to the containing Problem.

        Parameters
        ----------
        problem : <Problem>
            Reference to the containing problem.
        """
        self._problem = weakref.ref(problem)

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        model = problem.model

        self._total_jac = None

        # Determine if any design variables are discrete.
        self._designvars_discrete = [name for name, meta in self._designvars.items()
                                     if meta['source'] in model._discrete_outputs]
        if not self.supports['integer_design_vars'] and len(self._designvars_discrete) > 0:
            msg = "Discrete design variables are not supported by this driver: "
            msg += '.'.join(self._designvars_discrete)
            raise RuntimeError(msg)

        self._split_dvs(model)

        self._remote_dvs = remote_dv_dict = {}
        self._remote_cons = remote_con_dict = {}
        self._dist_driver_vars = dist_dict = {}
        self._remote_objs = remote_obj_dict = {}

        # Only allow distributed design variables on drivers that support it.
        if self.supports['distributed_design_vars'] is False:
            dist_vars = []
            abs2meta_in = model._var_allprocs_abs2meta['input']
            discrete_in = model._var_allprocs_discrete['input']
            for dv, meta in self._designvars.items():

                # For Auto-ivcs, we need to check the distributed metadata on the target instead.
                if meta['source'].startswith('_auto_ivc.'):
                    for abs_name in model._var_allprocs_prom2abs_list['input'][dv]:
                        # we can use abs name to check for discrete vars here because
                        # relative names are absolute names at the model level.
                        if abs_name in discrete_in:
                            # Discrete vars aren't distributed.
                            break

                        if abs2meta_in[abs_name]['distributed']:
                            dist_vars.append(dv)
                            break
                elif meta['distributed']:
                    dist_vars.append(dv)

            if dist_vars:
                dstr = ', '.join(dist_vars)
                msg = "Distributed design variables are not supported by this driver, but the "
                msg += f"following variables are distributed: [{dstr}]"
                raise RuntimeError(msg)

        # Now determine if later we'll need to allgather cons, objs, or desvars.
        if model.comm.size > 1:
            loc_vars = set(model._outputs._abs_iter())
            # some of these lists could have duplicate src names if aliases are used. We'll
            # fix that when we convert to sets after the allgather.
            remote_dvs = [n for n in _src_name_iter(self._designvars) if n not in loc_vars]
            remote_cons = [n for n in _src_name_iter(self._cons) if n not in loc_vars]
            remote_objs = [n for n in _src_name_iter(self._objs) if n not in loc_vars]

            con_set = set()
            obj_set = set()
            dv_set = set()

            all_remote_vois = model.comm.allgather((remote_dvs, remote_cons, remote_objs))
            for rem_dvs, rem_cons, rem_objs in all_remote_vois:
                con_set.update(rem_cons)
                obj_set.update(rem_objs)
                dv_set.update(rem_dvs)

            # If we have remote VOIs, pick an owning rank for each and use that
            # to bcast to others later
            owning_ranks = model._owning_rank
            abs2idx = model._var_allprocs_abs2idx
            abs2meta_out = model._var_allprocs_abs2meta['output']
            sizes = model._var_sizes['output']
            rank = model.comm.rank
            nprocs = model.comm.size

            # Loop over all VOIs.
            for vname, voimeta in chain(self._responses.items(), self._designvars.items()):
                # vname may be a promoted name or an alias

                indices = voimeta['indices']
                vsrc = voimeta['source']

                meta = abs2meta_out[vsrc]
                i = abs2idx[vsrc]

                if meta['distributed']:

                    dist_sizes = sizes[:, i]
                    tot_size = np.sum(dist_sizes)

                    # Determine which indices are on our proc.
                    offsets = sizes2offsets(dist_sizes)

                    if indices is not None:
                        indices = indices.shaped_array()
                        true_sizes = np.zeros(nprocs, dtype=INT_DTYPE)
                        for irank in range(nprocs):
                            dist_inds = indices[np.logical_and(indices >= offsets[irank],
                                                               indices < (offsets[irank] +
                                                                          dist_sizes[irank]))]
                            true_sizes[irank] = dist_inds.size
                            if irank == rank:
                                local_indices = dist_inds - offsets[rank]
                                distrib_indices = dist_inds

                        ind = indexer(local_indices, src_shape=(tot_size,), flat_src=True)
                        dist_dict[vname] = (ind, true_sizes, distrib_indices)
                    else:
                        dist_dict[vname] = (_flat_full_indexer, dist_sizes,
                                            slice(offsets[rank], offsets[rank] + dist_sizes[rank]))

                else:
                    owner = owning_ranks[vsrc]
                    sz = sizes[owner, i]

                    if vsrc in dv_set:
                        remote_dv_dict[vname] = (owner, sz)
                    if vsrc in con_set:
                        remote_con_dict[vname] = (owner, sz)
                    if vsrc in obj_set:
                        remote_obj_dict[vname] = (owner, sz)

        self._remote_responses = self._remote_cons.copy()
        self._remote_responses.update(self._remote_objs)

        # set up simultaneous deriv coloring
        if coloring_mod._use_total_sparsity:
            # reset the coloring
            if self._coloring_info.dynamic or self._coloring_info.static is not None:
                self._coloring_info.coloring = None

            coloring = self._get_static_coloring()
            if coloring is not None and self.supports['simultaneous_derivatives']:
                if model._owns_approx_jac:
                    coloring._check_config_partial(model)
                else:
                    coloring._check_config_total(self, model)

                if not problem.model._use_derivatives:
                    issue_warning("Derivatives are turned off.  Skipping simul deriv coloring.",
                                  category=DerivativesWarning)

    def _split_dvs(self, model):
        """
        Determine which design vars are relevant to linear constraints vs nonlinear constraints.

        For some optimizers, this information will be used to determine the columns of the total
        linear jacobian vs. the total nonlinear jacobian.

        Parameters
        ----------
        model : <Group>
            The model being used in the optimization problem.
        """
        lin_cons = tuple([meta['source'] for meta in self._cons.values() if meta['linear']])
        if lin_cons:
            relevance = model._relevance
            dvs = tuple([meta['source'] for meta in self._designvars.values()])

            with relevance.seeds_active(fwd_seeds=dvs, rev_seeds=lin_cons):
                self._lin_dvs = {dv: meta for dv, meta in self._designvars.items()
                                 if relevance.is_relevant(meta['source'])}

            nl_resps = [meta['source'] for meta in self._cons.values() if not meta['linear']]
            nl_resps.extend([meta['source'] for meta in self._objs.values()])

            with relevance.seeds_active(fwd_seeds=dvs, rev_seeds=tuple(nl_resps)):
                self._nl_dvs = {dv: meta for dv, meta in self._designvars.items()
                                if relevance.is_relevant(meta['source'])}

        else:
            self._lin_dvs = {}
            self._nl_dvs = self._designvars

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

    def _get_vars_to_record(self, obj=None):
        """
        Get variables to record based on recording options.

        Parameters
        ----------
        obj : Problem or Driver
            Parent object which has recording options.

        Returns
        -------
        dict
           Dictionary containing lists of variables to record.
        """
        if obj is None:
            obj = self

        recording_options = obj.recording_options

        problem = self._problem()
        model = problem.model

        incl = recording_options['includes']
        excl = recording_options['excludes']

        # includes and excludes for outputs are specified using promoted names
        # includes and excludes for inputs are specified using _absolute_ names
        abs2prom_output = model._var_allprocs_abs2prom['output']
        abs2prom_inputs = model._var_allprocs_abs2prom['input']

        # set of promoted output names and absolute input and residual names
        # used for matching includes/excludes
        match_names = set()

        # 1. If record_outputs is True, get the set of outputs
        # 2. Filter those using includes and excludes to get the baseline set of variables to record
        # 3. Add or remove from that set any desvars, objs, and cons based on the recording
        #    options of those

        # includes and excludes for outputs are specified using _promoted_ names
        # vectors are keyed on absolute name, discretes on relative/promoted name
        myinputs = myoutputs = myresiduals = []

        if recording_options['record_outputs']:
            match_names = match_names | set(abs2prom_output.values())
            myoutputs = [n for n, prom in abs2prom_output.items() if check_path(prom, incl, excl)]

            model_outs = model._outputs

            if model._var_discrete['output']:
                # if we have discrete outputs then residual name set doesn't match output one
                if recording_options['record_residuals']:
                    myresiduals = [n for n in myoutputs if model_outs._contains_abs(n)]
            elif recording_options['record_residuals']:
                myresiduals = myoutputs

        elif recording_options['record_residuals']:
            match_names = match_names | set(model._residuals.keys())
            myresiduals = [n for n in model._residuals._abs_iter()
                           if check_path(abs2prom_output[n], incl, excl)]

        myoutputs = set(myoutputs)
        if recording_options['record_desvars']:
            myoutputs.update(_src_name_iter(self._designvars))
        if recording_options['record_objectives'] or recording_options['record_responses']:
            myoutputs.update(_src_name_iter(self._objs))
        if recording_options['record_constraints'] or recording_options['record_responses']:
            myoutputs.update(_src_name_iter(self._cons))

        # inputs (if in options). inputs use _absolute_ names for includes/excludes
        if 'record_inputs' in recording_options:
            if recording_options['record_inputs']:
                match_names = match_names | set(abs2prom_inputs.keys())
                myinputs = [n for n in abs2prom_inputs if check_path(n, incl, excl)]

        # check that all exclude/include globs have at least one matching output or input name
        for pattern in excl:
            if not has_match(pattern, match_names):
                issue_warning(f"{obj.msginfo}: No matches for pattern '{pattern}' in "
                              "recording_options['excludes'].")
        for pattern in incl:
            if not has_match(pattern, match_names):
                issue_warning(f"{obj.msginfo}: No matches for pattern '{pattern}' in "
                              "recording_options['includes'].")

        # sort lists to ensure that vars are iterated over in the same order on all procs
        vars2record = {
            'input': sorted(myinputs),
            'output': sorted(myoutputs),
            'residual': sorted(myresiduals)
        }

        return vars2record

    def _setup_recording(self):
        """
        Set up case recording.
        """
        self._filtered_vars_to_record = self._get_vars_to_record()
        self._rec_mgr.startup(self, self._problem().comm)

    def _run(self):
        """
        Execute this driver.

        This calls the run() method, which should be overriden by the subclass.

        Returns
        -------
        DriverResult
            DriverResult object, containing information about the run.
        """
        problem = self._problem()
        model = problem.model

        if self.supports['optimization'] and problem.options['group_by_pre_opt_post']:
            if model._pre_components:
                with model._relevance.nonlinear_active('pre'):
                    self._run_solve_nonlinear()

            with SaveDriverResult(self):
                with model._relevance.nonlinear_active('iter'):
                    self.result.success = not self.run()

            if model._post_components:
                with model._relevance.nonlinear_active('post'):
                    self._run_solve_nonlinear()

        else:
            with SaveDriverResult(self):
                self.result.success = not self.run()

        return self.result

    def _get_voi_val(self, name, meta, remote_vois, driver_scaling=True,
                     get_remote=True, rank=None):
        """
        Get the value of a variable of interest (objective, constraint, or design var).

        This will retrieve the value if the VOI is remote.

        Parameters
        ----------
        name : str
            Name of the variable of interest.
        meta : dict
            Metadata for the variable of interest.
        remote_vois : dict
            Dict containing (owning_rank, size) for all remote vois of a particular
            type (design var, constraint, or objective).
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
        rank : int or None
            If not None, gather value to this rank only.

        Returns
        -------
        float or ndarray
            The value of the named variable of interest.
        """
        model = self._problem().model
        comm = model.comm
        get = model._outputs._abs_get_val
        indices = meta['indices']
        src_name = meta['source']

        if MPI:
            distributed = comm.size > 0 and name in self._dist_driver_vars
        else:
            distributed = False

        if name in remote_vois:
            owner, size = remote_vois[name]
            # if var is distributed or only gathering to one rank
            # TODO - support distributed var under a parallel group.
            if owner is None or rank is not None:
                val = model.get_val(src_name, get_remote=get_remote, rank=rank, flat=True)
                if indices is not None:
                    val = val[indices.flat()]
            else:
                if owner == comm.rank:
                    if indices is None:
                        val = get(src_name, flat=True).copy()
                    else:
                        val = get(src_name, flat=True)[indices.as_array()]
                else:
                    if indices is not None:
                        size = indices.indexed_src_size
                    val = np.empty(size)

                if get_remote:
                    comm.Bcast(val, root=owner)

        elif distributed:
            local_val = model.get_val(src_name, get_remote=False, flat=True)
            local_indices, sizes, _ = self._dist_driver_vars[name]
            if local_indices is not _full_slice:
                local_val = local_val[local_indices()]

            if get_remote:
                local_val = np.ascontiguousarray(local_val)
                offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                offsets[1:] = np.cumsum(sizes[:-1])
                val = np.zeros(np.sum(sizes))
                comm.Allgatherv(local_val, [val, sizes, offsets, MPI.DOUBLE])
            else:
                val = local_val

        else:
            if src_name in model._discrete_outputs:
                val = model._discrete_outputs[src_name]
                if name in self._designvars_discrete:
                    # At present, only integers are supported by OpenMDAO drivers.
                    # We check the values here.
                    if not ((np.isscalar(val) and isinstance(val, (int, np.integer))) or
                            (isinstance(val, np.ndarray) and np.issubdtype(val[0], np.integer))):
                        if np.isscalar(val):
                            suffix = f"A value of type '{type(val).__name__}' was specified."
                        elif isinstance(val, np.ndarray):
                            suffix = f"An array of type '{val.dtype.name}' was specified."
                        else:
                            suffix = ''
                        raise ValueError("Only integer scalars or ndarrays are supported as values "
                                         "for discrete variables when used as a design variable. "
                                         + suffix)
            elif indices is None:
                val = get(src_name, flat=True).copy()
            else:
                val = get(src_name, flat=True)[indices.as_array()]

        if self._has_scaling and driver_scaling:
            # Scale design variable values
            adder = meta['total_adder']
            if adder is not None:
                val += adder

            scaler = meta['total_scaler']
            if scaler is not None:
                val *= scaler

        return val

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        warn_deprecation('get_driver_objective_calls is deprecated. '
                         'Use `driver.result.model_evals`')
        return self.result.model_evals

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        warn_deprecation('get_driver_derivative_calls is deprecated. '
                         'Use `driver.result.deriv_evals`')
        return self.result.deriv_evals

    def get_design_var_values(self, get_remote=True, driver_scaling=True):
        """
        Return the design variable values.

        Parameters
        ----------
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        return {n: self._get_voi_val(n, dvmeta, self._remote_dvs, get_remote=get_remote,
                                     driver_scaling=driver_scaling)
                for n, dvmeta in self._designvars.items()}

    def set_design_var(self, name, value, set_remote=True):
        """
        Set the value of a design variable.

        'name' can be a promoted output name or an alias.

        Parameters
        ----------
        name : str
            Global pathname of the design variable.
        value : float or ndarray
            Value for the design variable.
        set_remote : bool
            If True, set the global value of the variable (value must be of the global size).
            If False, set the local value of the variable (value must be of the local size).
        """
        problem = self._problem()
        meta = self._designvars[name]

        src_name = meta['source']

        # if the value is not local, don't set the value
        if (src_name in self._remote_dvs and
                problem.model._owning_rank[src_name] != problem.comm.rank):
            return

        if name in self._designvars_discrete:

            # Note, drivers set values here and generally should know it is setting an integer.
            # However, the DOEdriver may pull a non-integer value from its generator, so we
            # convert it.
            if isinstance(value, float):
                value = int(value)
            elif isinstance(value, np.ndarray):
                if isinstance(problem.model._discrete_outputs[src_name], int):
                    # Setting an integer value with a 1D array - don't want to convert to array.
                    value = int(value.item())
                else:
                    value = value.astype(int)

            problem.model._discrete_outputs[src_name] = value

        elif problem.model._outputs._contains_abs(src_name):
            desvar = problem.model._outputs._abs_get_val(src_name)
            if name in self._dist_driver_vars:
                loc_idxs, _, dist_idxs = self._dist_driver_vars[name]
                loc_idxs = loc_idxs()  # don't use indexer here
            else:
                loc_idxs = meta['indices']
                if loc_idxs is None:
                    loc_idxs = _full_slice
                else:
                    loc_idxs = loc_idxs()
                dist_idxs = _full_slice

            if set_remote:
                # provided value is the global value, use indices for this proc
                desvar[loc_idxs] = np.atleast_1d(value)[dist_idxs]
            else:
                # provided value is the local value
                desvar[loc_idxs] = np.atleast_1d(value)

            # Undo driver scaling when setting design var values into model.
            if self._has_scaling:
                scaler = meta['total_scaler']
                if scaler is not None:
                    desvar[loc_idxs] *= 1.0 / scaler

                adder = meta['total_adder']
                if adder is not None:
                    desvar[loc_idxs] -= adder

    def get_objective_values(self, driver_scaling=True):
        """
        Return objective values.

        Parameters
        ----------
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        return {n: self._get_voi_val(n, obj, self._remote_objs,
                                     driver_scaling=driver_scaling)
                for n, obj in self._objs.items()}

    def get_constraint_values(self, ctype='all', lintype='all', driver_scaling=True):
        """
        Return constraint values.

        Parameters
        ----------
        ctype : str
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.
        lintype : str
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        con_dict = {}
        it = self._cons.items()
        if lintype == 'linear':
            it = filter_by_meta(it, 'linear')
        elif lintype == 'nonlinear':
            it = filter_by_meta(it, 'linear', exclude=True)
        if ctype == 'eq':
            it = filter_by_meta(it, 'equals', chk_none=True)
        elif ctype == 'ineq':
            it = filter_by_meta(it, 'equals', chk_none=True, exclude=True)

        for name, meta in it:
            con_dict[name] = self._get_voi_val(name, meta, self._remote_cons,
                                               driver_scaling=driver_scaling)

        return con_dict

    def _get_ordered_nl_responses(self):
        """
        Return the names of nonlinear responses in the order used by the driver.

        Default order is objectives followed by nonlinear constraints.  This is used for
        simultaneous derivative coloring and sparsity determination.

        Returns
        -------
        list of str
            The nonlinear response names in order.
        """
        order = list(self._objs)
        order.extend(n for n, meta in self._cons.items() if not meta['linear'])
        return order

    def _update_voi_meta(self, model, responses, desvars):
        """
        Collect response and design var metadata from the model and size desvars and responses.

        Parameters
        ----------
        model : System
            The System that represents the entire model.
        responses : dict
            Response metadata dictionary.
        desvars : dict
            Design variable metadata dictionary.

        Returns
        -------
        int
            Total size of responses, with linear constraints excluded.
        int
            Total size of design vars.
        """
        self._objs = objs = {}
        self._cons = cons = {}

        self._responses = responses
        self._designvars = desvars

        # driver _responses are keyed by either the alias or the promoted name
        response_size = 0
        for name, meta in responses.items():
            if meta['type'] == 'con':
                cons[name] = meta
                if meta['linear']:
                    continue  # don't add to response size
            else:
                objs[name] = meta

            response_size += meta['global_size']

        desvar_size = sum(meta['global_size'] for meta in desvars.values())

        self._has_scaling = model._setup_driver_units()

        return response_size, desvar_size

    def get_exit_status(self):
        """
        Return exit status of driver run.

        Returns
        -------
        str
            String indicating result of driver run.
        """
        return 'SUCCESS' if self.result.success else 'FAIL'

    def check_relevance(self):
        """
        Check if there are constraints that don't depend on any design vars.

        This usually indicates something is wrong with the problem formulation.
        """
        # relevance not relevant if not using derivatives
        if not self.supports['gradients']:
            return

        if 'singular_jac_behavior' in self.options:
            singular_behavior = self.options['singular_jac_behavior']
            if singular_behavior == 'ignore':
                return
        else:
            singular_behavior = 'warn'

        problem = self._problem()

        # Do not perform this check if any subgroup uses approximated partials.
        # This causes the relevance graph to be invalid.
        for system in problem.model.system_iter(include_self=True, recurse=True, typ=Group):
            if system._has_approx:
                return

        bad = {n for n in self._problem().model._relevance._no_dv_responses
               if n not in self._designvars}
        if bad:
            bad_conns = [n for n, m in self._cons.items() if m['source'] in bad]
            bad_objs = [n for n, m in self._objs.items() if m['source'] in bad]
            badmsg = []
            if bad_conns:
                badmsg.append(f"constraint(s) {bad_conns}")
            if bad_objs:
                badmsg.append(f"objective(s) {bad_objs}")
            bad = ' and '.join(badmsg)
            # Note: There is a hack in ScipyOptimizeDriver for older versions of COBYLA that
            #       implements bounds on design variables by adding them as constraints.
            #       These design variables as constraints will not appear in the wrt list.
            msg = f"{self.msginfo}: {bad} do not depend on any " \
                  "design variables. Please check your problem formulation."
            if singular_behavior == 'error':
                raise RuntimeError(msg)
            else:
                issue_warning(msg, category=DriverWarning)

    def run(self):
        """
        Execute this driver.

        The base `Driver` just runs the model. All other drivers overload
        this method.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        self.result.reset()
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()

        self.iter_count += 1

        return False

    @property
    def _recording_iter(self):
        return self._problem()._metadata['recording_iter']

    @DriverResult.track_stats(kind='model')
    def _run_solve_nonlinear(self):
        return self._problem().model.run_solve_nonlinear()

    @DriverResult.track_stats(kind='deriv')
    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', driver_scaling=True):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        All derivatives are returned using driver scaling.

        Parameters
        ----------
        of : list of variable name str or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name str or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : str
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt). For
            the scipy optimizer, 'array' is also supported.
        driver_scaling : bool
            If True (default), scale derivative values by the quantities specified when the desvars
            and responses were added. If False, leave them unscaled.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        problem = self._problem()
        debug_print = 'totals' in self.options['debug_print'] and (not MPI or
                                                                   problem.comm.rank == 0)

        if debug_print:
            header = 'Driver total derivatives for iteration: ' + str(self.iter_count)
            print(header)
            print(len(header) * '-' + '\n')

        if self._total_jac is None:
            total_jac = _TotalJacInfo(problem, of, wrt, return_format,
                                      approx=problem.model._owns_approx_jac,
                                      debug_print=debug_print,
                                      driver_scaling=driver_scaling)

            if total_jac.has_lin_cons and self.supports['linear_constraints']:
                self._total_jac_linear = total_jac
            else:
                self._total_jac = total_jac
        else:
            total_jac = self._total_jac

        totals = total_jac.compute_totals()

        if self.recording_options['record_derivatives']:
            self.record_derivatives()

        return totals

    def record_derivatives(self):
        """
        Record the current total jacobian.
        """
        if self._total_jac is not None and self._rec_mgr._recorders:
            metadata = create_local_meta(self._get_name())
            self._total_jac.record_derivatives(self, metadata)

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        status = -1 if self._problem is None else self._problem()._metadata['setup_status']
        if status >= _SetupStatus.POST_FINAL_SETUP:
            record_iteration(self, self._problem(), self._get_name())
        else:
            raise RuntimeError(f'{self.msginfo} attempted to record iteration but '
                               'driver has not been initialized; `run_model()`, '
                               '`run_driver()`, or `final_setup()` must be called '
                               'before recording.')

    def _get_recorder_metadata(self, case_name):
        """
        Return metadata from the latest iteration for use in the recorder.

        Parameters
        ----------
        case_name : str
            Name of current case.

        Returns
        -------
        dict
            Metadata dictionary for the recorder.
        """
        return create_local_meta(case_name)

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "Driver"

    def declare_coloring(self, num_full_jacs=coloring_mod._DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                         tol=coloring_mod._DEF_COMP_SPARSITY_ARGS['tol'],
                         orders=coloring_mod._DEF_COMP_SPARSITY_ARGS['orders'],
                         perturb_size=coloring_mod._DEF_COMP_SPARSITY_ARGS['perturb_size'],
                         min_improve_pct=coloring_mod._DEF_COMP_SPARSITY_ARGS['min_improve_pct'],
                         show_summary=coloring_mod._DEF_COMP_SPARSITY_ARGS['show_summary'],
                         show_sparsity=coloring_mod._DEF_COMP_SPARSITY_ARGS['show_sparsity'],
                         use_scaling=coloring_mod._DEF_COMP_SPARSITY_ARGS['use_scaling']):
        """
        Set options for total deriv coloring.

        Parameters
        ----------
        num_full_jacs : int
            Number of times to repeat partial jacobian computation when computing sparsity.
        tol : float
            Tolerance used to determine if an array entry is nonzero during sparsity determination.
        orders : int
            Number of orders above and below the tolerance to check during the tolerance sweep.
        perturb_size : float
            Size of input/output perturbation during generation of sparsity.
        min_improve_pct : float
            If coloring does not improve (decrease) the number of solves more than the given
            percentage, coloring will not be used.
        show_summary : bool
            If True, display summary information after generating coloring.
        show_sparsity : bool
            If True, display sparsity with coloring info after generating coloring.
        use_scaling : bool
            If True, use driver scaling when generating the sparsity.
        """
        self._coloring_info.coloring = None
        self._coloring_info.num_full_jacs = num_full_jacs
        self._coloring_info.tol = tol
        self._coloring_info.orders = orders
        self._coloring_info.perturb_size = perturb_size
        self._coloring_info.min_improve_pct = min_improve_pct
        if self._coloring_info.static is None:
            self._coloring_info.dynamic = True
        else:
            self._coloring_info.dynamic = False
        self._coloring_info.show_summary = show_summary
        self._coloring_info.show_sparsity = show_sparsity
        self._coloring_info.use_scaling = use_scaling

    def use_fixed_coloring(self, coloring=coloring_mod.STD_COLORING_FNAME):
        """
        Tell the driver to use a precomputed coloring.

        Parameters
        ----------
        coloring : str or Coloring
            A coloring filename or a Coloring object.  If no arg is passed, filename will be
            determined automatically.
        """
        if self.supports['simultaneous_derivatives']:
            if coloring_mod._force_dyn_coloring and coloring is coloring_mod.STD_COLORING_FNAME:
                # force the generation of a dynamic coloring this time
                self._coloring_info.dynamic = True
                self._coloring_info.static = None
            else:
                self._coloring_info.static = coloring
                self._coloring_info.dynamic = False

            self._coloring_info.coloring = None
        else:
            raise RuntimeError("Driver '%s' does not support simultaneous derivatives." %
                               self._get_name())

    def _setup_tot_jac_sparsity(self, coloring=None):
        """
        Set up total jacobian subjac sparsity.

        Drivers that can use subjac sparsity should override this.

        Parameters
        ----------
        coloring : Coloring or None
            Current coloring.
        """
        pass

    def _get_static_coloring(self):
        """
        Get the Coloring for this driver.

        If necessary, load the Coloring from a file.

        Returns
        -------
        Coloring or None
            The pre-existing or loaded Coloring, or None
        """
        coloring = None
        info = self._coloring_info
        static = info.static
        model = self._problem().model

        if isinstance(static, coloring_mod.Coloring):
            coloring = static
            info.coloring = coloring
        else:
            coloring = info.coloring

            if coloring is None and (static is coloring_mod.STD_COLORING_FNAME or
                                     isinstance(static, str)):
                if static is coloring_mod.STD_COLORING_FNAME:
                    fname = self._get_total_coloring_fname(mode='input')
                else:
                    fname = static

                print(f"loading total coloring from file {fname}")
                coloring = info.coloring = coloring_mod.Coloring.load(fname)
                info.update(coloring._meta)

                ofname = self._get_total_coloring_fname(mode='output')
                if ((model._full_comm is not None and model._full_comm.rank == 0) or
                        (model._full_comm is None and model.comm.rank == 0)):
                    coloring.save(ofname)

        if coloring is not None and info.static is not None:
            problem = self._problem()
            if coloring._rev and problem._orig_mode not in ('rev', 'auto'):
                revcol = coloring._rev[0][0]
                if revcol:
                    raise RuntimeError("Simultaneous coloring does reverse solves but mode has "
                                       f"been set to '{problem._orig_mode}'")
            if coloring._fwd and problem._orig_mode not in ('fwd', 'auto'):
                fwdcol = coloring._fwd[0][0]
                if fwdcol:
                    raise RuntimeError("Simultaneous coloring does forward solves but mode has "
                                       f"been set to '{problem._orig_mode}'")

        return coloring

    def _get_total_coloring_fname(self, mode='output'):
        return self._problem().get_coloring_dir(mode='output') / 'total_coloring.pkl'

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

    def _pre_run_model_debug_print(self):
        """
        Optionally print some debugging information before the model runs.
        """
        debug_opt = self.options['debug_print']
        rank = self._problem().comm.rank
        if not debug_opt or debug_opt == ['totals']:
            return

        if not MPI or rank == 0:
            header = 'Driver debug print for iter coord: {}'.format(
                self._recording_iter.get_formatted_iteration_coordinate())
            print(header)
            print(len(header) * '-')

        if 'desvars' in debug_opt:
            model = self._problem().model
            desvar_vals = {n: model.get_val(n, get_remote=True, rank=0) for n in self._designvars}
            if not MPI or rank == 0:
                print("Design Vars")
                if desvar_vals:
                    pprint.pprint(desvar_vals)
                else:
                    print("None")
                print()

        sys.stdout.flush()

    def _post_run_model_debug_print(self):
        """
        Optionally print some debugging information after the model runs.
        """
        rank = self._problem().comm.rank

        if 'nl_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='nonlinear', driver_scaling=False)
            if not MPI or rank == 0:
                print("Nonlinear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'ln_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='linear', driver_scaling=False)
            if not MPI or rank == 0:
                print("Linear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'objs' in self.options['debug_print']:
            objs = self.get_objective_values(driver_scaling=False)
            if not MPI or rank == 0:
                print("Objectives")
                if objs:
                    pprint.pprint(objs)
                else:
                    print("None")
                print()

        sys.stdout.flush()

    def get_reports_dir(self):
        """
        Get the path to the directory where the report files should go.

        If it doesn't exist, it will be created.

        Returns
        -------
        str
            The path to the directory where reports should be written.
        """
        return self._problem().get_reports_dir()

    def _get_coloring(self, run_model=None):
        """
        Get the total coloring for this driver.

        If necessary, dynamically generate it.

        Parameters
        ----------
        run_model : bool or None
            If False, don't run model, else use problem _run_counter to decide.
            This is ignored if the coloring has already been computed.

        Returns
        -------
        Coloring or None
            Coloring object, possible loaded from a file or dynamically generated, or None
        """
        if coloring_mod._use_total_sparsity:
            if run_model and self._coloring_info.coloring is not None:
                issue_warning("The 'run_model' argument is ignored because the coloring has "
                              "already been computed.")

            if self._coloring_info.dynamic and self._coloring_info.do_compute_coloring():
                ofname = self._get_total_coloring_fname(mode='output')
                self._coloring_info.coloring = \
                    coloring_mod.dynamic_total_coloring(self,
                                                        run_model=run_model,
                                                        fname=ofname)

            return self._coloring_info.coloring

    def _update_result(self, result):
        """
        Set additional attributes and information to the DriverResult.
        """
        pass

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
