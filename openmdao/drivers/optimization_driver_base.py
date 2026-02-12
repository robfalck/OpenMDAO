"""
Base class for optimization drivers.
"""
from enum import Enum

import numpy as np
import scipy as sp

from openmdao.core.driver import Driver
from openmdao.vectors.optimizer_vector import OptimizerVector


class VOIType(str, Enum):

    DESIGN_VAR = 'design_var'
    CONSTRAINT = 'constraint'
    OBJECTIVE = 'objective'

class OptimizationDriverBase(Driver):
    """
    Base class for optimization drivers using OptimizerVector interface.

    This class provides a unified interface for optimization drivers to work with
    cached OptimizerVector instances for design variables, constraints, and objectives.

    Attributes
    ----------
    _vectors : dict
        Dictionary containing OptimizerVector instances with keys:
        - 'design_vars': OptimizerVector for design variables
        - 'constraints': OptimizerVector for constraints
        - 'objectives': OptimizerVector for objectives
    """

    def __init__(self, **kwargs):
        """Initialize OptimizationDriverBase."""
        super().__init__(**kwargs)
        self._vectors: dict[str, OptimizerVector] = {
            VOIType.DESIGN_VAR: OptimizerVector(),
            VOIType.CONSTRAINT: OptimizerVector(),
            VOIType.OBJECTIVE: OptimizerVector(),
        }
        self._autoscaler = None

    @property
    def autoscaler(self):
        """
        Docstring for autoscaler

        Returns
        -------
        The Autoscaler used by this optimization driver.
        """
        return self._autoscaler

    @autoscaler.setter
    def autoscaler(self, autoscaler):
        self._autoscaler = autoscaler

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        if self.autoscaler is None:
            from openmdao.drivers.autoscalers.default_autoscaler import DefaultAutoscaler
            self._autoscaler = DefaultAutoscaler()

        # Allocate the vectors for the optimizer.
        self._vectors[VOIType.DESIGN_VAR] = self.get_vector_from_model(
            voi_type=VOIType.DESIGN_VAR, driver_scaling=True)
        
        self._vectors[VOIType.CONSTRAINT] = self.get_vector_from_model(
            voi_type=VOIType.CONSTRAINT, driver_scaling=True)

        self._vectors[VOIType.OBJECTIVE] = self.get_vector_from_model(
            voi_type=VOIType.OBJECTIVE, driver_scaling=True)

    def get_vector_from_model(self, voi_type: VOIType, driver_scaling=True,
                              out: OptimizerVector | None=None):
        """
        Get a new OptimizerVector from the model, or populate the data in an existing one.

        Note that when providing 

        Parameters
        ----------
        voi_type: VOIType
            The kind of variable interest being retrieved.
        driver_scaling: bool
            If True, return vector values in the optimizer-scaled space.
        out: OptimizerVector or None
            If None, return a new OptimizerVector. Otherwise, provides an existing
            OptimizerVector and its underlying data is populated in-place.
        
        Returns
        -------
        OptimizerVector
            If out is None, a new OptimizerVector is return, otherwise the vector data
            is set into the given OptimizerVector's underlying data in-place and a reference
            to out is returned.
        """
        varmeta_map = {VOIType.DESIGN_VAR: self._designvars,
                       VOIType.CONSTRAINT: self._cons,
                       VOIType.OBJECTIVE: self._objs}

        # Build metadata for OptimizerVector with flat array indexing
        vecmeta = {}
        dv_array = []
        idx = 0

        for name, meta in varmeta_map[voi_type].items():
            size = meta['size']
            start_idx = idx
            end_idx = idx + size

            if out is None:
                vecmeta[name] = {
                    'start_idx': idx,
                    'end_idx': idx + size,
                    'size': size,
                    'total_scaler': meta.get('total_scaler'),
                    'total_adder': meta.get('total_adder')
                }

                # Meta that only applies to constraints and/or design vars
                if 'linear' in meta:
                    vecmeta['linear'] = meta.get('linear', False)
                if 'equals' in meta:
                    vecmeta['equals'] = meta.get('equals')
                if 'lower' in meta:
                    vecmeta['equals'] = meta.get('equals')
                if 'upper' in meta:
                    vecmeta['equals'] = meta.get('equals')

                # Internally we never request voi with driver_scaling, instead
                # applying the autoscaler.
                val = self._get_voi_val(name, meta, self._remote_dvs,
                                        driver_scaling=False, get_remote=True)
                dv_array.append(np.atleast_1d(val).flat)
            else:
                # The vector already exists, just populate it.
                out.asarray()[start_idx:end_idx] = self._get_voi_val(name, meta, self._remote_dvs,
                                                                     driver_scaling=False,
                                                                     get_remote=True)
            
            idx += size

        if out is None:
            # Create flat array
            flat_array = np.concatenate(dv_array) if dv_array else np.array([])
            out = OptimizerVector(flat_array, vecmeta)

        # Apply autoscaler to the vector
        

        return out
        
    def set_design_vars(self):
        """
        Set design variable values into the model from a dictionary.

        Parameters
        ----------
        values_dict : dict
            Dictionary mapping design variable names to values.
        """
        for name, value in self._vectors[VOIType.DESIGN_VAR].items():
            self.set_design_var(name, value, set_remote=True, unscale=False)

    def _unscale_lagrange_multipliers(self, multipliers, assume_dv=False):
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
        assume_dv : bool
            This function can unscale the multipliers of either design variables or constraints.
            Since variables can be both a design variable and a constraint, this flag
            disambiguates the type of multiplier we're handling so the appropriate scaling
            factors can be used.

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

        obj_scaler = obj_meta['total_scaler'] or 1.0

        unscaled_multipliers = {}

        for name, val in multipliers.items():
            if name in self._designvars and assume_dv:
                scaler = self._designvars[name]['total_scaler']
            else:
                scaler = self._responses[name]['total_scaler']
            scaler = scaler or 1.0

            unscaled_multipliers[name] = val * scaler / obj_scaler

        return unscaled_multipliers

    def compute_lagrange_multipliers(self, driver_scaling=False, feas_tol=1.0E-6,
                                     use_sparse_solve=True):
        """
        Get the approximated Lagrange multipliers of one or more constraints.

        This method assumes that the optimizer is in a converged state, satisfying both the
        primal constraints as well as the optimality conditions.

        The estimation of which constraints are active depends upon the feasibility tolerance
        specified. This applies to the driver-scaled values of the constraints, and should be
        the same as that used by the optimizer, if available.

        Parameters
        ----------
        driver_scaling : bool
            If False, return the Lagrange multipliers estimates in their physical units.
            If True, return the Lagrange multiplier estimates in a driver-scaled state.
        feas_tol : float or None
            The feasibility tolerance under which the optimization was run. If None, attempt
            to determine this automatically based on the specified optimizer settings.
        use_sparse_solve : bool
            If True, use scipy.sparse.linalg.lstsq to solve for the multipliers. Otherwise, numpy
            will be used with dense arrays.

        Returns
        -------
        active_desvars : dict[str: dict]
            A dictionary with an entry for each active design variable.
            For each active design variable, the corresponding dictionary
            provides the 'multipliers', active 'indices', and 'active_bounds'.
        active_cons : dict[str: dict]
            A dictionary with an entry for each active constraint.
            For each active constraint, the corresponding dictionary
            provides the 'multipliers', active 'indices', and 'active_bounds'.
        """
        if not self.supports['optimization']:
            raise NotImplementedError('Lagrange multipliers are only available for '
                                      'drivers which support optimization.')

        prob = self._problem()

        obj_name = list(self._objs.keys())[0]
        constraints = self._cons
        des_vars = self._designvars

        of_totals = {obj_name, *constraints.keys()}

        active_cons, active_dvs = self._get_active_cons_and_dvs(feas_atol=feas_tol,
                                                                feas_rtol=feas_tol)

        # Active cons and dvs provide the active indices in the design vars and constraints.
        # But these design vars and constraints may themselves be indices of a larger
        # variable.
        totals = prob.compute_totals(list(of_totals),
                                     list(des_vars),
                                     driver_scaling=True)

        grad_f = {inp: totals[obj_name, inp] for inp in des_vars.keys()}

        n = sum([grad_f_val.size for grad_f_val in grad_f.values()])

        grad_f_vec = np.zeros((n))
        offset = 0
        for grad_f_val in grad_f.values():
            inp_size = grad_f_val.size
            grad_f_vec[offset:offset + inp_size] = grad_f_val
            offset += inp_size

        active_jac_blocks = []

        if not active_cons and not active_dvs:
            return {}, {}

        for (dv_name, active_meta) in active_dvs.items():
            # For active design variable bounds, the constraint gradient
            # wrt des vars is just an identity matrix sized by the number of
            # active elements in the design variable.
            active_idxs = active_meta['indices']

            size = des_vars[dv_name]['size']
            con_grad = {(dv_name, inp): np.eye(size)[active_idxs, ...] if inp == dv_name
                        else np.zeros((size, dv_meta['size']))[active_idxs, ...]
                        for (inp, dv_meta) in des_vars.items()}

            if use_sparse_solve:
                active_jac_blocks.append([sp.sparse.csr_matrix(cg) for cg in con_grad.values()])
            else:
                active_jac_blocks.append(list(con_grad.values()))

        for (con_name, active_meta) in active_cons.items():
            # If the constraint is a design variable, the constraint gradient
            # wrt des vars is just an identity matrix sized by the number of
            # active elements in the design variable.
            active_idxs = active_meta['indices']
            if con_name in des_vars.keys():
                size = des_vars[con_name]['size']
                con_grad = {(con_name, inp): np.eye(size)[active_idxs, ...] if inp == con_name
                            else np.zeros((size, dv_meta['size']))[active_idxs, ...]
                            for (inp, dv_meta) in des_vars.items()}
            else:
                con_grad = {(con_name, inp): totals[con_name, inp][active_idxs, ...]
                            for inp in des_vars.keys()}
            if use_sparse_solve:
                active_jac_blocks.append([sp.sparse.csr_matrix(cg) for cg in con_grad.values()])
            else:
                active_jac_blocks.append(list(con_grad.values()))

        if use_sparse_solve:
            active_cons_mat = sp.sparse.block_array(active_jac_blocks)
        else:
            active_cons_mat = np.block(active_jac_blocks)

        if use_sparse_solve:
            lstsq_sol = sp.sparse.linalg.lsqr(active_cons_mat.T, -grad_f_vec)
        else:
            lstsq_sol = np.linalg.lstsq(active_cons_mat.T, -grad_f_vec, rcond=None)
        multipliers_vec = lstsq_sol[0]

        dv_multipliers = dict()
        con_multipliers = dict()
        offset = 0

        dv_vals = self.get_design_var_values()
        con_vals = self.get_constraint_values()

        for desvar, act_info in active_dvs.items():
            act_idxs = act_info['indices']
            active_size = len(act_idxs)
            mult_vals = multipliers_vec[offset:offset + active_size]
            dv_multipliers[desvar] = np.zeros_like(dv_vals[desvar])
            dv_multipliers[desvar].flat[act_idxs] = mult_vals
            offset += active_size

        for constraint, act_info in active_cons.items():
            act_idxs = act_info['indices']
            active_size = len(act_idxs)
            mult_vals = multipliers_vec[offset:offset + active_size]
            if constraint in des_vars:
                con_multipliers[constraint] = np.zeros_like(dv_vals[constraint])
            else:
                con_multipliers[constraint] = np.zeros_like(con_vals[constraint])
            con_multipliers[constraint].flat[act_idxs] = mult_vals
            offset += active_size

        if not driver_scaling:
            dv_multipliers = self._unscale_lagrange_multipliers(dv_multipliers, assume_dv=True)
            con_multipliers = self._unscale_lagrange_multipliers(con_multipliers, assume_dv=False)

        for key, val in dv_multipliers.items():
            active_dvs[key]['multipliers'] = val

        for key, val in con_multipliers.items():
            active_cons[key]['multipliers'] = val

        return active_dvs, active_cons
