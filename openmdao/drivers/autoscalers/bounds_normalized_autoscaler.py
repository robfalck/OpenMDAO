"""Autoscaler that normalizes design variables to the range [0, 1] based on their bounds."""

import numpy as np

from openmdao.core.constants import INF_BOUND
from openmdao.drivers.autoscalers.autoscaler import Autoscaler
from openmdao.vectors.optimizer_vector import OptimizerVector


class BoundsNormalizedAutoscaler(Autoscaler):
    """Normalize design variables to [0, 1] using their bounds, on top of standard scaling.

    This autoscaler applies the standard affine scaling from the parent :class:`Autoscaler`
    and then adds a second normalization layer that maps every design variable to the
    unit interval:

        x_norm = (x_scaled - lower_scaled) / (upper_scaled - lower_scaled)

    This is useful for derivative-free optimizers (e.g. COBYLA) whose step-size parameter
    is a single scalar applied uniformly to all design variables. When variables span
    different orders of magnitude, normalizing them to [0, 1] ensures the step size is
    proportionally meaningful for every variable.

    Constraints and objectives are **not** normalized; they pass through the standard
    :class:`Autoscaler` transformations unchanged.

    All design variables must have finite lower and upper bounds (after standard scaling).

    Attributes
    ----------
    _dv_norm_lower : np.ndarray or None
        Flat array of lower bounds in driver-scaled space. Set during setup().
    _dv_norm_range : np.ndarray or None
        Flat array of (upper - lower) ranges in driver-scaled space. Set during setup().
    _dv_norm_meta : dict or None
        Metadata dict (slice/size per variable) shared with the cached bounds vectors.
    """

    def __init__(self):
        """Initialize BoundsNormalizedAutoscaler."""
        super().__init__()
        self._dv_norm_lower = None
        self._dv_norm_range = None
        self._dv_norm_meta = None

    def setup(self, driver):
        """
        Set up the autoscaler and compute normalization parameters.

        Parameters
        ----------
        driver : Driver
            The driver associated with this autoscaler.

        Raises
        ------
        RuntimeError
            If any design variable does not have finite lower and upper bounds after
            standard scaling, or if upper <= lower for any design variable.
        """
        super().setup(driver)

        lower_vec, upper_vec, _ = self._scaled_lower['design_var'], \
            self._scaled_upper['design_var'], None

        lower_arr = lower_vec.asarray()
        upper_arr = upper_vec.asarray()

        if np.any(lower_arr <= -INF_BOUND) or np.any(upper_arr >= INF_BOUND):
            raise RuntimeError(
                f'{driver.msginfo}: BoundsNormalizedAutoscaler requires finite lower and '
                'upper bounds on all design variables.'
            )

        dv_range = upper_arr - lower_arr
        if np.any(dv_range <= 0.0):
            raise RuntimeError(
                f'{driver.msginfo}: BoundsNormalizedAutoscaler requires upper > lower for '
                'every design variable.'
            )

        self._dv_norm_lower = lower_arr.copy()
        self._dv_norm_range = dv_range.copy()
        self._dv_norm_meta = lower_vec._meta

        # Replace cached design-var bounds with the normalized [0, 1] versions.
        total_size = len(lower_arr)
        zero_data = np.zeros(total_size)
        one_data = np.ones(total_size)
        meta = self._dv_norm_meta
        self._scaled_lower['design_var'] = OptimizerVector('design_var', zero_data, meta)
        self._scaled_upper['design_var'] = OptimizerVector('design_var', one_data, meta)

    @property
    def has_scaling(self):
        """
        Return True since normalization is always active.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def apply_design_var_unscaling(self, vec):
        """
        Unscale design variables from normalized [0, 1] space to model space.

        Reverses the normalization layer, then delegates to the parent unscaling:
            x_scaled = x_norm * range + lower_scaled

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='design_var' in normalized [0, 1] space.
        """
        if not vec.driver_scaling:
            return vec
        # Reverse normalization: [0, 1] -> driver-scaled space
        vec._data[:] = vec._data * self._dv_norm_range + self._dv_norm_lower
        # Now apply standard unscaling: driver-scaled -> model space
        super().apply_design_var_unscaling(vec)

    def apply_design_var_scaling(self, vec):
        """
        Scale design variables from model space to normalized [0, 1] space.

        Applies standard scaling first, then normalizes to [0, 1]:
            x_norm = (x_scaled - lower_scaled) / range

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='design_var' in model space.
        """
        if vec.driver_scaling:
            return vec
        # First apply standard scaling: model space -> driver-scaled space
        super().apply_design_var_scaling(vec)
        # Then normalize to [0, 1]
        vec._data[:] = (vec._data - self._dv_norm_lower) / self._dv_norm_range

    def apply_jac_scaling(self, jac_dict):
        """
        Scale a Jacobian from model space to normalized optimizer space.

        Applies standard Jacobian scaling, then multiplies each design variable column
        by the corresponding normalization range (chain rule):
            dJ/dx_norm_i = dJ/dx_scaled_i * range_i

        Parameters
        ----------
        jac_dict : dict
            Jacobian blocks, either nested or flat (tuple-keyed) format.
        """
        super().apply_jac_scaling(jac_dict)

        # Build a flat array of per-element range values matching the DV order
        dv_range = self._dv_norm_range  # flat array, ordered by DV concatenation

        for key, jac_block in jac_dict.items():
            if isinstance(key, tuple):
                in_name = key[1]
                if in_name in self._var_meta['design_var']:
                    meta = self._dv_norm_meta[in_name]
                    col_range = dv_range[meta['slice']]
                    jac_block *= col_range
            else:
                for in_name, block in jac_block.items():
                    if in_name in self._var_meta['design_var']:
                        meta = self._dv_norm_meta[in_name]
                        col_range = dv_range[meta['slice']]
                        block *= col_range

    def apply_mult_unscaling(self, desvar_multipliers, con_multipliers):
        """
        Unscale Lagrange multipliers from normalized optimizer space to model space.

        Calls the parent unscaling, then divides each DV multiplier by its normalization
        range (inverse chain rule for the additional normalization layer).

        Parameters
        ----------
        desvar_multipliers : dict[str, np.ndarray]
            Optimizer-scaled Lagrange multipliers for active design variables.
        con_multipliers : dict[str, np.ndarray]
            Optimizer-scaled Lagrange multipliers for active constraints.

        Returns
        -------
        desvar_multipliers : dict[str, np.ndarray]
            Unscaled design variable multipliers (modified in-place).
        con_multipliers : dict[str, np.ndarray]
            Unscaled constraint multipliers (modified in-place).
        """
        super().apply_mult_unscaling(desvar_multipliers, con_multipliers)

        if desvar_multipliers:
            for name, mult in desvar_multipliers.items():
                if name in self._dv_norm_meta:
                    meta = self._dv_norm_meta[name]
                    col_range = self._dv_norm_range[meta['slice']]
                    mult /= col_range

        return desvar_multipliers, con_multipliers
