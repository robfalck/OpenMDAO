from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openmdao.core.driver import Driver
    from openmdao.vectors.optimizer_vector import OptimizerVector


class Autoscaler:
    """
    Base class of autoscalers that transform optimizer variables between model and optimizer spaces.

    Autoscalers apply scaling transformations to **continuous** design variables, constraints,
    and objectives, converting between physical (model) space and optimizer (scaled) space
    They also handle transformation of Lagrange multipliers from optimizer space back to physical
    (model) space and of jacobians from model space to optimizer space.

    Discrete design variables, constraints, and objectives are not scaled/unscaled by autoscalers.

    By default, this autoscaler performs an affine scaling:
        x_scaled = (x_model + combined_adder) * combined_scaler

    The Autoscaler internally combines unit conversion factors (unit_scaler, unit_adder)
    with user-declared scaling factors (scaler, adder) into cached combined
    values. This recreates the pre-separation combined scaling behavior, where both unit
    conversion and user scaling are applied together in a single affine transformation.

    Combined scaling factors are computed as:
        combined_scaler = unit_scaler * total_scaler
        combined_adder = unit_adder + total_adder / unit_scaler

    These combined values are cached during setup for efficiency and used in all subsequent
    scaling operations. This allows autoscalers to handle both unit conversions and user-
    specified scaling in a self-contained manner.

    If implementing specialized autoscaler algorithms that derive from this class,
    the developer could choose to utilize or ignore the scaling applied with the optimizer
    variables via scaler/adder/ref0/ref. However, the unit conversions (which are part of
    the combined scaling) will need to be accounted for to maintain physical correctness.

    Subclasses must implement the following methods:

    Methods
    -------
    setup(driver)
        Initialize the autoscaler with driver metadata.
        Called once during driver setup.
    
    update(driver)
        Update any scaling parameters after the model has been executed.
        This can be used to assess the current values in the model,
        the current jacobian, etc.

    apply_vec_scaling(vec)
        Scale a vector from model space to optimizer space.
        Modifies vec in-place.

    apply_vec_unscaling(vec, name)
        Return an unscaled copy of a single variable from optimizer space to model space.
        Does not modify vec; returns a new array.

    apply_mult_unscaling(desvar_multipliers, con_multipliers)
        Unscale Lagrange multipliers from optimizer space to physical space.
        Modifies the input dictionaries in-place.
    
    apply_jac_scaling
        TBD

    This flexible interface allows autoscalers to choose any implementation strategy,
    from simple loops to complex matrix operations.

    """

    def setup(self, driver: 'Driver'):
        """
        Perform setup of autoscaler during final setup of the problem.

        Parameters
        ----------
        driver : Driver
            The driver associated with this autoscaler.
        """
        self._var_meta : dict[str, dict[str, dict]] = {
            'design_var': driver._designvars,
            'constraint': driver._cons,
            'objective': driver._objs
        }

        # Compute and cache combined scalers for all variables
        self._combined_scalers = {}

        self._has_scaling = False

        for voi_type in ['design_var', 'constraint', 'objective']:
            self._combined_scalers[voi_type] = {}
            for name, meta in self._var_meta[voi_type].items():
                total_scaler, total_adder = self._compute_combined_scaling(meta)
                self._has_scaling = self._has_scaling \
                    or (total_scaler is not None) \
                    or (total_adder is not None)
                self._combined_scalers[voi_type][name] = {
                    'scaler': total_scaler,
                    'adder': total_adder
                }
    
    @property
    def has_scaling(self):
        return self._has_scaling

    def update(self, driver: 'Driver'):
        """
        Perform any last minute setup of the autoscaler at the start of the driver's execution.

        This method is called during driver.run when the model has been executed. It can be used
        to configure the scaling based on values in the model, the current model jacobian, etc.

        Parameters
        ----------
        driver : Driver
            The driver that is running this autoscaler.
        """
        pass

    def _compute_combined_scaling(self, meta):
        """
        Combine unit conversion and user-declared scaling into single scaler/adder.

        This routine will be useful for those scaling algorithms which effectively
        decompose scaling into a single scalar and single adder per each variable,
        such as OpenMDAO's default user-specified scaling.

        Combines the two-step transformation:
            y_scaled = ((y + unit_adder) * unit_scaler + total_adder) * total_scaler

        Into single affine transformation:
            y_scaled = (y + combined_adder) * combined_scaler

        Parameters
        ----------
        meta : dict
            Metadata dictionary containing scaling parameters.

        Returns
        -------
        tuple
            (combined_scaler, combined_adder) where each can be None, float, or array
        """
        unit_scaler = meta.get('unit_scaler')
        unit_adder = meta.get('unit_adder')
        total_scaler = meta.get('total_scaler')
        total_adder = meta.get('total_adder')

        # If all None, no scaling needed
        if all(x is None for x in [unit_scaler, unit_adder, total_scaler, total_adder]):
            return None, None

        # Compute combined_scaler = unit_scaler * total_scaler
        if unit_scaler is None and total_scaler is None:
            combined_scaler = None
        elif unit_scaler is None:
            combined_scaler = total_scaler
        elif total_scaler is None:
            combined_scaler = unit_scaler
        else:
            combined_scaler = unit_scaler * total_scaler

        # Compute combined_adder = unit_adder + total_adder / unit_scaler
        if unit_adder is None and total_adder is None:
            combined_adder = None
        elif unit_adder is None:
            # combined_adder = total_adder / unit_scaler
            if unit_scaler is None:
                combined_adder = total_adder
            else:
                combined_adder = total_adder / unit_scaler
        elif total_adder is None:
            combined_adder = unit_adder
        else:
            # combined_adder = unit_adder + total_adder / unit_scaler
            if unit_scaler is None:
                combined_adder = unit_adder + total_adder
            else:
                combined_adder = unit_adder + total_adder / unit_scaler

        return combined_scaler, combined_adder

    def apply_vec_unscaling(self, vec: 'OptimizerVector'):
        """
        Unscale the optmization variables from the optimizer space to the model space, in place.

        This method will generally be applied to each design variable at every iteration.

        Parameters
        ----------
        vec : OptimizerVector
            A vector of the scaled optimization variables.

        Returns
        -------
        OptimizerVector
            The unscaled optimization vector.
        """
        if not vec.scaled:
            return vec
        
        for name in vec:
            # Use cached combined scaler/adder - includes both unit conversion and user scaling
            combined = self._combined_scalers[vec.voi_type][name]
            scaler = combined['scaler']
            adder = combined['adder']

            # Unscale: x_model = x_optimizer / scaler - adder
            if scaler is not None:
                vec[name] /= scaler
            if adder is not None:
                vec[name] -= adder
        vec.scaled = False

        return vec
    
    def apply_vec_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the vector from the model space to the optimizer space.

        Scaling is applied to the optimizer vector in-place.
        """
        if vec.scaled:
            return vec
        for name in vec:
            # Use cached combined scaler/adder - includes both unit conversion and user scaling
            combined = self._combined_scalers[vec.voi_type][name]
            scaler = combined['scaler']
            adder = combined['adder']

            # Scale: x_optimizer = (x_model + adder) * scaler
            if adder is not None:
                vec[name] += adder
            if scaler is not None:
                vec[name] *= scaler
        vec.scaled = True

    def apply_mult_unscaling(self, desvar_multipliers, con_multipliers):
        """
        Unscale the Lagrange multipliers from optimizer space to model space.
        
        This method transforms Lagrange multipliers of active constraints (including
        active design variable bounds) from the scaled optimization space back to 
        physical (model) space.
        
        At optimality, we assume the KKT stationarity condition holds:
        
            ∇ₓf(x) + ∇ₓg(x)^T λ = 0
        
        where:
            - ∇ₓf is the gradient of the objective
            - ∇ₓg(x)^T is the Jacobian of all active constraints (each row is ∇ₓg_i^T)
            - λ is the vector of Lagrange multipliers (in optimizer-scaled)
        
        The constraint vector g(x) includes:
            - Active design variables (on their bounds, to within some tolerance)
            - Equality constraints (always active)
            - Active inequality constraints (on their bounds, to within some tolerance)
        
        Scaling Transformations
        -----------------------
        Define scaling transformations that map from unscaled (physical) space to
        scaled (optimizer) space:
        
            x_scaled = T_x(x)         (design variables)
            g_scaled = T_g(g(x))      (constraints)
            f_scaled = T_f(f(x))      (objective)
        
        Applying the chain rule to the scaled stationarity condition:
        
            ∇ₓ_scaled f_scaled + ∇ₓ_scaled g_scaled^T λ_scaled = 0
        
        The gradients in scaled space are:
        
            ∇ₓ_scaled f_scaled = (dTf/df) * ∇ₓf * (dTₓ/dx)^(-1)
            ∇ₓ_scaled g_scaled = (dTg/dg) * ∇ₓg * (dTₓ/dx)^(-1)
        
        Substituting into the scaled stationarity condition and multiplying by (dTₓ/dx)^T:
        
            (dTf/df) * ∇ₓf + (dTg/dg) * ∇ₓg^T * λ_scaled = 0
        
        Dividing by (dTf/df) and comparing with the unscaled condition ∇ₓf + ∇ₓg^T λ = 0:
        
            λ = (dTg/dg) / (dTf/df) * λ_scaled
        
        For the Default autoscaler, we have
        
            T_x(x) = (x + adder_x) * scaler_x
            T_g(g) = (g + adder_g) * scaler_g
            T_f(f) = (f + adder_f) * scaler_f
        
        The derivatives are:
        
            dT_x/dx = scaler_x
            dT_g/dg = scaler_g
            dT_f/df = scaler_f
        
        Therefore:
        
            λ_constraint = (scaler_g / scaler_f) * λ_constraint_scaled
    
            λ_bound = (scaler_x / scaler_f) * λ_bound_scaled
        
        The adder terms do not appear in the multiplier transformation
        because they are constant offsets that vanish under differentiation.
        
        Parameters
        ----------
        desvar_multipliers : dict[str, np.ndarray]
            A dict of optimizer-scaled Lagrange multipliers keyed by each active design variable.
        con_multipliers : dict[str, np.ndarray]
            A dict of optimizer-scaled Lagrange multipliers keyed by each active constraint.
        
        Returns
        -------
        desvar_multipliers : dict[str, np.ndarray]
            A reference to the desvar_multipliers given on input. The values of the multipliers
            were unscaled in-place.
        con_multipliers : dict[str, np.ndarray]
            A reference to the con_multipliers given on input. The values of the multipliers
            were unscaled in-place.
        """
        if not self._has_scaling:
            return desvar_multipliers, con_multipliers

        # Get the objective scaler from cached combined scalers
        obj_name = list(self._var_meta['objective'].keys())[0]
        obj_scaler = self._combined_scalers['objective'][obj_name]['scaler'] or 1.0

        if desvar_multipliers:
            for name, mult in desvar_multipliers.items():
                # Get the design variable scaler from cached combined scalers
                scaler = self._combined_scalers['design_var'][name]['scaler'] or 1.0
                mult *= scaler / obj_scaler

        if con_multipliers:
            for name, mult in con_multipliers.items():
                # Get the constraint scaler from cached combined scalers
                scaler = self._combined_scalers['constraint'][name]['scaler'] or 1.0
                mult *= scaler / obj_scaler

        return desvar_multipliers, con_multipliers

    def apply_jac_scaling(self, jac_dict):
        """
        Scale a Jacobian dictionary from model space to optimizer space.

        Applies the scaling transformation to convert a Jacobian computed in the model's
        coordinate system to the optimizer's scaled coordinate system.

        The scaling transformation for the Jacobian is:
            J_scaled = (dT_f/df) * J_model * (dT_x/dx)^-1
                     = scaler_f * J_model / scaler_x

        This accounts for how the scaling transformations affect the derivatives.

        Parameters
        ----------
        jac_dict : dict
            Dictionary of Jacobian blocks. Can be either:
            - Nested dict where jac_dict[output_name][input_name] = array
            - Flat dict where jac_dict[(output_name, input_name)] = array

        Notes
        -----
        The method modifies the Jacobian dictionary in-place, scaling each partial
        derivative block according to the output and input scalers.

        When a scaler is None (identity transformation), it's treated as 1.0 for
        multiplication and division.
        """
        if not self._has_scaling:
            return

        for key, jac_block in jac_dict.items():
            # Handle both nested dict and flat dict formats
            if isinstance(key, tuple):
                # Flat dict format: key is (output_name, input_name)
                out_name, in_name = key
            else:
                # Nested dict format: key is output_name, need to iterate inner dicts
                out_name = key
                for in_name, block in jac_block.items():
                    # Determine output scaler
                    if out_name in self._var_meta['objective']:
                        out_scaler = self._combined_scalers['objective'][out_name]['scaler']
                    elif out_name in self._var_meta['constraint']:
                        out_scaler = self._combined_scalers['constraint'][out_name]['scaler']
                    else:
                        # Unknown output, skip scaling this row
                        continue

                    # Determine input scaler
                    if in_name in self._var_meta['design_var']:
                        in_scaler = self._combined_scalers['design_var'][in_name]['scaler']
                    else:
                        # Unknown input, skip scaling this entry
                        continue

                    # Scale the Jacobian block in-place: J_scaled = J_model * out_scaler / in_scaler
                    # Use in-place operations to preserve view relationship with underlying array
                    if out_scaler is not None:
                        block[...] = (out_scaler * block.T).T
                    if in_scaler is not None:
                        block *= 1.0 / in_scaler
                continue

            # Handle flat dict format (key is a tuple)
            # Determine output scaler
            if out_name in self._var_meta['objective']:
                out_scaler = self._combined_scalers['objective'][out_name]['scaler']
            elif out_name in self._var_meta['constraint']:
                out_scaler = self._combined_scalers['constraint'][out_name]['scaler']
            else:
                # Unknown output, skip scaling this entry
                continue

            # Determine input scaler
            if in_name in self._var_meta['design_var']:
                in_scaler = self._combined_scalers['design_var'][in_name]['scaler']
            else:
                # Unknown input, skip scaling this entry
                continue

            # Scale the Jacobian block in-place: J_scaled = J_model * out_scaler / in_scaler
            # Must use in-place operations to preserve view relationship with underlying array
            if out_scaler is not None:
                jac_block[...] = (out_scaler * jac_block.T).T
            if in_scaler is not None:
                jac_block *= 1.0 / in_scaler
    
    def apply_discrete_unscaling(self, name, voi_type, val):
        """
        Unscale the discrete optimization variable given by name.

        Parameters
        ----------
        name : str
            The name of the variable being unscaled.
        voi_type : str ('design_var', 'constraint', 'objective')
            The kind of variable to be unscaled.
        val : float or ndarray
            The scaled value of the varaible.

        Returns
        -------
        float or ndarray
            An unscaled copy of the variable if scaling was defined, otherwise
            val is returned unchanged.
        """ 
        combined = self._combined_scalers[voi_type][name]
        scaler = combined['scaler']
        adder = combined['adder']

        # Unscale: x_model = x_optimizer / scaler - adder
        if scaler or adder:
            out = np.copy(val)
            if scaler is not None:
                out /= scaler
            if adder is not None:
                out -= adder
            return out

        return val
    
    def apply_discrete_scaling(self, name, voi_type, val):
        """
        Scale the discrete optimization variable from the model space to the optimizer space.

        Parameters
        ----------
        name : str
            The name of the variable being unscaled.
        voi_type : str ('design_var', 'constraint', 'objective')
            The kind of variable to be unscaled.
        val : float or ndarray
            The scaled value of the varaible.

        Returns
        -------
        float or ndarray
            An scaled copy of the variable if scaling was defined, otherwise
            val is returned unchanged.
        """
        # Use cached combined scaler/adder - includes both unit conversion and user scaling
        combined = self._combined_scalers[voi_type][name]
        scaler = combined['scaler']
        adder = combined['adder']

        # Scale: x_optimizer = (x_model + adder) * scaler
        if scaler or adder:
            out = np.copy(val)
            if adder is not None:
                out += adder
            if scaler is not None:
                out *= scaler
            return out

        return val