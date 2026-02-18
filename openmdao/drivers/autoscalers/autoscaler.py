from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmdao.core.driver import Driver
    from openmdao.vectors.optimizer_vector import OptimizerVector


class Autoscaler:
    """
    Base class of autoscalers that transform optimizer variables between model and optimizer spaces.

    Autoscalers apply scaling transformations to design variables, constraints, and objectives,
    converting between physical (model) space and optimizer (scaled) space. They also handle
    transformation of Lagrange multipliers from optimizer space back to physical space.

    By default, this autoscaler performs an affine scaling:
        x_scaled = (x_model + combined_adder) * combined_scaler

    The Autoscaler internally combines unit conversion factors (unit_scaler, unit_adder)
    with user-declared scaling factors (total_scaler, total_adder) into cached combined
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

    apply_scaling(vec)
        Scale a vector from model space to optimizer space.
        Modifies vec in-place.

    apply_unscaling(vec, name)
        Return an unscaled copy of a single variable from optimizer space to model space.
        Does not modify vec; returns a new array.

    unscale_lagrange_multipliers(desvar_multipliers, con_multipliers)
        Unscale Lagrange multipliers from optimizer space to physical space.
        Modifies the input dictionaries in-place.

    Notes
    -----
    The OptimizerVector interface provides access to:
    - Full flat array via `.asarray()` - for vectorized operations
    - Variable-by-variable access via `[name]` - for element-wise operations
    - Metadata via `.get_metadata()` - for scaling parameters

    This flexible interface allows autoscalers to choose any implementation strategy,
    from simple loops to complex matrix operations.

    Examples
    --------
    Implementing a custom autoscaler for large-scale problems:

    >>> class VectorizedAutoscaler(Autoscaler):
    ...     def setup(self, driver):
    ...         # Pre-compute full scaler/adder arrays
    ...         self.dv_scalers = np.ones(total_size)
    ...         self.dv_adders = np.zeros(total_size)
    ...         # ... populate with driver metadata ...
    ...
    ...     def apply_scaling(self, vec):
    ...         data = vec.asarray()
    ...         data += self.dv_adders
    ...         data *= self.dv_scalers

    See Also
    --------
    DefaultAutoscaler : Simple element-wise scaling implementation
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

        for voi_type in ['design_var', 'constraint', 'objective']:
            self._combined_scalers[voi_type] = {}
            for name, meta in self._var_meta[voi_type].items():
                scaler, adder = self._compute_combined_scaling(meta)
                self._combined_scalers[voi_type][name] = {
                    'scaler': scaler,
                    'adder': adder
                }

    def pre_run(self, driver: 'Driver'):
        """
        Perform any last minute setup of the autoscaler at the start of the driver's execution.

        The model is fully setup at this point and may be run, allowing the autoscaler to
        set itself up based on the outputs of the model.

        Parameters
        ----------
        driver : Driver
            The driver that is running this autoscaler.
        """
        pass

    def _compute_combined_scaling(self, meta):
        """
        Combine unit conversion and user-declared scaling into single scaler/adder.

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

    def apply_unscaling(self, vec: 'OptimizerVector', name: str):
        """
        Unscale the optmization variables from the optimizer space to the model space.

        This method will generally be applied to each design variable at every iteration.

        Parameters
        ----------
        vec : OptimizationVector
            A vector of the scaled optimization variables.
        name : str
            The name of the optimization variable to be unscaled.

        Returns
        -------
        np.array
            The unscaled value of the variable specified by name.
        """
        # Use cached combined scaler/adder - includes both unit conversion and user scaling
        combined = self._combined_scalers[vec.voi_type][name]
        scaler = combined['scaler']
        adder = combined['adder']

        # Unscale: x_model = x_optimizer / scaler - adder
        # IMPORTANT: copy the vector here.
        out = vec[name].copy()
        if scaler is not None:
            out /= scaler
        if adder is not None:
            out -= adder

        return out
    
    def apply_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the vector from the model space to the optimizer space.

        Scaling is applied to the optimizer vector in-place.
        """
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

    def unscale_lagrange_multipliers(self, desvar_multipliers, con_multipliers):
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
