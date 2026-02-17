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
        x_scaled = (x_model + total_adder) * total_scaler

    Variables `total_scaler` and `total_adder` are computed as a function of the
    scaler/adder/ref/ref0 provided with design variables, objectives, and constraints.

    They also take into account any unit conversions between the model and the optimization
    variables, since users are allowed the specify design vars/constraints/objectives in 
    different units compared to the associated model variables.

    If implementing specialized autoscaler algorithms that derive from this class,
    the developer could choose to utilize or ignore the scaling applied with the optimizer
    variables via scaler/adder/ref0/ref. However, the optimizer variable units will need
    to be accounted for.

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
        meta = self._var_meta[vec.voi_type][name]
        scaler = meta['total_scaler']
        adder = meta['total_adder']

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
            meta = self._var_meta[vec.voi_type][name]
            scaler = meta['total_scaler']
            adder = meta['total_adder']

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
        obj_meta = list(self._var_meta['objective'].values())[0]
        obj_scaler = obj_meta['total_scaler'] or 1.0

        if desvar_multipliers:
            for name, mult in desvar_multipliers.items():
                scaler = self._var_meta['design_var'][name]['total_scaler'] or 1.0
                mult *= scaler / obj_scaler

        if con_multipliers:
            for name, mult in con_multipliers.items():
                scaler = self._var_meta['constraint'][name]['total_scaler'] or 1.0
                mult *= scaler / obj_scaler

        return desvar_multipliers, con_multipliers
