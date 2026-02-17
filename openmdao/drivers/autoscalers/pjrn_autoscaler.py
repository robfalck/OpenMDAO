from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmdao.core.driver import Driver

from openmdao.core.constants import INF_BOUND
from openmdao.drivers.autoscalers.autoscaler import Autoscaler
from openmdao.utils.om_warnings import issue_warning

import numpy as np



class PJRNAutoscaler(Autoscaler):
    """
    Projected Jacobian Rows Normalization (PJRN) autoscaler.
    
    This implements the PJRN scaling technique from:
    Sagliano, M. (2014). "Performance analysis of linear and nonlinear techniques 
    for automatic scaling of discretized control problems." Operations Research Letters, 
    42(3), 213-216.
    
    The PJRN method scales constraints by normalizing the rows of the Jacobian matrix 
    after projection by the inverse of the design variable scaling matrix. This accounts 
    for both the relationship between states and constraints (via the Jacobian) and the 
    normalization of the design variables themselves.
    
    For constraints F(x) and G(x), the scaling factors are:
        K_F[i,i] = 1 / ||(∇F · K_x^{-1})[i,:]||
        K_G[i,i] = 1 / ||(∇G · K_x^{-1})[i,:]||
    
    where K_x is the diagonal design variable scaling matrix.
    
    This class only overrides the setup() method to compute PJRN scaling factors.
    All scaling/unscaling operations are inherited from DefaultAutoscaler.

    Parameters
    ----------
    allow_unbounded_desvars : bool
        By default, the PJRNAutoscaler requires each design variable have bounds and
        will error if any design variables are specified without bounds. Setting this
        argument to True will result in the autoscaler assuming that the bounds of 
        each design varaible are equal to the are equal to the max(abs(val)).
    igore_existing_scaling : bool
        If True, compute scaling parameters ignore existing scaling via scaler/adder/ref0/ref
        and units.
    """

    def __init__(self, allow_unbounded_desvars=False, ignore_existing_scaling=False):
        super().__init__()

        self._allow_unbounded_desvars = allow_unbounded_desvars
        self._ignore_existing_scaling = ignore_existing_scaling

    def pre_run(self, driver: 'Driver'):
        """
        Set up the PJRN autoscaler.
        
        This method computes the scaling factors based on the Jacobian of the problem
        evaluated at the current point. It must be called after the driver is set up
        and the model has been run at least once.
        
        Parameters
        ----------
        driver : Driver
            The OpenMDAO driver containing the optimization problem.
        """
        # Compute the design variable scaling matrix (K_x)
        self._compute_desvar_scaling()
        
        # Compute total derivatives (Jacobian)
        # This gets derivatives of objectives and constraints w.r.t. design variables
        totals = driver._compute_totals(
            return_format='dict',
            driver_scaling=False  # We want unscaled derivatives
        )
        
        # Compute PJRN scaling for objectives
        self._compute_objective_scaling(totals)
        
        # Compute PJRN scaling for constraints
        self._compute_constraint_scaling(totals)

    def _compute_desvar_scaling(self):
        """
        Compute the design variable scaling matrix K_x based on variable bounds.
        
        Following the standard approach (eq. 3-4 in Sagliano), design variables 
        are scaled to [0, 1] using:
            x_scaled = (x - x_L) / (x_U - x_L)
        
        This gives:
            K_x[i,i] = 1 / (x_U[i] - x_L[i])
        
        We store K_x^{-1} = (x_U - x_L) for use in the projection step.
        """
        self._desvar_scaler_inv = {}
        unbounded_desvars = []
        
        for name, meta in self._var_meta['design_var'].items():
            lower = meta.get('lower', -INF_BOUND)
            upper = meta.get('upper', INF_BOUND)
            
            if np.all(lower > -INF_BOUND) and np.all(upper < INF_BOUND):
                # Compute K_x^{-1} (inverse of scaler)
                # This is the range: upper - lower
                self._desvar_scaler_inv[name] = upper - lower
            else:
                # For unbounded variables, use identity scaling
                # In practice, artificial bounds should be used
                unbounded_desvars.append(name)
                size = meta.get('size', 1)
                self._desvar_scaler_inv[name] = np.ones(size)
        
        if unbounded_desvars:
            msg = ('The following design variables have no bounds but they '
                   'are required by PJRNAutoscaler:') + '\n- ' + '\n- '.join(unbounded_desvars)
            import textwrap
            msg = textwrap.indent(msg, '  ', lambda s: s.strip().startswith('-'))
            
            if self._allow_unbounded_desvars:
                msg += ('\nProceeding with using 1.0 as the scaler for '
                        'these design variables (allow_unbounded_desvars=True)')
                issue_warning(msg)
            else:
                raise ValueError(msg)

    def _compute_objective_scaling(self, totals):
        """
        Compute PJRN scaling for objectives.
        
        For each objective, compute the norm of the gradient projected by K_x^{-1}:
            K_J = 1 / ||∇J · K_x^{-1}||
        
        Parameters
        ----------
        totals : dict
            Dictionary of total derivatives from compute_totals.
        """
        for obj_name in self._var_meta['objective'].keys():
            # Compute projected gradient: ∇J · K_x^{-1}
            projected_grad = []
            
            for dv_name in self._var_meta['design_var'].keys():
                key = (obj_name, dv_name)
                if key in totals:
                    deriv = totals[key]
                    # Project by K_x^{-1}
                    projected = deriv * self._desvar_scaler_inv[dv_name]
                    projected_grad.append(projected.flatten())
            
            if projected_grad:
                # Concatenate all partial derivatives and compute norm
                projected_grad = np.concatenate(projected_grad)
                grad_norm = np.linalg.norm(projected_grad)
                
                if grad_norm > 1e-15:  # Avoid division by zero
                    scaler = 1.0 / grad_norm
                else:
                    scaler = 1.0
                
                # Store the computed scaler in the metadata
                self._var_meta['objective'][obj_name]['total_scaler'] = scaler
                self._var_meta['objective'][obj_name]['total_adder'] = 0.0

    def _compute_constraint_scaling(self, totals):
        """
        Compute PJRN scaling for constraints.
        
        For each constraint row i, compute:
            K_G[i,i] = 1 / ||(∇G_i · K_x^{-1})||
        
        Parameters
        ----------
        totals : dict
            Dictionary of total derivatives from compute_totals.
        """
        for con_name, con_meta in self._var_meta['constraint'].items():
            con_size = con_meta.get('size', 1)
            
            # Initialize scaler array for this constraint
            scalers = np.ones(con_size)
            
            # For each row of the constraint
            for i in range(con_size):
                # Compute projected Jacobian row: (∇G_i) · K_x^{-1}
                projected_row = []
                
                for dv_name in self._var_meta['design_var'].keys():
                    key = (con_name, dv_name)
                    if key in totals:
                        deriv = totals[key]
                        
                        # Handle both 1D and 2D arrays
                        if deriv.ndim == 1:
                            deriv_row = deriv[i] if con_size > 1 else deriv
                        else:
                            deriv_row = deriv[i, :] if con_size > 1 else deriv.flatten()
                        
                        # Project by K_x^{-1}
                        projected = deriv_row * self._desvar_scaler_inv[dv_name]
                        projected_row.append(projected.flatten())
                
                if projected_row:
                    # Concatenate and compute row norm
                    projected_row = np.concatenate(projected_row)
                    row_norm = np.linalg.norm(projected_row)
                    
                    if row_norm > 1e-15:  # Avoid division by zero
                        scalers[i] = 1.0 / row_norm
                    else:
                        scalers[i] = 1.0
            
            # Store the computed scalers in the metadata
            if con_size == 1:
                self._var_meta['constraint'][con_name]['total_scaler'] = scalers[0]
            else:
                self._var_meta['constraint'][con_name]['total_scaler'] = scalers
            
            self._var_meta['constraint'][con_name]['total_adder'] = 0.0