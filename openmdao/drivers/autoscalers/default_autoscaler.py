from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmdao.core.driver import Driver
    from openmdao.vectors.optimizer_vector import OptimizerVector


class AutoscalerBase:

    def setup(self, driver: 'Driver'):
        self._var_meta : dict[str, dict[str, dict]] = {
            'design_var': driver._designvars,
            'constraint': driver._cons,
            'objective': driver._objs
        }


class DefaultAutoscaler(AutoscalerBase):

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

    def unscale_lagrange_multipliers(self, lambdas: 'OptimizerVector'):
        """
        Unscale the lagrange multipliers from the optimizer space to the model space.

        Parameters
        ----------
        lambdas : OptimizerVector
            A vector of Lagrange multipliers.
        """
        pass

    def scale_desvars(self, desvars: 'OptimizerVector'):
        """
        Scale the design variables from the model space to the optimizer space.

        This will be called to initialize the optimizers design variable vector.

        Parameters
        ----------
        desvars: OptimizerVector
            A vector of the design variables in model (unscaled) space.
        """
        vector_data = desvars.asarray()
        for name, meta in desvars.get_metadata().items():
            dv_meta = self._driver._designvars[name]
            scaler = dv_meta['total_scaler']
            adder = dv_meta['total_adder']

            start_idx = meta['start_idx']
            end_idx = meta['end_idx']

            # Scale: x_optimizer = (x_model + adder) * scaler
            if adder is not None:
                vector_data[start_idx:end_idx] += adder
            if scaler is not None:
                vector_data[start_idx:end_idx] *= scaler

    def scale_cons(self, cons: 'OptimizerVector'):
        """
        Scale the constraint variables from the model space to the optimizer space.

        Parameters
        ----------
        cons: OptimizerVector
            A vector of the constraint variables in model (unscaled) space.
        """
        vector_data = cons.asarray()
        for name, meta in cons.get_metadata().items():
            con_meta = self._driver._cons[name]
            scaler = con_meta['total_scaler']
            adder = con_meta['total_adder']

            start_idx = meta['start_idx']
            end_idx = meta['end_idx']

            # Scale: c_optimizer = (c_model + adder) * scaler
            if adder is not None:
                vector_data[start_idx:end_idx] += adder
            if scaler is not None:
                vector_data[start_idx:end_idx] *= scaler

    def scale_objs(self, objs: 'OptimizerVector'):
        """
        Scale the objective variables from the model space to the optimizer space.

        Parameters
        ----------
        objs: OptimizerVector
            A vector of the objective variables in model (unscaled) space.
        """
        vector_data = objs.asarray()
        for name, meta in objs.get_metadata().items():
            obj_meta = self._driver._objs[name]
            scaler = obj_meta['total_scaler']
            adder = obj_meta['total_adder']

            start_idx = meta['start_idx']
            end_idx = meta['end_idx']

            # Scale: f_optimizer = (f_model + adder) * scaler
            if adder is not None:
                vector_data[start_idx:end_idx] += adder
            if scaler is not None:
                vector_data[start_idx:end_idx] *= scaler
