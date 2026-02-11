from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmdao.core.driver import Driver
    from openmdao.vectors.driver_vector import DriverVector


class DefaultAutoscaler:

    def __init__(self):
        pass

    def setup(self, driver: 'Driver'):
        self._driver = driver

    def unscale_desvars(self, desvars: 'DriverVector'):
        """
        Unscale the design variables from the optimizer space to the model space.

        This will be called at every iteration and should therefore be as efficient as possible.
        Several implementations of autoscaling effectively compute it as
        "x_model = M @ x_optimizer".
        In large problems, M will need to be sparse or be a linear operator to save memory.

        Parameters
        ----------
        desvars : DriverVector
            A vector of design variables in driver-scaled (optimizer) space.
        """
        vector_data = desvars.asarray()
        for name, meta in desvars.get_metadata().items():
            dv_meta = self._driver._designvars[name]
            scaler = dv_meta['total_scaler']
            adder = dv_meta['total_adder']

            start_idx = meta['start_idx']
            end_idx = meta['end_idx']

            # Unscale: x_model = x_optimizer / scaler - adder
            if scaler is not None:
                vector_data[start_idx:end_idx] /= scaler
            if adder is not None:
                vector_data[start_idx:end_idx] -= adder

    def unscale_lagrange_multipliers(self, lambdas: 'DriverVector'):
        """
        Unscale the lagrange multipliers from the optimizer space to the model space.

        Parameters
        ----------
        lambdas : DriverVector
            A vector of Lagrange multipliers.
        """
        pass

    def scale_desvars(self, desvars: 'DriverVector'):
        """
        Scale the design variables from the model space to the optimizer space.

        This will be called to initialize the optimizers design variable vector.

        Parameters
        ----------
        desvars: DriverVector
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

    def scale_cons(self, cons: 'DriverVector'):
        """
        Scale the constraint variables from the model space to the optimizer space.

        Parameters
        ----------
        cons: DriverVector
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

    def scale_objs(self, objs: 'DriverVector'):
        """
        Scale the objective variables from the model space to the optimizer space.

        Parameters
        ----------
        objs: DriverVector
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
