
class DefaultAutoscaler:

    def __init__(self):
        # self._desvars_total_scalers
        # self._desvars_total_adders
        # self._cons_total_scalers
        # self._cons_total_adders
        # self._objs_total_scalers
        # self._cons_total_adders
        pass

    def setup(self, driver):
        pass

    def unscale_desvars(self, driver):
        """
        Scale the design variables from the optimizer space to the model space.

        This will be called at every iteration and should therefore be as efficient as possible.
        Several implementations of autoscaling effectively compute it as
        "x_model = M @ x_optimizer".
        In large problems, M will need to be sparse or be a linear operator to save memory.

        Parameters
        ----------
        driver: Driver
            The driver using this autoscaler.
        """
        pass
    
    def scale_desvars(self, driver):
        """
        Scale the design variables from the model space to the optimizer space.

        This will be called to initialize the optimizers design variable vector.

        In the previous example, this would effectively apply "x_optimizer = M^{-1} @ x_model

        Parameters
        ----------
        driver: Driver
            The driver using this autoscaler.
        """
        pass

    def scale_cons(self, driver):
        """
        Scale the constraints from the model space to the optimizer space.

        This is executed every time the driver needs the current constraint values.

        Parameters
        ----------
        driver: Driver
            The driver using this autoscaler.
        """
        pass

    def scale_objs(self, driver):
        """
        Scale the objectives from the model space to the optimizer space.

        This is executed every time the driver needs the current objective values.

        Parameters
        ----------
        driver: Driver
            The driver using this autoscaler.
        """
        pass
