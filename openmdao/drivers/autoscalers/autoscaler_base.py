
class DefaultAutoscaler:

    def __init__(self):
        self._desvars_total_scalers
        self._desvars_total_adders
        self._cons_total_scalers
        self._cons_total_adders
        self._objs_total_scalers
        self._cons_total_adders
        pass

    def setup(self, driver):
        self._driver = driver

    def unscale_desvar(self, name):
        """
        Scale the design variables from the optimizer space to the model space.

        This will be called at every iteration and should therefore be as efficient as possible.
        Several implementations of autoscaling effectively compute it as
        "x_model = M @ x_optimizer".
        In large problems, M will need to be sparse or be a linear operator to save memory.

        Parameters
        ----------
        name: str
            The design variable name or alias to be unscaled.
        """
        pass
    
    def scale_desvar(self, name):
        """
        Scale the design variables from the model space to the optimizer space.

        This will be called to initialize the optimizers design variable vector.

        In the previous example, this would effectively apply "x_optimizer = M^{-1} @ x_model

        Parameters
        ----------
        name: str
            The design variable name or alias to be scaled.
        """
        pass

    def scale_con(self, name):
        """
        Scale the constraints from the model space to the optimizer space.

        This is executed every time the driver needs the current constraint values.

        Parameters
        ----------
        name: str
            The constraint variable name or alias to be scaled.
        """
        pass

    def scale_obj(self):
        """
        Scale the objectives from the model space to the optimizer space.

        This is executed every time the driver needs the current objective values.

        Parameters
        ----------
        name: str
            The objective variable name or alias to be scaled.
        """
        pass
