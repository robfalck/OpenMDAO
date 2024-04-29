class DriverResult():
    """
    A generic container to store the results of a driver run.
    This object is returned by problem.run_driver().

    Attributes
    ----------
    runtime : int
        The time required to run the driver.
    iter_count : int
        The number of driver iterations performed.
    model_evals : int
        The number of times the driver called model.solve_nonlinear.
    deriv_evals : int
        The number of times the driver called its _compute_totals method.
    exit_status : str
        A string providing a the exit status of the driver.
        This string is driver and optimier dependent.
    success : bool
        A boolean flag indicating whether or not the driver run was successful.
        Criteria for driver success is driver and optimizer dependent.
    """
    def __init__(self):
        self.runtime = 0.0
        self.iter_count = 0
        self.model_evals = 0
        self.deriv_evals = 0
        self.exit_status = 'NOT_RUN'
        self.success = False

    def reset(self):
        self.runtime = 0.0
        self.iter_count = 0
        self.model_evals = 0
        self.deriv_evals = 0
        self.exit_status = 'NOT_RUN'
        self.success = False

class OptimizationResult(DriverResult):

    def update_from_scipy(self, scipy_results):
        """
        Load ScipyOptimizer results into the OptimizationResult object.

        Parameters
        ----------
        scipy_results : OptimizeResult
            The scipy OptimizeResult object returned by minimize and other scipy optimizers.
        """
        scipy_map = {'message': 'message',
                     'nfev': 'model_evals',
                     'nit': 'iter_count',
                     'njev': 'deriv_evals',
                     'status': 'exit_status',
                     'success': 'success'}
        for scipy_name, local_name in scipy_map.items():
            try:
                setattr(self, local_name, getattr(scipy_results, scipy_name))
            except AttributeError:
                pass
