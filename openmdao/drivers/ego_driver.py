"""
OpenMDAO Wrapper for the scipy.optimize.minimize family of local optimizers.
"""
import sys
from packaging.version import Version

import numpy as np
from scipy import __version__ as scipy_version

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.general_utils import format_as_float_or_array
from openmdao.utils.mpi import MPI

# from smt.applications import EGO

try:
    from smt.surrogate_models import KPLS, KRG, KPLSK, MGP, GEKPLS
    from smt.utils.design_space import DesignSpace
    _has_smt = True
except ImportError as e:
    _has_smt = False

# Optimizers in scipy.minimize
_optimizers = {'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP'}
if Version(scipy_version) >= Version("1.1"):  # Only available in newer versions
    _optimizers.add('trust-constr')


CITATIONS = """
@article{saves2024smt,
        author = {P. Saves and R. Lafage and N. Bartoli and Y. Diouane and J. Bussemaker and T. Lefebvre and J. T. Hwang and J. Morlier and J. R. R. A. Martins},
        title = {{SMT 2.0: A} Surrogate Modeling Toolbox with a focus on Hierarchical and Mixed Variables Gaussian Processes},
        journal = {Advances in Engineering Sofware},
        year = {2024},
        volume = {188},
        pages = {103571},
        doi = {https://doi.org/10.1016/j.advengsoft.2023.103571}}
"""





class EGODriver(Driver):
    """
    Driver which implements Efficient Global Optimization (EGO).

    All design variables must be bounded.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    _ego : object
        The SMT EGO instance.
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from scipy.optimize call.
    opt_settings : dict
        Dictionary of solver-specific options. See the scipy.optimize.minimize documentation.
    _check_jac : bool
        Used internally to control when to perform singular checks on computed total derivs.
    _con_cache : dict
        Cached result of constraint evaluations because scipy asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : {}
        Cached result of nonlinear constraint derivatives because scipy asks for them in a separate
        function.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _obj_and_nlcons : list
        List of objective + nonlinear constraints. Used to compute total derivatives
        for all except linear constraints.
    _dvlist : list
        Copy of _designvars.
    _lincongrad_cache : np.ndarray
        Pre-calculated gradients of linear constraints.
    _num_obj_eval : int
        The number of times the objective function has been evaluated.
    """

    def __init__(self, **kwargs):
        """
        Initialize the EGODriver.
        """
        if not _has_smt:
            raise RuntimeError('The surrogate modeling toolbox (smt) is a required '
                               'dependency when using the EGO driver, but it was '
                               'not found in this python installation. To use the '
                               'EGODriver, install SMT using:\n'
                               'python -m pip install smt')

        super().__init__(**kwargs)


        self.supports['optimization'] = True

        # What we don't support

        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        self.result = None
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.fail = False
        self.iter_count = 0
        self._ego = None
        self._check_jac = False
        self._exc_info = None
        self._total_jac_format = 'array'
        self._num_obj_eval = 0

        self._training_data ={}

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('max_iter', types=int, default=50,
                             desc='Maximum allowable number of iterations.')
        self.options.declare('gradient_finish', allow_none=True, default=None,
                             desc='Gradient based driver instance used for gradient finish.')
        self.options.declare('EI_finish_tol', types=(float,), default=1.0E-2,
                             desc='Tolerance of expected improvement that terminates the EGO phase.')
        self.options.declare('qEI', values=('KB', 'KBLB', 'KBUB', 'KBRand', 'CLmin'), default='KBLB',
                             desc='	Approximated q-EI maximization strategy')
        self.options.declare('criterion', values=('EI', 'SBO', 'LCB'), default='EI',
                             desc='Criterion for determination of next evaluation point.')
        self.options.declare('maxiter', 100, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('n_start', types=int, default=20, desc='Number of EI optimization start points')
        self.options.declare('random_state', types=int, allow_none=True, default=None)
        self.options.declare('surrogate', types=(KPLS, KRG, KPLSK, MGP, GEKPLS), allow_none=True, default=None,
                             desc='The Kriging surrogate used by the EGO optimizer.')

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        gf_driver = self.options['gradient_finish']
        return 'EGO' if gf_driver is None else f'EGO_{gf_driver._get_name()}'

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super()._setup_driver(problem)

        self.supports._read_only = False
        self.supports['gradients'] = False
        self.supports['inequality_constraints'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports._read_only = True
        self._check_jac = False

    def set_training_data(self, training_data=None):
        """Set the initial training data for the internal Kriging model in the driver.

        If any design variable is not given training data, the Kriging model will
        use LHS sampling to determine points for the sampling data and then run the model
        to determine the training values of the objective.

        If the objective variable is specified in training data then all design
        variables must have corresponding numbers of training data.

        If each variable name is given the same number of training points, AND the
        objective variable is given a corresponding number of training points, the Kriging model will use that training set.

        Parameters
        ----------
        training_data : dict
            The training data to be used to populate the kriging model initially. If the objective variable
            is one of the keys, then all desgin variables must be present and have the same size. If the
            objective variable is not present, then the driver will first execute the model at the specified
            training points to train the kriging model.
        """
        self._training_data = training_data

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        return self._num_obj_eval

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        return None


    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        # opt = self.options['optimizer']
        model = problem.model
        self.iter_count = 0
        self._num_obj_eval = 0
        # self._total_jac = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()
        # TODO: check for unbounded desvars

        # Initial Run
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            self.iter_count += 1

        # self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

        # Size Problem
        ndesvar = 0
        for desvar in self._designvars.values():
            size = desvar['global_size'] if desvar['distributed'] else desvar['size']
            ndesvar += size

        x_init = np.empty(ndesvar)
        x_bounds = np.zeros([ndesvar, 2])
        num_x_train_points_prev = None
        x_train = None
        y_train = None

        # Initial Design Vars
        i = 0

        for name, meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            x_init[i:i + size] = desvar_vals[name]

            x_bounds[i: i+size, 0] = format_as_float_or_array(name, meta['lower'],
                                                              val_if_none=np.nan,
                                                              flatten=True)

            x_bounds[i: i+size, 1] = format_as_float_or_array(name, meta['upper'],
                                                              val_if_none=np.nan,
                                                              flatten=True)

            if self._training_data is not None and name in self._training_data:
                num_x_train_points = self._training_data[name].shape[0]
                if num_x_train_points_prev is None:
                    x_train = np.empty((num_x_train_points, ndesvar))
                elif num_x_train_points != num_x_train_points_prev:
                    raise ValueError('The number of training points for each design variable must be the same.')
                x_train[:, i: i+size] = self._training_data[name]

            i += size

        # Currently use a continuous design space.
        i = 0

        for name, meta in self._objs.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']

            if self._training_data is not None and name in self._training_data:
                num_y_train_points = self._training_data[name].shape[0]
                y_train = np.empty((num_y_train_points, 1))
                # if num_x_train_points_prev is None:
                #     x_train = np.empty((num_x_train_points, ndesvar))
                if num_y_train_points != num_x_train_points:
                    raise ValueError('The number of training points for each design variable must be the same.')
                y_train[:, i: i+size] = self._training_data[name]

        design_space = DesignSpace(x_bounds, seed=self.options['random_state'])

        # optimize
        surrogate = KRG(design_space=design_space, print_global=False)

        try:

            if y_train is not None:
                kwargs = {'ydoe': y_train}
            else:
                kwargs = {}

            self._ego = EGO(n_iter=self.options['max_iter'],
                            criterion=self.options['criterion'],
                            xdoe=x_train,
                            surrogate=surrogate,
                            random_state=self.options['random_state'],
                            **kwargs)

            x_opt, y_opt, argmin, x_data, y_data = self._ego.optimize(fun=self._objfunc)

        # If an exception was swallowed in one of our callbacks, we want to raise it
        # rather than the cryptic message from scipy.
        except Exception as msg:
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        self.result = x_opt, y_opt

        # Currently assume always successful.
        self.fail = False

        # if hasattr(result, 'success'):
        #     self.fail = False if result.success else True
        #     if self.fail:
        #         if self._problem().comm.rank == 0:
        #             print('Optimization FAILED.')
        #             print(result.message)
        #             print('-' * 35)

        #     elif self.options['disp']:
        #         if self._problem().comm.rank == 0:
        #             print('Optimization Complete')
        #             print('-' * 35)
        # else:
        #     self.fail = True  # It is not known, so the worst option is assumed
        #     if self._problem().comm.rank == 0:
        #         print('Optimization Complete (success not known)')
        #         print(result.message)
        #         print('-' * 35)

        return self.fail

    def _objfunc(self, x_new):
        """
        Evaluate and return the objective function.

        Model is executed here.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point(s).

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        model = self._problem().model

        num_eval = x_new.shape[0]
        f_new = np.empty([num_eval, 1])

        try:

            for i in range(num_eval):

                # Pass in new inputs
                j = 0
                if MPI:
                    model.comm.Bcast(x_new, root=0)
                for name, meta in self._designvars.items():
                    size = meta['size']
                    self.set_design_var(name, x_new[i, j:j + size])
                    j += size

                with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                    self.iter_count += 1
                    model.run_solve_nonlinear()

                # Get the objective function evaluations
                for obj in self.get_objective_values().values():
                    # EGO maximizes by default, so flip the sign for consistency
                    # with other OpenMDAO drivers.
                    f_new[i, 0] = obj

                # EGO does not support constraints
                # self._con_cache = self.get_constraint_values()
                self._num_obj_eval += 1

        except Exception as msg:
            if self._exc_info is None:  # only record the first one
                self._exc_info = sys.exc_info()
            return 0

        return f_new

    # def _con_val_func(self, x_new, name, dbl, idx):
    #     """
    #     Return the value of the constraint function requested in args.

    #     The lower or upper bound is **not** subtracted from the value. Used for optimizers,
    #     which take the bounds of the constraints (e.g. trust-constr)

    #     Parameters
    #     ----------
    #     x_new : ndarray
    #         Array containing input values at new design point.
    #     name : str
    #         Name of the constraint to be evaluated.
    #     dbl : bool
    #         True if double sided constraint.
    #     idx : float
    #         Contains index into the constraint array.

    #     Returns
    #     -------
    #     float
    #         Value of the constraint function.
    #     """
    #     return self._con_cache[name][idx]

    # def _confunc(self, x_new, name, dbl, idx):
    #     """
    #     Return the value of the constraint function requested in args.

    #     Note that this function is called for each constraint, so the model is only run when the
    #     objective is evaluated.

    #     Parameters
    #     ----------
    #     x_new : ndarray
    #         Array containing input values at new design point.
    #     name : str
    #         Name of the constraint to be evaluated.
    #     dbl : bool
    #         True if double sided constraint.
    #     idx : float
    #         Contains index into the constraint array.

    #     Returns
    #     -------
    #     float
    #         Value of the constraint function.
    #     """
    #     if self._exc_info is not None:
    #         self._reraise()

    #     cons = self._con_cache
    #     meta = self._cons[name]

    #     # Equality constraints
    #     equals = meta['equals']
    #     if equals is not None:
    #         if isinstance(equals, np.ndarray):
    #             equals = equals[idx]
    #         return cons[name][idx] - equals

    #     # Note, scipy defines constraints to be satisfied when positive,
    #     # which is the opposite of OpenMDAO.
    #     upper = meta['upper']
    #     if isinstance(upper, np.ndarray):
    #         upper = upper[idx]

    #     lower = meta['lower']
    #     if isinstance(lower, np.ndarray):
    #         lower = lower[idx]

    #     if dbl or (lower <= -INF_BOUND):
    #         return upper - cons[name][idx]
    #     else:
    #         return cons[name][idx] - lower

    # def _gradfunc(self, x_new):
    #     """
    #     Evaluate and return the gradient for the objective.

    #     Gradients for the constraints are also calculated and cached here.

    #     Parameters
    #     ----------
    #     x_new : ndarray
    #         Array containing input values at new design point.

    #     Returns
    #     -------
    #     ndarray
    #         Gradient of objective with respect to input array.
    #     """
    #     prob = self._problem()
    #     model = prob.model

    #     try:
    #         grad = self._compute_totals(of=self._obj_and_nlcons, wrt=self._dvlist,
    #                                     return_format=self._total_jac_format)
    #         self._grad_cache = grad

    #         # First time through, check for zero row/col.
    #         if self._check_jac and self._total_jac is not None:
    #             for subsys in model.system_iter(include_self=True, recurse=True, typ=Group):
    #                 if subsys._has_approx:
    #                     break
    #             else:
    #                 raise_error = self.options['singular_jac_behavior'] == 'error'
    #                 self._total_jac.check_total_jac(raise_error=raise_error,
    #                                                 tol=self.options['singular_jac_tol'])
    #             self._check_jac = False

    #     except Exception as msg:
    #         if self._exc_info is None:  # only record the first one
    #             self._exc_info = sys.exc_info()
    #         return np.array([[]])

    #     # print("Gradients calculated for objective")
    #     # print('   xnew', x_new)
    #     # print('   grad', grad[0, :])

    #     return grad[0, :]

    # def _congradfunc(self, x_new, name, dbl, idx):
    #     """
    #     Return the cached gradient of the constraint function.

    #     Note, scipy calls the constraints one at a time, so the gradient is cached when the
    #     objective gradient is called.

    #     Parameters
    #     ----------
    #     x_new : ndarray
    #         Array containing input values at new design point.
    #     name : str
    #         Name of the constraint to be evaluated.
    #     dbl : bool
    #         Denotes if a constraint is double-sided or not.
    #     idx : float
    #         Contains index into the constraint array.

    #     Returns
    #     -------
    #     float
    #         Gradient of the constraint function wrt all inputs.
    #     """
    #     if self._exc_info is not None:
    #         self._reraise()

    #     meta = self._cons[name]

    #     if meta['linear']:
    #         grad = self._lincongrad_cache
    #     else:
    #         grad = self._grad_cache
    #     grad_idx = self._con_idx[name] + idx

    #     # print("Constraint Gradient returned")
    #     # print('   xnew', x_new)
    #     # print('   grad', name, 'idx', idx, grad[grad_idx, :])

    #     # Equality constraints
    #     if meta['equals'] is not None:
    #         return grad[grad_idx, :]

    #     # Note, scipy defines constraints to be satisfied when positive,
    #     # which is the opposite of OpenMDAO.
    #     lower = meta['lower']
    #     if isinstance(lower, np.ndarray):
    #         lower = lower[idx]

    #     if dbl or (lower <= -INF_BOUND):
    #         return -grad[grad_idx, :]
    #     else:
    #         return grad[grad_idx, :]

    def _reraise(self):
        """
        Reraise any exception encountered when scipy calls back into our method.
        """
        exc_info = self._exc_info
        self._exc_info = None  # clear since we're done with it
        raise exc_info[1].with_traceback(exc_info[2])


def signature_extender(fcn, extra_args):
    """
    Closure function, which appends extra arguments to the original function call.

    The first argument is the design vector. The possible extra arguments from the callback
    of :func:`scipy.optimize.minimize` are not passed to the function.

    Some algorithms take a sequence of :class:`~scipy.optimize.NonlinearConstraint` as input
    for the constraints. For this class it is not possible to pass additional arguments.
    With this function the signature will be correct for both scipy and the driver.

    Parameters
    ----------
    fcn : callable
        Function, which takes the design vector as the first argument.
    extra_args : tuple or list
        Extra arguments for the function.

    Returns
    -------
    callable
        The function with the signature expected by the driver.
    """
    def closure(x, *args):
        return fcn(x, *extra_args)

    return closure
