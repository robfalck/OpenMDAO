from typing import Literal, TYPE_CHECKING, Optional
from enum import Enum
from pydantic import BaseModel, Field, model_validator, field_validator


class NonlinearSolverType(str, Enum):
    """Enumeration of available nonlinear solver types."""

    NEWTON = "NewtonSolver"
    BROYDEN = "BroydenSolver"
    NONLINEAR_BLOCK_GS = "NonlinearBlockGS"
    NONLINEAR_BLOCK_JAC = "NonlinearBlockJac"
    NONLINEAR_RUNONCE = "NonlinearRunOnce"


class LinearSolverType(str, Enum):
    """Enumeration of available linear solver types."""

    DIRECT = "DirectSolver"
    LINEAR_BLOCK_GS = "LinearBlockGS"
    LINEAR_BLOCK_JAC = "LinearBlockJac"
    LINEAR_RUNONCE = "LinearRunOnce"
    SCIPY_KRYLOV = "ScipyKrylov"
    PETSC_KRYLOV = "PETScKrylov"


class LinesearchSolverType(str, Enum):
    """Enumeration of available linesearch solver types."""

    BOUNDS_ENFORCE = "BoundsEnforceLS"
    ARMIJO_GOLDSTEIN = "ArmijoGoldsteinLS"


class SolverOptionsSpec(BaseModel):

    maxiter : int = Field(
        default=10,
        description='maximum number of iterations')

    atol : float = Field(
        default=1.0E-10,
        description='absolute error tolerance')  
      
    rtol : float = Field(
        default=1.0E-10,
        description='relative error tolerance')   
     
    iprint : int = Field(
        default=1,
        description='whether to print output')  

    err_on_non_converge : bool = Field(
        default=False,
        description="When True, AnalysisError will be raised if we do't converge.") 
    

class NonlinearSolverOptionsSpec(SolverOptionsSpec):
    
    debug_print : bool = Field(
        default=False,
        description='If True, the values of input and output variables at '
                    'the start of iteration are printed and written to a file '
                    'after a failure to converge or when encountering an'
                    'invalid value in the residual.')

    stall_limit : int = Field(
        default=0,
        description='Number of iterations after which, if the residual norms are '
                    'identical within the stall_tol, then terminate as if max '
                    'iterations were reached. Default is 0, which disables this feature.')

    stall_tol : float = Field(
        default=1.0E-12,
        description='When stall checking is enabled, the threshold below which the '
                    'residual norm is considered unchanged.')

    stall_tol_type : Literal['abs', 'rel'] = Field(
        default='rel',
        description='Specifies whether the absolute or relative norm of the '
                    'residual is used for stall detection.')

    restart_from_successful : bool = Field(
        default=False,
        description='If True, the states are cached after a successful solve and '
                    'used to restart the solver in the case of a failed solve.')

class NonlinearSolverSpec(BaseModel):
    """
    Specification for a nonlinear solver.

    Nonlinear solvers are used to solve implicit relationships in the model.
    They can optionally include a nested linesearch solver.

    Examples
    --------
    Basic nonlinear solver:
        NonlinearSolverSpec(
            solver_type='NewtonSolver',
            options={'maxiter': 10, 'atol': 1e-10}
        )

    Nonlinear solver with linesearch:
        NonlinearSolverSpec(
            solver_type='NewtonSolver',
            options={'maxiter': 10},
            linesearch=LinesearchSolverSpec(
                solver_type='ArmijoGoldsteinLS',
                options={'bound_enforcement': 'vector'}
            )
        )

    Block Gauss-Seidel solver:
        NonlinearSolverSpec(
            solver_type='NonlinearBlockGS',
            options={'maxiter': 20, 'rtol': 1e-6}
        )
    """

    solver_type: NonlinearSolverType | Literal["NewtonSolver", "BroydenSolver", "NonlinearBlockGS",
                                               "NonlinearBlockJac", "NonlinearRunOnce"] = Field(
        ...,
        description="Type of nonlinear solver"
    )

    options: dict = Field(
        default_factory=dict,
        description="Solver options (e.g., maxiter, atol, rtol, etc.)"
    )

    linesearch: Optional['LinesearchSolverSpec'] = Field(
        default=None,
        description="Optional linesearch solver (typically used with NewtonSolver or BroydenSolver)"
    )


class LinearSolverSpec(BaseModel):
    """
    Specification for a linear solver.

    Linear solvers are used to solve linear systems of equations,
    typically as part of computing derivatives or within a nonlinear solver.

    Examples
    --------
    Direct solver:
        LinearSolverSpec(
            solver_type='DirectSolver',
            options={}
        )

    Iterative Krylov solver:
        LinearSolverSpec(
            solver_type='ScipyKrylov',
            options={'maxiter': 100, 'atol': 1e-12}
        )

    Block Gauss-Seidel solver:
        LinearSolverSpec(
            solver_type='LinearBlockGS',
            options={'maxiter': 10}
        )
    """

    solver_type: LinearSolverType | Literal["DirectSolver", "LinearBlockGS", "LinearBlockJac",
                                            "LinearRunOnce", "ScipyKrylov", "PETScKrylov",
                                            "LinearUserDefined"] = Field(
        ...,
        description="Type of linear solver"
    )

    options: dict = Field(
        default_factory=dict,
        description="Solver options (e.g., maxiter, atol, rtol, etc.)"
    )



class LinesearchSolverOptionsSpec(NonlinearSolverOptionsSpec):

    bound_enforcement : Literal['vector', 'scalar', 'wall'] = Field(
        default='scalar',
        description='If this is set to "vector", the entire vector is backtracked '
        'together when a bound is violated. If this is set to "scalar", only the '
        'violating entries are set to the bound and then the backtracking occurs '
        'on the vector as a whole. If this is set to "wall", only the violating '
        'entries are set to the bound, and then the backtracking follows the wall '
        '- i.e., the violating entries do not change during the line search.')
    
    print_bound_enforce : bool = Field(
        default=False,
        description='Set to True to print out names and values of variables that are '
        'pulled back to their bounds.')
    

class  BoundsEnforceLSOptionsSpec(LinesearchSolverOptionsSpec):

    @field_validator('atol', 'rtol', 'maxiter', 'err_on_non_converge',
                     'restart_from_successful', mode='before')
    @classmethod
    def validate_unused_args(cls, v):
        raise ValueError(f'Option {v} is not used by BoundsEnforceLS.')


class LinesearchSolverSpec(NonlinearSolverSpec):
    """
    Specification for a linesearch solver.

    Linesearch solvers extend NonlinearSolverSpec but accept linesearch solver types
    (BoundsEnforceLS, ArmijoGoldsteinLS) instead of nonlinear solver types.
    """

    solver_type: LinesearchSolverType | Literal["BoundsEnforceLS", "ArmijoGoldsteinLS"] = Field(  # type: ignore[override]
        ...,
        description="Type of linesearch solver"
    )
    

class BoundsEnforceLSSpec(LinesearchSolverSpec):

    options: BoundsEnforceLSOptionsSpec = Field(  # type: ignore[override]
        default_factory=BoundsEnforceLSOptionsSpec,
        description='Options for BoundsEnforceLS.'
    )


# NonlinearSolverSpec requires that LinesearchSolveSpec be defined, so now rebuild it.
NonlinearSolverSpec.model_rebuild()