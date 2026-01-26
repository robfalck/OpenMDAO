from typing import Literal, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, field_serializer

from openmdao.specs.solver_registry import register_solver_spec, _SOLVER_SPEC_REGISTRY
from openmdao.specs.solver_options_base import SolverOptionsSpec


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


class NewtonSolverOptionsSpec(NonlinearSolverOptionsSpec):
    """Options specific to NewtonSolver."""

    solve_subsystems: bool = Field(
        ...,
        description='Set to True to enable Hybrid Newton which solves subsystems. '
                    'This is required to be set by the user.'
    )

    max_sub_solves: int = Field(
        default=10,
        description='Maximum number of subsystem solves.'
    )

    cs_reconverge: bool = Field(
        default=True,
        description='When True, when this solver solves under a complex step, nudge '
                    'the solution vector by a small amount so that it reconverges.'
    )

    reraise_child_analysiserror: bool = Field(
        default=False,
        description='When the option is true, a solver will reraise any AnalysisError '
                    'that arises during subsolve.'
    )


class NonlinearBlockGSOptionsSpec(NonlinearSolverOptionsSpec):
    """Options specific to NonlinearBlockGS."""

    use_aitken: bool = Field(
        default=False,
        description='Set to True to use Aitken relaxation.'
    )

    aitken_min_factor: float = Field(
        default=0.1,
        description='Lower limit for Aitken relaxation factor.'
    )

    aitken_max_factor: float = Field(
        default=1.5,
        description='Upper limit for Aitken relaxation factor.'
    )

    aitken_initial_factor: float = Field(
        default=1.0,
        description='Initial value for Aitken relaxation factor.'
    )

    cs_reconverge: bool = Field(
        default=True,
        description='When True, when this solver solves under a complex step, nudge '
                    'the solution vector by a small amount so that it reconverges.'
    )

    use_apply_nonlinear: bool = Field(
        default=False,
        description='Set to True to always call apply_nonlinear on the solver\'s system '
                    'after solve_nonlinear has been called.'
    )

    reraise_child_analysiserror: bool = Field(
        default=False,
        description='When the option is true, a solver will reraise any AnalysisError '
                    'that arises during subsolve.'
    )


class NonlinearBlockJacOptionsSpec(NonlinearSolverOptionsSpec):
    """Options specific to NonlinearBlockJac (only inherits base options)."""

    pass


class BroydenSolverOptionsSpec(NonlinearSolverOptionsSpec):
    """Options specific to BroydenSolver."""

    alpha: float = Field(
        default=0.4,
        description='Initial scale factor for the Jacobian.'
    )

    compute_jacobian: bool = Field(
        default=True,
        description='If True, compute an initial Jacobian; otherwise use identity matrix.'
    )

    converge_limit: float = Field(
        default=1.0,
        description='Threshold below which convergence is considered failed.'
    )

    cs_reconverge: bool = Field(
        default=True,
        description='When True, when this solver solves under a complex step, nudge '
                    'the solution vector by a small amount so that it reconverges.'
    )

    diverge_limit: float = Field(
        default=2.0,
        description='Limit for divergence; solver will attempt recovery if exceeded.'
    )

    max_converge_failures: int = Field(
        default=3,
        description='Maximum number of convergence failures before regenerating Jacobian.'
    )

    max_jacobians: int = Field(
        default=10,
        description='Maximum number of Jacobians to compute.'
    )

    state_vars: list = Field(
        default_factory=list,
        description='List of state variable names to solve for.'
    )

    update_broyden: bool = Field(
        default=True,
        description='If True, perform Broyden update on Jacobian.'
    )

    reraise_child_analysiserror: bool = Field(
        default=False,
        description='When the option is true, a solver will reraise any AnalysisError '
                    'that arises during subsolve.'
    )


class NonlinearRunOnceOptionsSpec(NonlinearSolverOptionsSpec):
    """Options specific to NonlinearRunOnce (only inherits base options)."""
    pass


class NonlinearSolverBaseSpec(BaseModel):
    """
    Base class for all nonlinear solver specifications.

    This class provides a common type for type annotations and registration,
    without defining any fields.
    """

    pass


class NonlinearSolverSpec(NonlinearSolverBaseSpec):
    """
    Specification for a nonlinear solver.

    Nonlinear solvers are used to solve implicit relationships in the model.
    They can optionally include a nested linesearch solver.

    Examples
    --------
    Basic nonlinear solver:
        NonlinearSolverSpec(
            options={'maxiter': 10, 'atol': 1e-10}
        )

    Nonlinear solver with linesearch:
        NonlinearSolverSpec(
            options={'maxiter': 10},
            linesearch=LinesearchSolverSpec(
                solver_type='ArmijoGoldsteinLS',
                options={'bound_enforcement': 'vector'}
            )
        )

    Block Gauss-Seidel solver:
        NonlinearSolverSpec(
            options={'maxiter': 20, 'rtol': 1e-6}
        )
    """

    options: dict = Field(
        default_factory=dict,
        description="Solver options (e.g., maxiter, atol, rtol, etc.)"
    )

    linesearch: Optional['LinesearchSolverSpec'] = Field(
        default=None,
        description="Optional linesearch solver (typically used with NewtonSolver or BroydenSolver)"
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


class BoundsEnforceLSOptionsSpec(LinesearchSolverOptionsSpec):

    @field_validator('atol', 'rtol', 'maxiter', 'err_on_non_converge',
                     'restart_from_successful', mode='before')
    @classmethod
    def validate_unused_args(cls, v):
        raise ValueError(f'Option {v} is not used by BoundsEnforceLS.')


class LinesearchSolverBaseSpec(BaseModel):
    """
    Base class for all linesearch solver specifications.

    This class provides a common type for type annotations and registration,
    without defining any fields.
    """

    pass


class LinesearchSolverSpec(LinesearchSolverBaseSpec):
    """
    Specification for a linesearch solver.

    Linesearch solvers extend NonlinearSolverSpec but accept linesearch solver types
    (BoundsEnforceLS, ArmijoGoldsteinLS) instead of nonlinear solver types.

    This is a generic linesearch solver spec. For specific linesearch solvers with
    typed options, use BoundsEnforceLSSpec or ArmijoGoldsteinLSSpec.
    """

    solver_type: str = Field(
        ...,
        description="Type of linesearch solver"
    )

    options: dict = Field(
        default_factory=dict,
        description="Solver options (e.g., maxiter, atol, rtol, etc.)"
    )


@register_solver_spec
class BoundsEnforceLSSpec(LinesearchSolverBaseSpec):
    """
    Specification for BoundsEnforceLS linesearch solver.

    BoundsEnforceLS enforces bounds on variables during line search.
    """

    solver_type: Literal['BoundsEnforceLS'] = 'BoundsEnforceLS'

    options: BoundsEnforceLSOptionsSpec = Field(
        default_factory=BoundsEnforceLSOptionsSpec,
        description='Options for BoundsEnforceLS.'
    )


@register_solver_spec
class ArmijoGoldsteinLSSpec(LinesearchSolverBaseSpec):
    """
    Specification for ArmijoGoldsteinLS linesearch solver.

    ArmijoGoldsteinLS uses Armijo-Goldstein line search for robustness.
    """

    solver_type: Literal['ArmijoGoldsteinLS'] = 'ArmijoGoldsteinLS'

    options: LinesearchSolverOptionsSpec = Field(
        default_factory=LinesearchSolverOptionsSpec,
        description='Options for ArmijoGoldsteinLS.'
    )


@register_solver_spec
class NewtonSolverSpec(NonlinearSolverBaseSpec):
    """
    Specification for NewtonSolver.

    NewtonSolver is a Newton-Raphson solver that can optionally solve subsystems
    (Hybrid Newton) and include a linesearch for robustness.

    Examples
    --------
    Basic Newton solver:
        NewtonSolverSpec(
            options=NewtonSolverOptionsSpec(solve_subsystems=True, maxiter=10)
        )

    Newton with linesearch:
        NewtonSolverSpec(
            options=NewtonSolverOptionsSpec(solve_subsystems=False, maxiter=20),
            linesearch=LinesearchSolverSpec(
                solver_type='ArmijoGoldsteinLS',
            )
        )
    """

    solver_type: Literal['NewtonSolver'] = 'NewtonSolver'

    options: NewtonSolverOptionsSpec = Field(
        default_factory=lambda: NewtonSolverOptionsSpec(solve_subsystems=False),
        description='Options for NewtonSolver.'
    )

    linesearch: LinesearchSolverBaseSpec | None = Field(
        default=None,
        description="Optional linesearch solver (typically used with NewtonSolver or BroydenSolver)"
    )

    @field_validator('linesearch', mode='before')
    @classmethod
    def route_linesearch_to_concrete_spec(cls, v):
        """Route linesearch solver dicts to their concrete spec types via registry."""
        solver_type = None

        if isinstance(v, dict):
            solver_type = v.get('solver_type')
        elif hasattr(v, 'solver_type'):
            # Handle LinesearchSolverSpec instances
            solver_type = v.solver_type
            v = v.model_dump() if hasattr(v, 'model_dump') else v

        if solver_type is not None:
            spec_class = _SOLVER_SPEC_REGISTRY.get(solver_type)
            if spec_class is not None:
                return spec_class.model_validate(v)

        return v

    @field_serializer('linesearch')
    def serialize_linesearch(self, v, _info):
        """Serialize linesearch spec properly."""
        if v is None:
            return None
        if hasattr(v, 'model_dump'):
            return v.model_dump()
        return v


@register_solver_spec
class NonlinearBlockGSSpec(NonlinearSolverBaseSpec):
    """
    Specification for NonlinearBlockGS (Nonlinear Block Gauss-Seidel).

    This solver sequentially solves subsystems using a fixed-point iteration
    approach. Can optionally use Aitken relaxation for acceleration.

    Examples
    --------
    Basic Block GS solver:
        NonlinearBlockGSSpec(
            options=NonlinearBlockGSOptionsSpec(maxiter=20)
        )

    Block GS with Aitken:
        NonlinearBlockGSSpec(
            options=NonlinearBlockGSOptionsSpec(
                use_aitken=True,
                maxiter=30
            )
        )
    """

    solver_type: Literal['NonlinearBlockGS'] = 'NonlinearBlockGS'

    options: NonlinearBlockGSOptionsSpec = Field(
        default_factory=NonlinearBlockGSOptionsSpec,
        description='Options for NonlinearBlockGS.'
    )


@register_solver_spec
class NonlinearBlockJacSpec(NonlinearSolverBaseSpec):
    """
    Specification for NonlinearBlockJac (Nonlinear Block Jacobi).

    This solver solves subsystems in parallel using a Jacobi-style iteration.
    Uses only base nonlinear solver options.

    Examples
    --------
    Basic Block Jacobi solver:
        NonlinearBlockJacSpec(
            options=NonlinearBlockJacOptionsSpec(maxiter=15)
        )
    """

    solver_type: Literal['NonlinearBlockJac'] = 'NonlinearBlockJac'

    options: NonlinearBlockJacOptionsSpec = Field(
        default_factory=NonlinearBlockJacOptionsSpec,
        description='Options for NonlinearBlockJac.'
    )


@register_solver_spec
class BroydenSolverSpec(NonlinearSolverBaseSpec):
    """
    Specification for BroydenSolver.

    BroydenSolver uses Broyden's method to solve nonlinear systems by building
    up an approximate Jacobian using secant updates.

    Examples
    --------
    Basic Broyden solver:
        BroydenSolverSpec(
            options=BroydenSolverOptionsSpec(maxiter=30, alpha=0.5)
        )

    Broyden with state variables:
        BroydenSolverSpec(
            options=BroydenSolverOptionsSpec(
                state_vars=['x', 'y'],
                maxiter=50
            )
        )
    """

    solver_type: Literal['BroydenSolver'] = 'BroydenSolver'

    options: BroydenSolverOptionsSpec = Field(
        default_factory=BroydenSolverOptionsSpec,
        description='Options for BroydenSolver.'
    )

    linesearch: Optional['LinesearchSolverBaseSpec'] = Field(
        default=None,
        description="Optional linesearch solver (typically used with BroydenSolver)"
    )

    @field_validator('linesearch', mode='before')
    @classmethod
    def route_linesearch_to_concrete_spec(cls, v):
        """Route linesearch solver dicts to their concrete spec types via registry."""
        solver_type = None

        if isinstance(v, dict):
            solver_type = v.get('solver_type')
        elif hasattr(v, 'solver_type'):
            # Handle LinesearchSolverSpec instances
            solver_type = v.solver_type
            v = v.model_dump() if hasattr(v, 'model_dump') else v

        if solver_type is not None:
            spec_class = _SOLVER_SPEC_REGISTRY.get(solver_type)
            if spec_class is not None:
                return spec_class.model_validate(v)

        return v

    @field_serializer('linesearch')
    def serialize_linesearch(self, v, _info):
        """Serialize linesearch spec properly."""
        if v is None:
            return None
        if hasattr(v, 'model_dump'):
            return v.model_dump()
        return v


@register_solver_spec
class NonlinearRunOnceSpec(NonlinearSolverBaseSpec):
    """
    Specification for NonlinearRunOnce.

    NonlinearRunOnce runs the nonlinear solver only once (no iterations).
    Useful for explicit systems or when you just want to evaluate once.

    Examples
    --------
    Run-once solver:
        NonlinearRunOnceSpec(
            options=NonlinearRunOnceOptionsSpec()
        )
    """

    solver_type: Literal['NonlinearRunOnce'] = 'NonlinearRunOnce'

    options: NonlinearRunOnceOptionsSpec = Field(
        default_factory=NonlinearRunOnceOptionsSpec,
        description='Options for NonlinearRunOnce.'
    )


# Rebuild specs to resolve forward references
NonlinearSolverSpec.model_rebuild()
NewtonSolverSpec.model_rebuild()
BroydenSolverSpec.model_rebuild()
