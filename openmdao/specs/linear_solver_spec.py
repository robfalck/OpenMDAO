from typing import Literal, Optional
from pydantic import BaseModel, Field

from openmdao.specs.solver_options_base import SolverOptionsSpec
from openmdao.specs.solver_registry import register_solver_spec


class LinearSolverOptionsSpec(SolverOptionsSpec):
    """Base specification for linear solver options (common to all linear solvers)."""

    pass


class DirectSolverOptionsSpec(LinearSolverOptionsSpec):
    """Options specific to DirectSolver (only inherits base options)."""

    pass


class LinearBlockGSOptionsSpec(LinearSolverOptionsSpec):
    """Options specific to LinearBlockGS."""

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


class LinearBlockJacOptionsSpec(LinearSolverOptionsSpec):
    """Options specific to LinearBlockJac (only inherits base options)."""

    pass


class LinearRunOnceOptionsSpec(LinearSolverOptionsSpec):
    """Options specific to LinearRunOnce (only inherits base options)."""
    
    pass


class ScipyKrylovOptionsSpec(LinearSolverOptionsSpec):
    """Options specific to ScipyKrylov."""

    pass


class PETScKrylovOptionsSpec(LinearSolverOptionsSpec):
    """Options specific to PETScKrylov."""

    ksp_type: str = Field(
        default='gmres',
        description='PETSc KSP method type (gmres, cg, bcgs, etc.)'
    )

    pc_type: str = Field(
        default='none',
        description='PETSc preconditioner type (none, jacobi, ilu, bjacobi, etc.)'
    )

    ksp_max_it: Optional[int] = Field(
        default=None,
        description='Maximum iterations for PETSc KSP (if None, uses maxiter from base).'
    )

    ksp_rtol: Optional[float] = Field(
        default=None,
        description='Relative tolerance for PETSc KSP (if None, uses rtol from base).'
    )

    ksp_atol: Optional[float] = Field(
        default=None,
        description='Absolute tolerance for PETSc KSP (if None, uses atol from base).'
    )


class LinearSolverBaseSpec(BaseModel):
    """
    Base class for all linear solver specifications.

    This class provides a common type for type annotations and registration,
    without defining any fields.
    """

    solver_type: str = Field(...,
        description='An identifier for the linear solver class.')


@register_solver_spec
class DirectSolverSpec(LinearSolverBaseSpec):
    """
    Specification for DirectSolver.

    DirectSolver uses direct dense matrix factorization to solve the linear system.
    It's typically the most robust but can be memory-intensive for large systems.

    Examples
    --------
    Basic direct solver:
        DirectSolverSpec(
            solver_type='DirectSolver',
            options=DirectSolverOptionsSpec()
        )
    """

    solver_type: Literal['DirectSolver'] = 'DirectSolver'

    options: DirectSolverOptionsSpec | dict = Field(
        default_factory=DirectSolverOptionsSpec,
        description='Options for DirectSolver.'
    )


@register_solver_spec
class LinearBlockGSSpec(LinearSolverBaseSpec):
    """
    Specification for LinearBlockGS (Linear Block Gauss-Seidel).

    This solver sequentially solves subsystem linear systems using a fixed-point
    iteration approach. Can optionally use Aitken relaxation for acceleration.

    Examples
    --------
    Basic Block GS solver:
        LinearBlockGSSpec(
            solver_type='LinearBlockGS',
            options=LinearBlockGSOptionsSpec(maxiter=20)
        )

    Block GS with Aitken:
        LinearBlockGSSpec(
            solver_type='LinearBlockGS',
            options=LinearBlockGSOptionsSpec(
                use_aitken=True,
                maxiter=30
            )
        )
    """

    solver_type: Literal['LinearBlockGS'] = 'LinearBlockGS'

    options: LinearBlockGSOptionsSpec | dict = Field(
        default_factory=LinearBlockGSOptionsSpec,
        description='Options for LinearBlockGS.'
    )


@register_solver_spec
class LinearBlockJacSpec(LinearSolverBaseSpec):
    """
    Specification for LinearBlockJac (Linear Block Jacobi).

    This solver solves subsystem linear systems in parallel using a Jacobi-style iteration.
    Uses only base linear solver options.

    Examples
    --------
    Basic Block Jacobi solver:
        LinearBlockJacSpec(
            solver_type='LinearBlockJac',
            options=LinearBlockJacOptionsSpec(maxiter=15)
        )
    """

    solver_type: Literal['LinearBlockJac'] = 'LinearBlockJac'

    options: LinearBlockJacOptionsSpec | dict = Field(
        default_factory=LinearBlockJacOptionsSpec,
        description='Options for LinearBlockJac.'
    )


@register_solver_spec
class LinearRunOnceSpec(LinearSolverBaseSpec):
    """
    Specification for LinearRunOnce.

    LinearRunOnce runs the linear solver only once (no iterations).
    Useful when you just need a single factorization or solve.

    Examples
    --------
    Run-once solver:
        LinearRunOnceSpec(
            solver_type='LinearRunOnce',
            options=LinearRunOnceOptionsSpec()
        )
    """

    solver_type: Literal['LinearRunOnce'] = 'LinearRunOnce'

    options: LinearRunOnceOptionsSpec | dict = Field(
        default_factory=LinearRunOnceOptionsSpec,
        description='Options for LinearRunOnce.'
    )


@register_solver_spec
class ScipyKrylovSpec(LinearSolverBaseSpec):
    """
    Specification for ScipyKrylov.

    ScipyKrylov uses Krylov subspace methods from scipy.sparse.linalg for solving
    linear systems. Supports various methods like LGMRES, LSMR, etc.

    Examples
    --------
    Basic Krylov solver:
        ScipyKrylovSpec(
            options=ScipyKrylovOptionsSpec(
                maxiter=100
            )
        )

    With preconditioner:
        ScipyKrylovSpec(
            options=ScipyKrylovOptionsSpec(
                maxiter=50
            )
        )
    """

    solver_type: Literal['ScipyKrylov'] = 'ScipyKrylov'

    options: ScipyKrylovOptionsSpec | dict = Field(
        default_factory=ScipyKrylovOptionsSpec,
        description='Options for ScipyKrylov.'
    )


@register_solver_spec
class PETScKrylovSpec(LinearSolverBaseSpec):
    """
    Specification for PETScKrylov.

    PETScKrylov uses Krylov subspace methods from PETSc for solving linear systems.
    Provides access to PETSc's extensive solver and preconditioner options.

    Examples
    --------
    Basic PETSc Krylov solver:
        PETScKrylovSpec(
            solver_type='PETScKrylov',
            options=PETScKrylovOptionsSpec(
                maxiter=100
            )
        )

    With ILU preconditioner:
        PETScKrylovSpec(
            solver_type='PETScKrylov',
            options=PETScKrylovOptionsSpec(
                ksp_type='cg',
                pc_type='ilu',
                maxiter=50
            )
        )
    """

    solver_type: Literal['PETScKrylov'] = 'PETScKrylov'

    options: PETScKrylovOptionsSpec | dict = Field(
        default_factory=PETScKrylovOptionsSpec,
        description='Options for PETScKrylov.'
    )
