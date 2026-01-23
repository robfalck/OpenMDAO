import collections
from typing import Literal, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel, Field, model_validator, field_validator

from openmdao.specs.connection_spec import ConnectionSpec
from openmdao.specs.systems_registry import register_system_spec
from openmdao.specs.input_defaults_spec import InputDefaultsSpec
from openmdao.specs.promotes_spec import PromotesSpec


if TYPE_CHECKING:
    from openmdao.specs.subsystem_spec import SubsystemSpec


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


class LinesearchSolverSpec(BaseModel):
    """
    Specification for a linesearch solver.

    Linesearch solvers are used as nested solvers within nonlinear solvers
    (typically Newton or Broyden) to improve convergence.

    Examples
    --------
    Basic linesearch:
        LinesearchSolverSpec(
            solver_type='ArmijoGoldsteinLS',
            options={'bound_enforcement': 'vector'}
        )

    Bounds enforcement linesearch:
        LinesearchSolverSpec(
            solver_type='BoundsEnforceLS',
            options={}
        )
    """

    solver_type: LinesearchSolverType | Literal["BoundsEnforceLS", "ArmijoGoldsteinLS"] = Field(
        ...,
        description="Type of linesearch solver"
    )

    options: dict = Field(
        default_factory=dict,
        description="Linesearch solver options"
    )


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

    linesearch: LinesearchSolverSpec | None = Field(
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


@register_system_spec
class GroupSpec(BaseModel):
    """Specification for a group in isolation."""

    type: Literal["group"] = "group"
    
    # Child subsystems - use string annotation to avoid circular import
    subsystems: list['SubsystemSpec'] = Field(
        default_factory=list,
        description="Child subsystems with their inclusion specifications"
    )
    
    # Connections within this group
    connections: list[ConnectionSpec] = Field(
        default_factory=list,
        description="Connections between subsystems"
    )
    
    # Solver specifications
    nonlinear_solver: NonlinearSolverSpec | None = Field(
        default=None,
        description="Nonlinear solver for this group"
    )

    linear_solver: LinearSolverSpec | None = Field(
        default=None,
        description="Linear solver for this group"
    )

    input_defaults : list[InputDefaultsSpec] | dict[str, dict] = Field(
        default_factory=list,
        description='Default values for any inputs to be provided by automatic indep var comps.'
    )

    promotes: list[PromotesSpec] = Field(
        default_factory=list,
        description="Advanced promotions for specific subsystems via Group.promotes(). "
                    "Each PromotesSpec with a subsys_name specifies a Group.promotes() call."
    )

    # Group behavior
    assembled_jac_type: Literal["csc", "dense", None] = Field(
        default=None,
        description="Type of assembled Jacobian"
    )

    @field_validator('input_defaults', mode='before')
    @classmethod
    def pre_validate_input_defaults(cls, v):
        """
        Validate input defaults.

        If provided as a dictionary, convert to a list.
        """
        if isinstance(v, dict):
            return [InputDefaultsSpec(name=name, **meta) for name, meta in v.items()]
        return v

    @field_validator('input_defaults', mode='after')
    @classmethod
    def post_validate_input_defaults(cls, v):
        """
        Validate input defaults after creation.

        Ensure elements in input_defaults have unique names.
        """
        names = [item.name for item in v]
        name_counts = collections.Counter(names)
        dups = [name for name, count in name_counts.items() if count > 1]

        if dups:
            raise ValueError(f'The following names have multiple input defaults: {dups}')

        return v



    @model_validator(mode='after')
    def validate_connections(self):
        """Validate that connections reference valid subsystems."""
        subsystem_names = {subsys.name for subsys in self.subsystems}

        for conn in self.connections:
            src_system = conn.src.split('.')[0]
            tgt_system = conn.tgt.split('.')[0]

            if src_system not in subsystem_names:
                raise ValueError(f"Connection source '{conn.src}' references unknown "
                                 "system '{src_system}'")
            if tgt_system not in subsystem_names:
                raise ValueError(f"Connection target '{conn.tgt}' references unknown "
                                 "system '{tgt_system}'")

        return self