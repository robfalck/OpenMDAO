"""
Base specification for all OpenMDAO systems.

This module defines SystemSpec, a base class for both ComponentSpec and GroupSpec
that captures properties common to all systems including solvers, Jacobian options,
derivatives methods, and recording options.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from openmdao.specs.design_var_spec import DesignVarSpec
from openmdao.specs.response_spec import ConstraintSpec, ObjectiveSpec
from openmdao.specs.options_spec import RecordingOptionsSpec
# Imports at the end to avoid circular imports and ensure proper resolution
from openmdao.specs.solver_spec import NonlinearSolverSpec, LinearSolverSpec  # noqa: E402, F401



class SystemSpec(BaseModel):
    """
    Base specification for all OpenMDAO systems (Components and Groups).

    This class captures properties common to both components and groups,
    including solver specifications, Jacobian assembly type, derivatives method,
    and recording options.

    These are schema-level properties that apply to all systems, as opposed to
    runtime options which are handled via the options field in ComponentSpec
    and SystemOptionsSpec.

    Field descriptions provide details for each attribute.
    """

    nonlinear_solver: NonlinearSolverSpec | None = Field(
        default=None,
        description="Nonlinear solver for this system. "
                    "Used for implicit components and groups with coupling."
    )

    linear_solver: LinearSolverSpec | None = Field(
        default=None,
        description="Linear solver for this system. "
                    "Used for computing derivatives and linear solutions."
    )

    assembled_jac_type: Literal["csc", "csr", "dense"] | None = Field(
        default=None,
        description="Type of assembled Jacobian to use. "
                    "Options: 'csc' (compressed sparse column), "
                    "'csr' (compressed sparse row), 'dense'. "
                    "Only valid for Groups and ImplicitComponents."
    )

    derivs_method: Literal["jax", "cs", "fd"] | None = Field(
        default=None,
        description="Method for computing derivatives. "
                    "Options: 'jax' (automatic differentiation), "
                    "'cs' (complex step), 'fd' (finite difference)."
    )

    recording_options: RecordingOptionsSpec | None = Field(
        default_factory=RecordingOptionsSpec,
        description="Options for recording system data during execution."
    )

    design_vars: list[DesignVarSpec] = Field(default_factory=list,
                                              description='Optimization design variables ' \
                                              'for this system')

    constraints: list[ConstraintSpec] = Field(default_factory=list,
                                              description="Optimization constraints "
                                              "for this system")

    objective: list[ObjectiveSpec] = Field(default_factory=list,
                                           description="Optimization objective "
                                           "for this system")

# Rebuild SystemSpec to resolve all forward references
SystemSpec.model_rebuild()
