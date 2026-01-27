"""
Base specification for all OpenMDAO systems.

This module defines SystemSpec, a base class for both ComponentSpec and GroupSpec
that captures properties common to all systems including solvers, Jacobian options,
derivatives methods, and recording options.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, field_serializer

from openmdao.specs.design_var_spec import DesignVarSpec
from openmdao.specs.response_spec import ConstraintSpec, ObjectiveSpec
from openmdao.specs.options_spec import RecordingOptionsSpec
# Imports at the end to avoid circular imports and ensure proper resolution
from openmdao.specs.nonlinear_solver_spec import NonlinearSolverBaseSpec
from openmdao.specs.linear_solver_spec import LinearSolverBaseSpec
from openmdao.specs.solver_registry import _SOLVER_SPEC_REGISTRY



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

    system_type: str = Field(
        ...,
        description="Type identifier for this system (overridden by subclasses)",
        frozen=True
    )

    nonlinear_solver: NonlinearSolverBaseSpec | None = Field(
        default=None,
        description="Nonlinear solver for this system. "
                    "Used for implicit components and groups with coupling."
    )

    linear_solver: LinearSolverBaseSpec | None = Field(
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

    @field_validator('nonlinear_solver', mode='before')
    @classmethod
    def route_nonlinear_solver_to_concrete_spec(cls, v):
        """Route nonlinear solver dicts to their concrete spec types via registry."""
        if isinstance(v, dict):
            solver_type = v.get('solver_type')
            if solver_type is not None:
                spec_class = _SOLVER_SPEC_REGISTRY.get(solver_type)
                if spec_class is not None:
                    return spec_class.model_validate(v)
        return v

    @field_validator('linear_solver', mode='before')
    @classmethod
    def route_linear_solver_to_concrete_spec(cls, v):
        """Route linear solver dicts to their concrete spec types via registry."""
        if isinstance(v, dict):
            solver_type = v.get('solver_type')
            if solver_type is not None:
                spec_class = _SOLVER_SPEC_REGISTRY.get(solver_type)
                if spec_class is not None:
                    return spec_class.model_validate(v)
        return v

    @field_serializer('nonlinear_solver')
    def serialize_nonlinear_solver(self, v, _info):
        """Serialize nonlinear solver spec properly."""
        if v is None:
            return None
        # Serialize with exclude_defaults but ensure solver_type is always included
        if hasattr(v, 'model_dump'):
            data = v.model_dump(exclude_defaults=_info.exclude_defaults)
            # Ensure solver_type is always included (needed for deserialization routing)
            if hasattr(v, 'solver_type'):
                data['solver_type'] = v.solver_type
            return data
        return v

    @field_serializer('linear_solver')
    def serialize_linear_solver(self, v, _info):
        """Serialize linear solver spec properly."""
        if v is None:
            return None
        # Serialize with exclude_defaults but ensure solver_type is always included
        if hasattr(v, 'model_dump'):
            data = v.model_dump(exclude_defaults=_info.exclude_defaults)
            # Ensure solver_type is always included (needed for deserialization routing)
            if hasattr(v, 'solver_type'):
                data['solver_type'] = v.solver_type
            return data
        return v

    @field_serializer('system_type', when_used='unless-none')
    def serialize_system_type(self, v):
        """Serialize system_type field (needed for deserialization routing)."""
        return v

# Rebuild SystemSpec to resolve all forward references
SystemSpec.model_rebuild()
