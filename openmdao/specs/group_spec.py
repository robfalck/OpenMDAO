from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator

from openmdao.specs.connection_spec import ConnectionSpec


if TYPE_CHECKING:
    from openmdao.specs.subsystem_spec import SubsystemSpec


class SolverSpec(BaseModel):
    """Specification for a solver."""

    solver_type: str  # Would be an enum in full implementation
    options: dict = Field(default_factory=dict)


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
    nonlinear_solver: SolverSpec | None = Field(
        default=None,
        description="Nonlinear solver for this group"
    )
    
    linear_solver: SolverSpec | None = Field(
        default=None,
        description="Linear solver for this group"
    )
    
    # Group behavior
    assembled_jac_type: Literal["csc", "dense", None] = Field(
        default=None,
        description="Type of assembled Jacobian"
    )
    
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