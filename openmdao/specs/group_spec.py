from typing import Literal
from pydantic import BaseModel, Field, model_validator

from openmdao.specs.partials_spec import PartialsSpec
from openmdao.specs.subsystem_spec import SubsystemSpec

# ============================================================================
# Core System Specifications (define systems in isolation)
# ============================================================================

from openmdao.specs import ConnectionSpec


class ComponentSpec(BaseModel):
    """Specification for a component in isolation."""
    
    component_class: str = Field(
        ...,
        description="Fully qualified component class name"
    )
    
    init_kwargs: dict = Field(
        default_factory=dict,
        description="Component initialization kwargs"
    )
    
    # Execution specification
    is_explicit: bool = Field(
        default=True,
        description="Whether component is explicit"
    )
    
    distributed: bool = Field(
        default=False,
        description="Whether component is distributed"
    )
    
    # Partial derivatives specification
    partials: list['PartialsSpec'] = Field(
        default_factory=list,
        description="Partial derivative declarations"
    )


class SolverSpec(BaseModel):
    """Specification for a solver."""

    solver_type: str  # Would be an enum in full implementation
    options: dict = Field(default_factory=dict)


class GroupSpec(BaseModel):
    """Specification for a group in isolation."""
    
    type: Literal["group"] = "group"
    
    # Child subsystems (note: these are SubsystemSpecs, not SystemSpecs directly)
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
            src_system = conn.source.split('.')[0]
            tgt_system = conn.target.split('.')[0]
            
            if src_system not in subsystem_names:
                raise ValueError(f"Connection source '{conn.source}' references unknown "
                                 "system '{src_system}'")
            if tgt_system not in subsystem_names:
                raise ValueError(f"Connection target '{conn.target}' references unknown "
                                 "system '{tgt_system}'")
        
        return self
