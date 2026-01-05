from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.group_spec import GroupSpec



class SubsystemSpec(BaseModel):
    """
    Specification for how a system is included in a group.
    
    This combines:
    - The system definition itself (ComponentSpec or GroupSpec)
    - How it's named in the parent
    - Promotion specifications
    - MPI allocation options
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    # Required: name and the system itself
    name: str = Field(
        ...,
        description="Name of the subsystem in the parent group. Must be Pythonic."
    )
    
    system: ComponentSpec | GroupSpec = Field(
        ...,
        description="The system specification (component or group)"
    )
    
    # Promotion specifications
    promotes: list[str | tuple[str, str]] | None = Field(
        default=None,
        description="Variables to promote (inputs and outputs). "
                    "Can be strings or (old_name, new_name) tuples."
    )
    
    promotes_inputs: list[str | tuple[str, str]] | None = Field(
        default=None,
        description="Input variables to promote. Can be strings or (old_name, new_name) tuples."
    )
    
    promotes_outputs: list[str | tuple[str, str]] | None = Field(
        default=None,
        description="Output variables to promote. Can be strings or (old_name, new_name) tuples."
    )
    
    # MPI-related options
    min_procs: int = Field(
        default=1,
        ge=1,
        description="Minimum number of MPI processes usable by the subsystem."
    )
    
    max_procs: int | None = Field(
        default=None,
        description="Maximum number of MPI processes usable by the subsystem."
    )
    
    proc_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight given to the subsystem when allocating available MPI processes."
    )
    
    proc_group: str | None = Field(
        default=None,
        description="Name of a processor group for co-allocation on same MPI process(es)."
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate that name is Pythonic."""
        if not v:
            raise ValueError("Subsystem name cannot be empty")
        if not v[0].isalpha():
            raise ValueError("Subsystem name must start with a letter")
        if not all(c.isalnum() or c == '_' for c in v):
            raise ValueError("Subsystem name must contain only alphanumeric characters and "
                             "underscores")
        return v
    
    @field_validator('max_procs')
    @classmethod
    def validate_max_procs(cls, v, info):
        """Ensure max_procs >= min_procs if specified."""
        if v is not None and 'min_procs' in info.data:
            min_procs = info.data['min_procs']
            if v < min_procs:
                raise ValueError(f"max_procs ({v}) must be >= min_procs ({min_procs})")
        return v
    
    @model_validator(mode='after')
    def validate_promotion_consistency(self):
        """Validate that promotion specifications are consistent."""
        if self.promotes is not None and (self.promotes_inputs is not None or 
                                          self.promotes_outputs is not None):
            raise ValueError(
                "Cannot specify 'promotes' together with 'promotes_inputs' or 'promotes_outputs'."
            )
        return self


# Rebuild GroupSpec to resolve the 'SubsystemSpec' forward reference
GroupSpec.model_rebuild()
