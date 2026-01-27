from pydantic import BaseModel, Field, field_validator, field_serializer, model_validator, \
    ConfigDict

from openmdao.specs.system_spec import SystemSpec
from openmdao.specs.promotes_spec import PromotesSpec
from openmdao.specs.systems_registry import _SYSTEM_SPEC_REGISTRY
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
    
    system: SystemSpec = Field(
        ...,
        description="The system specification (component or group)"
    )

    # Promotion specifications
    promotes: list[PromotesSpec] | list[str | tuple[str, str]] | None = Field(
        default=None,
        description="Variables to promote (inputs and outputs)."
    )

    promotes_inputs: list[PromotesSpec] | list[str | tuple[str, str]] | None = Field(
        default=None,
        description="Input variables to promote."
    )

    promotes_outputs: list[PromotesSpec] | list[str | tuple[str, str]] | None = Field(
        default=None,
        description="Output variables to promote."
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


    @field_validator('system', mode='before')
    @classmethod
    def deserialize_system(cls, v):
        """
        Deserialize system using the registry to get the correct subclass.

        This allows proper deserialization of ComponentSpec subclasses
        (OMExplicitComponentSpec, ExecCompSpec, etc.) based on their 'system_type' field.
        """
        # If it's already an instance, return as-is
        if isinstance(v, (ComponentSpec, GroupSpec)):
            return v

        # If it's a dict, look up the appropriate class from the registry
        if isinstance(v, dict):
            type_name = v.get('system_type')
            if type_name and type_name in _SYSTEM_SPEC_REGISTRY:
                # Use the registered class for this type
                spec_class = _SYSTEM_SPEC_REGISTRY[type_name]
                return spec_class.model_validate(v)
            # Fallback to default behavior if type not in registry

        return v

    @field_serializer('system')
    def serialize_system(self, system: SystemSpec, _info):
        """
        Serialize system using its actual class's model_dump.

        This ensures that subclass-specific fields (like 'system_type') are included,
        while respecting the parent's exclude_defaults setting.
        """
        # Use the actual class's model_dump to preserve subclass fields
        return system.model_dump(exclude_defaults=_info.exclude_defaults)
    
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

    @field_validator('promotes', 'promotes_inputs', 'promotes_outputs', mode='before')
    @classmethod
    def convert_legacy_promotes(cls, v, info):
        """Convert legacy str/tuple format to PromotesSpec objects.

        This validator provides backward compatibility for JSON specs that use
        the old format of simple lists of strings or tuples, converting them
        to PromotesSpec objects with appropriate io_type.
        """
        if v is None:
            return None

        if not isinstance(v, list):
            return v

        # Determine io_type from field name
        field_name = info.field_name
        if field_name == 'promotes_inputs':
            io_type = 'input'
        elif field_name == 'promotes_outputs':
            io_type = 'output'
        else:
            io_type = 'any'

        result = []
        for item in v:
            if isinstance(item, PromotesSpec):
                result.append(item)
            elif isinstance(item, (str, tuple)):
                # Convert legacy format to PromotesSpec
                result.append(PromotesSpec(name=item, io_type=io_type))
            elif isinstance(item, dict):
                # Convert from dict (e.g., from JSON)
                result.append(PromotesSpec.model_validate(item))
            else:
                # Try to validate it as PromotesSpec
                result.append(PromotesSpec.model_validate(item))

        return result
    
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

    @field_serializer('promotes', 'promotes_inputs', 'promotes_outputs')
    def serialize_promotes_lists(self, v, _info):
        """Serialize promotes lists respecting exclude_defaults."""
        if not isinstance(v, list):
            return v
        return [item.model_dump(exclude_defaults=_info.exclude_defaults)
                if hasattr(item, 'model_dump') else item for item in v]


# Rebuild models to resolve forward references now that all imports are complete
GroupSpec.model_rebuild()
