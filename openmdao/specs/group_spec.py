import collections
from typing import TYPE_CHECKING
# from enum import Enum
from pydantic import Field, model_validator, field_validator, field_serializer

from openmdao.specs.system_spec import SystemSpec
from openmdao.specs.connection_spec import ConnectionSpec
from openmdao.specs.systems_registry import register_system_spec
from openmdao.specs.input_defaults_spec import InputDefaultsSpec
from openmdao.specs.promotes_spec import PromotesSpec


if TYPE_CHECKING:
    from openmdao.specs.subsystem_spec import SubsystemSpec


@register_system_spec
class GroupSpec(SystemSpec):
    """Specification for a group in isolation."""

    system_type: str = "group"

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

    input_defaults : list[InputDefaultsSpec] | dict[str, dict] = Field(
        default_factory=list,
        description='Default values for any inputs to be provided by automatic indep var comps.'
    )

    promotes: list[PromotesSpec] = Field(
        default_factory=list,
        description="Advanced promotions for specific subsystems via Group.promotes(). "
                    "Each PromotesSpec with a subsys_name specifies a Group.promotes() call."
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

    @field_serializer('subsystems', 'connections', 'promotes')
    def serialize_lists(self, v, _info):
        """Serialize list fields respecting exclude_defaults."""
        if not isinstance(v, list):
            return v
        return [item.model_dump(exclude_defaults=_info.exclude_defaults)
                if hasattr(item, 'model_dump') else item for item in v]

    def setup(self, group):
        """
        Set up a Group instance from this spec.

        This method is called during the group's setup phase to add
        subsystems, connections, input defaults, and promotions.

        Parameters
        ----------
        group : Group
            The group instance to configure.
        """
        # Import here to avoid circular imports
        from openmdao.specs.instantiation import (
            instantiate_from_spec, _extract_promote_names, _apply_promotes_call
        )

        # Add subsystems recursively
        for subsys_spec in self.subsystems:
            subsys = instantiate_from_spec(subsys_spec.system)

            # Convert PromotesSpec objects to simple names/tuples for add_subsystem()
            promotes_list = _extract_promote_names(subsys_spec.promotes)
            promotes_inputs_list = _extract_promote_names(subsys_spec.promotes_inputs)
            promotes_outputs_list = _extract_promote_names(subsys_spec.promotes_outputs)

            group.add_subsystem(
                subsys_spec.name,
                subsys,
                promotes_inputs=promotes_inputs_list,
                promotes_outputs=promotes_outputs_list,
                promotes=promotes_list,
                min_procs=subsys_spec.min_procs,
                max_procs=subsys_spec.max_procs,
                proc_weight=subsys_spec.proc_weight
            )

        # Add connections
        for conn_spec in self.connections:
            src_indices = None
            if conn_spec.src_indices.value is not None:
                src_indices = conn_spec.src_indices.value

            group.connect(conn_spec.src, conn_spec.tgt, src_indices=src_indices)

        # Set input defaults
        for indef_spec in self.input_defaults:
            group.set_input_defaults(name=indef_spec.name, val=indef_spec.val,
                                    units=indef_spec.units, src_shape=indef_spec.src_shape)

        # Add advanced promotions via Group.promotes() method calls
        # Group by subsys_name to collect all promotes for each subsystem
        promotes_by_subsys = {}
        for pspec in self.promotes:
            if pspec.subsys_name is not None:
                if pspec.subsys_name not in promotes_by_subsys:
                    promotes_by_subsys[pspec.subsys_name] = []
                promotes_by_subsys[pspec.subsys_name].append(pspec)

        for subsys_name, promotes_specs in promotes_by_subsys.items():
            _apply_promotes_call(group, subsys_name, promotes_specs)