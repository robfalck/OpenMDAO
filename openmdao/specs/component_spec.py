from typing import Literal, Generic

from pydantic import BaseModel, Field, field_validator

from openmdao.specs.variable_spec import VariableSpec
from openmdao.specs.systems_registry import register_system_spec
from openmdao.specs.options_spec import ComponentOptionsSpec, ComponentOptionsT


class ComponentSpec(BaseModel, Generic[ComponentOptionsT]):
    """Base specification for a component."""

    inputs: list[VariableSpec] = Field(
        ...,
        description='Input variables for this component.'
    )

    outputs: list[VariableSpec] = Field(
        ...,
        description='Output variables for this component.'
    )

    options: ComponentOptionsT | None = Field(
        default=None,
        description='User-configurable options on the component.'
    )

    @field_validator('inputs', 'outputs', mode='before')
    @classmethod
    def convert_to_variable_specs(cls, v):
        """
        Convert inputs/outputs to list of VariableSpec.
        
        Accepts:
        - List of VariableSpec instances
        - List of dicts: [{'name': 'x', ...}, {'name': 'y', ...}]
        - List of nested dicts: [{'x': {...}}, {'y': {...}}]
        - Dict mapping names to specs: {'x': {...}, 'y': {...}}
        - Single VariableSpec
        - Single dict
        """
        # If it's a dict (not a list), convert to list of dicts
        if isinstance(v, dict) and not isinstance(v, VariableSpec):
            # Check if it's a mapping of variable names to specs
            # {'x': {'shape': (1,)}, 'y': {'shape': (2,)}}
            v = [{'name': name, **props} if isinstance(props, dict) else {'name': name}
                for name, props in v.items()]
        
        if not isinstance(v, list):
            v = [v]
        
        result = []
        for item in v:
            if isinstance(item, VariableSpec):
                result.append(item)
            elif isinstance(item, dict):
                # Check if it's nested format: {'varname': {...}}
                if len(item) == 1 and 'name' not in item:
                    # Extract name from key
                    name, props = next(iter(item.items()))
                    if isinstance(props, dict):
                        props['name'] = name
                        result.append(VariableSpec.model_validate(props))
                    else:
                        raise ValueError(f"Invalid variable spec format: {item}")
                else:
                    # Standard format: {'name': 'x', ...}
                    result.append(VariableSpec.model_validate(item))
            else:
                raise ValueError(f"Invalid type: {type(item)}")
        
        return result


@register_system_spec
class OMExplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    type: Literal['OMExplicitComponent'] = 'OMExplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )

    @field_validator('options', mode='before')
    @classmethod
    def check_explicit_component_options(cls, v):
        """
        Validate that explicit components don't use implicit-only options.

        This is the equivalent of options.undeclare in OpenMDAO 3.x models.
        """
        if v is not None and v.assembled_jac_type is not None:
            raise ValueError(
                "assembled_jac_type is not valid for ExplicitComponents. "
                "This option only applies to Groups and ImplicitComponents."
            )
        return v


@register_system_spec
class OMImplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    type: Literal['OMImplicitComponent'] = 'OMImplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )


@register_system_spec
class OMJaxExplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    type: Literal['JaxExplicitComponent'] = 'JaxExplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )


@register_system_spec
class OMJaxImplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    type: Literal['OMJaxImplicitComponent'] = 'OMJaxImplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )
