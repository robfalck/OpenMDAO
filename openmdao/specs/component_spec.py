from typing import Literal

from pydantic import BaseModel, Field, field_validator

from openmdao.specs.variable_spec import VariableSpec
from openmdao.specs.systems_registry import register_system_spec


class ComponentSpec(BaseModel):
    """Base specification for a component."""

    inputs : list[VariableSpec] = Field(
        ...,
        description='Input variables for this component.'
    )

    outputs : list[VariableSpec] = Field(
        ...,
        description='Output variables for this component.'
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
class OMExplicitComponentSpec(ComponentSpec):

    type: Literal['OMExplicitComponent'] = 'OMExplicitComponent'

    path : str | None = Field(
        ...,
        description='The dotted path to the component class definition.'
    )

@register_system_spec
class OMImplicitComponentSpec(ComponentSpec):

    type: Literal['OMImplicitComponent'] = 'OMImplicitComponent'

    path : str | None = Field(
        ...,
        description='The dotted path to the component class definition.'
    )

@register_system_spec
class OMJaxExplicitComponentSpec(ComponentSpec):

    type: Literal['JaxExplicitComponent'] = 'JaxExplicitComponent'

    path : str | None = Field(
        ...,
        description='If given, the dotted path to the component class definition.'
    )

@register_system_spec
class OMJaxImplicitComponentSpec(ComponentSpec):

    type: Literal['JaxExplicitComponent'] = 'JaxExplicitComponent'

    path : str | None = Field(
        ...,
        description='The dotted path to the component class definition.'
    )




    # compute : str | None = Field(
    #     default=None,
    #     description="If given, the dotted path to the component's compute function."
    # )

    # compute_partials : str | None = Field(
    #     default=None,
    #     description="If given, the dotted path to the component's compute_partials function."
    # )

    # compute_jacvec_prod : str | None = Field(
    #     default=None,
    #     description="If given, the dotted path to the component's compute_jacvec_prod function.")

    # partials : list[PartialsSpec] = Field(
    #     default_factory=list,
    #     description="Partial derivatives declared for the outputs of this component."
    # )

    # @field_validator('partials', mode='before')
    # @classmethod
    # def convert_partials_to_list(cls, v):
    #     """Convert any collection of PartialsSpec to a list."""
    #     if isinstance(v, list):
    #         return v
    #     return list(v)
