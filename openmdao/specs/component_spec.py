from typing import Generic, Literal, Any

from pydantic import Field, field_validator, field_serializer

from openmdao.specs.system_spec import SystemSpec
from openmdao.specs.variable_spec import VariableSpec
from openmdao.specs.systems_registry import register_system_spec
from openmdao.specs.options_spec import ComponentOptionsSpec, ComponentOptionsT


class ComponentSpec(SystemSpec, Generic[ComponentOptionsT]):
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

    @field_serializer('inputs', 'outputs')
    def serialize_variables(self, v, _info):
        """Serialize variable lists respecting exclude_defaults."""
        if not isinstance(v, list):
            return v
        return [item.model_dump(exclude_defaults=_info.exclude_defaults)
                if hasattr(item, 'model_dump') else item for item in v]

    def setup(self, comp):
        """
        Configure a component instance from this spec.

        This method is called during the component's setup phase to add
        inputs, outputs, and optionally declare partials based on the spec.

        Parameters
        ----------
        comp : Component
            The component instance to configure.
        """
        # Add inputs
        for var_spec in self.inputs:
            kwargs = self._var_spec_to_add_io_kwargs(var_spec, 'input')
            comp.add_input(**kwargs)

        # Add outputs
        for var_spec in self.outputs:
            kwargs = self._var_spec_to_add_io_kwargs(var_spec, 'output')
            comp.add_output(**kwargs)

    def _var_spec_to_add_io_kwargs(self,
                                   var_spec: VariableSpec,
                                   iotype: Literal['input', 'output']):
        """
        Convert VariableSpec to add_input or add_output kwargs.

        Parameters
        ----------
        var_spec : VariableSpec
            The variable specification
        iotype : str
            Either 'input' or 'output' to specify the type of variable
        comp_options : ComponentOptionsSpec
            Component options used for referencing variable properties that
            are linked to options, such as "num_nodes" or "vec_size".

        Returns
        -------
        dict
            Keyword arguments for comp.add_input() or comp.add_output()
        """
        # Map VariableSpec fields to add_input/add_output parameters
        kwargs: dict[str, Any] = {'name': var_spec.name}

        # Add common optional parameters if they're not None/default
        if var_spec.val is not None:
            kwargs['val'] = var_spec.val
        if var_spec.shape is not None:
            shape = var_spec.shape
            # If any element in shape is a string, obtain the dimension size from the component
            # option of that name.
            if isinstance(shape, tuple) and any(isinstance(element, str) for element in shape):
                interp_shape = []
                for element in shape:
                    if isinstance(element, str):
                        try:
                            dim_size = getattr(self.options, element)
                        except AttributeError:
                            raise AttributeError(f'Shape specification {shape} containts a string '
                                                 f'{element} but the component options provide no '
                                                 f'option named {element}')
                        if not isinstance(dim_size, int):
                            raise TypeError(f'Shape specification {shape} containts a string '
                                            f'{element} but that component option does not '
                                            'provide an integer value: {dim_size}')
                        interp_shape.append(dim_size)
                    else:
                        interp_shape.append(element)
            else:
                kwargs['shape'] = shape
        if var_spec.units is not None:
            kwargs['units'] = var_spec.units
        if var_spec.desc:
            kwargs['desc'] = var_spec.desc
        if var_spec.tags:
            kwargs['tags'] = var_spec.tags
        if var_spec.shape_by_conn:
            kwargs['shape_by_conn'] = var_spec.shape_by_conn
        if var_spec.copy_shape is not None:
            kwargs['copy_shape'] = var_spec.copy_shape
        if var_spec.compute_shape is not None:
            kwargs['compute_shape'] = var_spec.compute_shape
        if var_spec.units_by_conn:
            kwargs['units_by_conn'] = var_spec.units_by_conn
        if var_spec.copy_units is not None:
            kwargs['copy_units'] = var_spec.copy_units
        if var_spec.compute_units is not None:
            kwargs['compute_units'] = var_spec.compute_units
        if var_spec.distributed:
            kwargs['distributed'] = var_spec.distributed
        if var_spec.primal_name is not None:
            kwargs['primal_name'] = var_spec.primal_name

        # Add iotype-specific parameters
        if iotype == 'input':
            # Input-only parameters
            if var_spec.require_conection:  # Note: typo in VariableSpec field name
                kwargs['require_connection'] = var_spec.require_conection

        elif iotype == 'output':
            # Output-only parameters
            if var_spec.lower is not None:
                kwargs['lower'] = var_spec.lower
            if var_spec.upper is not None:
                kwargs['upper'] = var_spec.upper
            if var_spec.ref is not None:
                kwargs['ref'] = var_spec.ref
            if var_spec.ref0 is not None:
                kwargs['ref0'] = var_spec.ref0
            if var_spec.adder is not None:
                kwargs['adder'] = var_spec.adder
            if var_spec.scaler is not None:
                kwargs['scaler'] = var_spec.scaler

        return kwargs


@register_system_spec
class OMExplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    system_type: str = 'OMExplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )

    @field_validator('assembled_jac_type', mode='before')
    @classmethod
    def check_explicit_component_assembled_jac_type(cls, v):
        """
        Validate that explicit components don't use assembled_jac_type.

        Only ImplicitComponents and Groups can use assembled_jac_type.
        """
        if v is not None:
            raise ValueError(
                "assembled_jac_type is not valid for ExplicitComponents. "
                "This option only applies to Groups and ImplicitComponents."
            )
        return v

    @field_validator('options', mode='before')
    @classmethod
    def check_explicit_component_options(cls, v):
        """
        Validate that explicit components don't use implicit-only options.

        This is the equivalent of options.undeclare in OpenMDAO 3.x models.
        """
        if v is not None and hasattr(v, 'assembled_jac_type') and v.assembled_jac_type is not None:
            raise ValueError(
                "assembled_jac_type is not valid for ExplicitComponents. "
                "This option only applies to Groups and ImplicitComponents."
            )
        return v


@register_system_spec
class OMImplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    system_type: str = 'OMImplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )


@register_system_spec
class OMJaxExplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    system_type: str = 'JaxExplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )


@register_system_spec
class OMJaxImplicitComponentSpec(ComponentSpec[ComponentOptionsSpec]):

    system_type: str = 'OMJaxImplicitComponent'

    path: str = Field(
        ...,
        description='The dotted path to the component class definition.'
    )
