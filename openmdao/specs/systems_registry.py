from pydantic import BaseModel


# Global registry for all system spec types
_SYSTEM_SPEC_REGISTRY: dict[str, type[BaseModel]] = {}

# Global registry for reverse conversion: component class -> spec class
_COMPONENT_TO_SPEC_REGISTRY: dict[type, type[BaseModel]] = {}


def register_system_spec(spec_class):
    """
    Register a system spec type.

    Extracts the type name from the class's 'system_type' field default value.

    Examples
    --------
    @register_system_spec
    class MyCustomSpec(ComponentSpec):
        system_type: str = "mycustom"
    """
    # Get the type name from the system_type field's default
    type_field = spec_class.model_fields.get('system_type')
    if type_field and hasattr(type_field, 'default'):
        type_value = type_field.default
    else:
        raise ValueError(
            f"{spec_class.__name__} must have a 'system_type' field with a default value"
        )
    
    _SYSTEM_SPEC_REGISTRY[type_value] = spec_class
    return spec_class


def register_component_to_spec(component_class, spec_class):
    """
    Register a mapping from component class to spec class.

    This enables reverse conversion (component -> spec) for concrete OpenMDAO objects.

    Parameters
    ----------
    component_class : type
        The component class (e.g., ExecComp, SellarDis1withDerivatives)
    spec_class : type
        The corresponding spec class (e.g., ExecCompSpec, OMExplicitComponentSpec)

    Examples
    --------
    from openmdao.api import ExecComp
    from openmdao.specs import ExecCompSpec

    register_component_to_spec(ExecComp, ExecCompSpec)
    """
    _COMPONENT_TO_SPEC_REGISTRY[component_class] = spec_class
    return spec_class