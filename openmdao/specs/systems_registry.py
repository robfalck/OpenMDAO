from pydantic import BaseModel


# Global registry for all system spec types
_SYSTEM_SPEC_REGISTRY: dict[str, type[BaseModel]] = {}


def register_system_spec(spec_class):
    """
    Register a system spec type.
    
    Extracts the type name from the class's 'type' field default value.
    
    Examples
    --------
    @register_system_spec
    class MyCustomSpec(ComponentSpec):
        type: str = "mycustom"
    """
    # Get the type name from the type field's default
    type_field = spec_class.model_fields.get('type')
    if type_field and hasattr(type_field, 'default'):
        type_value = type_field.default
    else:
        raise ValueError(
            f"{spec_class.__name__} must have a 'type' field with a default value"
        )
    
    _SYSTEM_SPEC_REGISTRY[type_value] = spec_class
    return spec_class