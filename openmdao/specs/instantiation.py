from typing import cast
from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.group_spec import GroupSpec


def instantiate_from_spec(spec : ComponentSpec | GroupSpec | dict | str):
    """
    Instantiate an OpenMDAO system from a specification.
    
    Parameters
    ----------
    spec : ComponentSpec, GroupSpec, dict, or str
        System specification. Can be a spec object, dict, or path to JSON file.
    
    Returns
    -------
    System
        Instantiated OpenMDAO component or group
    """
    from pathlib import Path
    from openmdao.specs.systems_registry import _SYSTEM_SPEC_REGISTRY
    
    # Convert file to spec
    if isinstance(spec, (str, Path)):
        spec = _load_spec_from_file(spec)
    
    # Convert dict to spec
    if isinstance(spec, dict):
        spec_type = spec.get('type')
        if spec_type is None:
            raise ValueError("Spec dict must have a 'type' field")
        
        spec_class = _SYSTEM_SPEC_REGISTRY.get(spec_type)
        if spec_class is None:
            raise ValueError(
                f"Unknown spec type: {spec_type}. "
                f"Known types: {list(_SYSTEM_SPEC_REGISTRY.keys())}"
            )
        
        spec = cast(ComponentSpec | GroupSpec, spec_class.model_validate(spec))
    
    # Handle different spec types
    if isinstance(spec, GroupSpec):
        return _instantiate_group(spec)
    else:
        return _instantiate_component(spec)


def _instantiate_component(spec):
    """Instantiate a component from spec."""
    import importlib
    
    # Determine which class to instantiate
    if hasattr(spec, 'path') and spec.path:
        module_path, class_name = spec.path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
    else:
        raise ValueError(
            f"Spec type '{spec.type}' requires a 'path' field to instantiate"
        )
    
    # Build init kwargs - check if spec provides a custom method
    if hasattr(spec, 'to_init_kwargs'):
        result = spec.to_init_kwargs()
        # Handle both tuple (args, kwargs) and dict (kwargs only) returns
        if isinstance(result, tuple):
            init_args, init_kwargs = result
            comp = component_class(*init_args, **init_kwargs)
            return comp
        else:
            # Backward compatible: just kwargs
            init_kwargs = result
    elif hasattr(spec, 'options') and spec.options:
        options_dict = (spec.options if isinstance(spec.options, dict)
                       else spec.options.model_dump(exclude_defaults=True))
        init_kwargs = options_dict
    else:
        init_kwargs = {}

    # Instantiate the component
    comp = component_class(**init_kwargs)

    return comp


def _instantiate_group(spec):
    """Instantiate a group from spec."""
    from openmdao.api import Group
    
    # Build init kwargs
    if hasattr(spec, 'to_init_kwargs'):
        init_kwargs = spec.to_init_kwargs()
    elif hasattr(spec, 'options') and spec.options:
        options_dict = (spec.options if isinstance(spec.options, dict)
                       else spec.options.model_dump(exclude_defaults=True))
        init_kwargs = options_dict
    else:
        init_kwargs = {}
    
    group = Group(**init_kwargs)
    
    # Add subsystems recursively
    for subsys_spec in spec.subsystems:
        subsys = instantiate_from_spec(subsys_spec.system)
        group.add_subsystem(
            subsys_spec.name,
            subsys,
            promotes_inputs=subsys_spec.promotes_inputs,
            promotes_outputs=subsys_spec.promotes_outputs,
            promotes=subsys_spec.promotes,
            min_procs=subsys_spec.min_procs,
            max_procs=subsys_spec.max_procs,
            proc_weight=subsys_spec.proc_weight
        )
    
    # Add connections
    for conn_spec in spec.connections:
        src_indices = None
        if conn_spec.src_indices.value is not None:
            src_indices = conn_spec.src_indices.value
        
        group.connect(conn_spec.src, conn_spec.tgt, src_indices=src_indices)
    
    return group


def _load_spec_from_file(filepath):
    """Load a spec from a JSON file."""
    from pathlib import Path
    import json
    
    filepath = Path(filepath)
    
    try:
        with open(filepath) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"File '{filepath}' is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}"
        ) from e
    except FileNotFoundError:
        raise FileNotFoundError(f"Spec file not found: {filepath}") from None
