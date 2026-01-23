from typing import cast, TYPE_CHECKING
from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.group_spec import GroupSpec

if TYPE_CHECKING:
    from openmdao.specs.exec_comp_spec import ExecCompSpec


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
        print('adding subsystem')
        print(subsys_spec)
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
        print('adding connection')
        print(conn_spec)
        src_indices = None
        if conn_spec.src_indices.value is not None:
            src_indices = conn_spec.src_indices.value
        
        group.connect(conn_spec.src, conn_spec.tgt, src_indices=src_indices)

    # Set input defaults
    for indef_spec in spec.input_defaults:
        print('setting input defaults')
        print(indef_spec)
        group.set_input_defaults(name=indef_spec.name, val=indef_spec.val, units=indef_spec.units,
                                 src_shape=indef_spec.src_shape)
    
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


def to_spec(system):
    """
    Convert a concrete OpenMDAO system to a spec.

    Parameters
    ----------
    system : System
        An OpenMDAO Component or Group instance

    Returns
    -------
    ComponentSpec | GroupSpec
        The corresponding spec object

    Raises
    ------
    ValueError
        If the system type is not supported for spec conversion
    """
    from openmdao.core.group import Group

    if isinstance(system, Group):
        return _group_to_spec(system)
    else:
        return _component_to_spec(system)


def _component_to_spec(component) -> ComponentSpec:
    """
    Convert a component instance to its spec.

    Parameters
    ----------
    component : Component
        An OpenMDAO component instance

    Returns
    -------
    ComponentSpec
        The corresponding spec
    """
    from openmdao.specs.systems_registry import _COMPONENT_TO_SPEC_REGISTRY

    comp_type = type(component)

    # Look up spec class in registry
    spec_class = _COMPONENT_TO_SPEC_REGISTRY.get(comp_type)
    if spec_class is None:
        raise ValueError(
            f"No spec registered for component type {comp_type.__name__}. "
            f"Registered types: {list(_COMPONENT_TO_SPEC_REGISTRY.keys())}"
        )

    # Special handling for ExecComp
    from openmdao.api import ExecComp
    if isinstance(component, ExecComp):
        return _execcomp_to_spec(component)

    # Generic component conversion
    path = f"{comp_type.__module__}.{comp_type.__qualname__}"
    inputs = _extract_variable_specs(component, 'input')
    outputs = _extract_variable_specs(component, 'output')

    return spec_class(path=path, inputs=inputs, outputs=outputs)


def _execcomp_to_spec(exec_comp) -> 'ExecCompSpec':
    """
    Handle ExecComp specially to extract expressions.

    Parameters
    ----------
    exec_comp : ExecComp
        An ExecComp instance

    Returns
    -------
    ExecCompSpec
        The corresponding spec
    """
    from openmdao.specs.exec_comp_spec import ExecCompSpec

    # Extract expressions
    exprs = exec_comp._exprs if hasattr(exec_comp, '_exprs') else []

    # Extract variable metadata
    inputs = _extract_variable_specs(exec_comp, 'input')
    outputs = _extract_variable_specs(exec_comp, 'output')

    return ExecCompSpec(exprs=exprs, inputs=inputs, outputs=outputs)


def _extract_variable_specs(component, io_type) -> list:
    """
    Extract VariableSpecs from component metadata.

    Parameters
    ----------
    component : Component
        An OpenMDAO component
    io_type : str
        Either 'input' or 'output'

    Returns
    -------
    list[VariableSpec]
        List of VariableSpec objects
    """
    from openmdao.specs.variable_spec import VariableSpec

    var_specs = []

    # Get the absolute-to-metadata dict
    abs_names = component._var_allprocs_abs2meta.get(io_type, {})

    # Get the relative variable names for this component
    if not hasattr(component, '_var_rel_names'):
        return var_specs

    rel_names = component._var_rel_names.get(io_type, [])

    # For each relative name, find the absolute name and extract metadata
    for rel_name in rel_names:
        # Look up metadata
        rel_meta = {}

        # Try _var_rel2meta first (flat structure with all variable names)
        if hasattr(component, '_var_rel2meta') and rel_name in component._var_rel2meta:
            rel_meta = component._var_rel2meta.get(rel_name, {})
        else:
            # Otherwise construct absolute name and look it up in _var_allprocs_abs2meta
            abs_name = f"{component.pathname}.{rel_name}"

            if abs_name not in abs_names:
                # Try without pathname if it's the root model
                if component.pathname == '':
                    abs_name = rel_name

            if abs_name in abs_names:
                rel_meta = abs_names[abs_name]

        # Create VariableSpec from metadata
        var_spec = VariableSpec(
            name=rel_name,
            shape=rel_meta.get('shape'),
            units=rel_meta.get('units'),
            desc=rel_meta.get('desc', ''),
            tags=rel_meta.get('tags', [])
        )
        var_specs.append(var_spec)

    return var_specs


def _group_to_spec(group) -> 'GroupSpec':
    """
    Convert a Group instance to GroupSpec.

    Parameters
    ----------
    group : Group
        An OpenMDAO Group instance

    Returns
    -------
    GroupSpec
        The corresponding spec
    """
    from openmdao.specs.group_spec import GroupSpec
    from openmdao.specs.subsystem_spec import SubsystemSpec

    subsystems = []

    # Extract subsystems
    # Try _subsystems_allprocs first (dict of subsystems)
    subsystems_dict = None
    if hasattr(group, '_subsystems_allprocs') and isinstance(group._subsystems_allprocs, dict):
        subsystems_dict = group._subsystems_allprocs
    elif hasattr(group, '_subsystems_myproc') and isinstance(group._subsystems_myproc, dict):
        subsystems_dict = group._subsystems_myproc

    if subsystems_dict:
        for name, sysinfo in subsystems_dict.items():
            # Get the actual system object
            # _SysInfo objects have a reference to the system
            if hasattr(sysinfo, 'system'):
                subsys = sysinfo.system
            else:
                subsys = sysinfo

            # Recursively convert subsystem
            subsys_spec = to_spec(subsys)

            # Determine promotions from metadata
            promotes_inputs = _extract_promotions(group, name, 'input')
            promotes_outputs = _extract_promotions(group, name, 'output')

            subsystems.append(SubsystemSpec(
                name=name,
                system=subsys_spec,
                promotes_inputs=promotes_inputs,
                promotes_outputs=promotes_outputs
            ))

    # Extract connections
    connections = _extract_connections(group)

    return GroupSpec(
        subsystems=subsystems,
        connections=connections
    )


def _extract_promotions(group, subsys_name, io_type) -> list[str]:
    """
    Extract promotion list for a subsystem.

    Parameters
    ----------
    group : Group
        The parent group
    subsys_name : str
        Name of the subsystem
    io_type : str
        Either 'input' or 'output'

    Returns
    -------
    list[str]
        List of promoted variable names
    """
    promoted_names = []

    if not hasattr(group, '_var_allprocs_abs2prom'):
        return promoted_names

    abs2prom = group._var_allprocs_abs2prom[io_type]

    for abs_name, prom_name in abs2prom.items():
        # Check if this absolute name belongs to the subsystem
        # Format: "groupname.subsysname.varname"
        if abs_name.startswith(f"{group.name}.{subsys_name}."):
            # Extract the original name from prom_name
            promoted_names.append(prom_name)

    return promoted_names


def _extract_connections(group) -> list:
    """
    Extract connections from a group.

    Parameters
    ----------
    group : Group
        The group

    Returns
    -------
    list[ConnectionSpec]
        List of connection specs
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []

    # Extract from the connections dict
    if hasattr(group, '_conn_abs_in2out'):
        for tgt_abs, src_abs in group._conn_abs_in2out.items():
            # Convert absolute names to relative names within the group
            src_rel = _make_relative(src_abs, group.name)
            tgt_rel = _make_relative(tgt_abs, group.name)

            connections.append(ConnectionSpec(src=src_rel, tgt=tgt_rel))

    return connections


def _make_relative(abs_name, group_name) -> str:
    """
    Convert an absolute name to relative within a group.

    Parameters
    ----------
    abs_name : str
        Absolute name like "group.subsys.var"
    group_name : str
        The group name

    Returns
    -------
    str
        Relative name like "subsys.var"
    """
    if group_name and abs_name.startswith(f"{group_name}."):
        return abs_name[len(group_name) + 1:]
    return abs_name
