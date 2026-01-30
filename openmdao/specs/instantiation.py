from typing import cast

from openmdao.specs.system_spec import SystemSpec
from openmdao.specs.group_spec import GroupSpec
from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.connection_spec import ConnectionSpec
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.specs.exec_comp_spec import ExecCompSpec
from openmdao.specs.promotes_spec import PromotesSpec
from openmdao.specs.solver_registry import _SOLVER_SPEC_REGISTRY, _SOLVER_REGISTRY


def _instantiate_solver(solver_spec):
    """
    Instantiate a solver from its spec.

    Parameters
    ----------
    solver_spec : NonlinearSolverSpec or LinearSolverSpec or LinesearchSolverSpec or dict
        Solver specification or dict to be converted to spec

    Returns
    -------
    Solver instance
    """
    import importlib

    # Convert dict to spec if needed - use registry to find correct spec class
    if isinstance(solver_spec, dict):
        solver_type = solver_spec.get('solver_type')
        if solver_type is None:
            raise ValueError("Solver spec dict must have a 'solver_type' field")

        # Look up specific spec class from registry
        spec_class = _SOLVER_SPEC_REGISTRY.get(solver_type)
        if spec_class is None:
            raise ValueError(f"Unknown solver type: {solver_type}")

        # Validate using the specific spec class
        solver_spec = spec_class.model_validate(solver_spec)

    solver_path = _SOLVER_REGISTRY.get(solver_spec.solver_type)
    if solver_path is None:
        raise ValueError(f"Unknown solver type: {solver_spec.solver_type}")

    module_path, class_name = solver_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    solver_class = getattr(module, class_name)

    # Instantiate solver
    solver = solver_class()

    # Set options - handle both dict and options spec objects
    if hasattr(solver_spec, 'options'):
        options = solver_spec.options
        if isinstance(options, dict):
            # Direct dict assignment
            for key, value in options.items():
                solver.options[key] = value
        else:
            # Options spec object - extract via model_dump
            options_dict = options.model_dump(exclude_defaults=False)
            for key, value in options_dict.items():
                solver.options[key] = value

    # Handle nested linesearch (for nonlinear solvers)
    if hasattr(solver_spec, 'linesearch') and solver_spec.linesearch is not None:
        linesearch = _instantiate_solver(solver_spec.linesearch)
        solver.linesearch = linesearch

    return solver


def _wrap_setup_method(comp, spec):
    """
    Replace component's setup method with a spec-driven wrapper.

    The wrapper bypasses the component's original setup() method and delegates
    all configuration to spec.setup(). The spec is responsible for configuring
    the component appropriately, including calling the component's original
    setup() if needed (e.g., ExecComp for expression parsing).

    Parameters
    ----------
    comp : Component
        The component instance to modify
    spec : ComponentSpec
        The spec that will configure the component
    """
    from types import MethodType

    def wrapper_setup(self):
        """
        Wrap setup method with one provided by the spec..

        The spec is responsible for configuring the component.
        """
        # Let the spec configure the component
        if hasattr(spec, 'setup') and callable(spec.setup):
            spec.setup(self)

    # Bind the wrapper to the component instance
    comp.setup = MethodType(wrapper_setup, comp)


def _apply_system_spec_properties(system, spec):
    """
    Apply system-level properties from SystemSpec to an instantiated system.

    This handles properties common to all systems (Components and Groups):
    - Solvers (nonlinear_solver, linear_solver)
    - Design variables (design_vars)
    - Constraints (constraints)
    - Objectives (objective)
    - Options (assembled_jac_type, derivs_method)
    - Recording options (recording_options)

    Parameters
    ----------
    system : System
        Instantiated OpenMDAO system (Component or Group)
    spec : SystemSpec
        System specification (ComponentSpec or GroupSpec)
    """
    # Set nonlinear solver
    if spec.nonlinear_solver is not None:
        system.nonlinear_solver = _instantiate_solver(spec.nonlinear_solver)

    # Set linear solver
    if spec.linear_solver is not None:
        system.linear_solver = _instantiate_solver(spec.linear_solver)

    # Set assembled_jac_type option
    if spec.assembled_jac_type is not None:
        system.options['assembled_jac_type'] = spec.assembled_jac_type

    # Set derivs_method option
    if spec.derivs_method is not None:
        system.options['derivs_method'] = spec.derivs_method

    # Add design variables
    for dv_spec in spec.design_vars:
        system.add_design_var(
            dv_spec.name,
            lower=dv_spec.lower,
            upper=dv_spec.upper,
            ref=dv_spec.ref,
            ref0=dv_spec.ref0,
            indices=dv_spec.indices.value if dv_spec.indices.value is not None else None,
            adder=dv_spec.adder,
            scaler=dv_spec.scaler,
            units=dv_spec.units,
            parallel_deriv_color=dv_spec.parallel_deriv_color,
            cache_linear_solution=dv_spec.cache_linear_solution,
            flat_indices=dv_spec.flat_indices
        )

    # Add constraints
    for con_spec in spec.constraints:
        system.add_constraint(
            con_spec.name,
            lower=con_spec.lower,
            upper=con_spec.upper,
            equals=con_spec.equals,
            ref=con_spec.ref,
            ref0=con_spec.ref0,
            adder=con_spec.adder,
            scaler=con_spec.scaler,
            indices=con_spec.indices.value if con_spec.indices.value is not None else None,
            linear=con_spec.linear,
            parallel_deriv_color=con_spec.parallel_deriv_color,
            cache_linear_solution=con_spec.cache_linear_solution,
            flat_indices=con_spec.flat_indices,
            alias=con_spec.alias
        )

    # Add objectives
    for obj_spec in spec.objective:
        system.add_objective(
            obj_spec.name,
            ref=obj_spec.ref,
            ref0=obj_spec.ref0,
            index=obj_spec.index,
            adder=obj_spec.adder,
            scaler=obj_spec.scaler,
            parallel_deriv_color=obj_spec.parallel_deriv_color,
            cache_linear_solution=obj_spec.cache_linear_solution,
            alias=obj_spec.alias
        )

    # Configure recording options
    if spec.recording_options is not None:
        system.recording_options['record_inputs'] = spec.recording_options.record_inputs
        system.recording_options['record_outputs'] = spec.recording_options.record_outputs
        system.recording_options['record_residuals'] = spec.recording_options.record_residuals
        system.recording_options['includes'] = spec.recording_options.includes
        system.recording_options['excludes'] = spec.recording_options.excludes
        system.recording_options['options_excludes'] = spec.recording_options.options_excludes


def instantiate_from_spec(spec : SystemSpec | dict | str) -> Group | Component:
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
    from openmdao.specs.group_spec import GroupSpec
    from openmdao.specs.systems_registry import _SYSTEM_SPEC_REGISTRY

    # Convert file to spec
    if isinstance(spec, (str, Path)):
        spec = _load_spec_from_file(spec)

    # Convert dict to spec
    if isinstance(spec, dict):
        spec_type = spec.get('system_type')
        if spec_type is None:
            raise ValueError("Spec dict must have a 'system_type' field")

        spec_class = _SYSTEM_SPEC_REGISTRY.get(spec_type)
        if spec_class is None:
            raise ValueError(
                f"Unknown spec type: {spec_type}. "
                f"Known types: {list(_SYSTEM_SPEC_REGISTRY.keys())}"
            )

        spec = cast(SystemSpec, spec_class.model_validate(spec))

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
            f"Spec type '{spec.system_type}' requires a 'path' field to instantiate"
        )

    # Build init kwargs - check if spec provides a custom method
    if hasattr(spec, 'to_init_kwargs'):
        result = spec.to_init_kwargs()
        # Handle both tuple (args, kwargs) and dict (kwargs only) returns
        if isinstance(result, tuple):
            init_args, init_kwargs = result
            comp = component_class(*init_args, **init_kwargs)
        else:
            # Backward compatible: just kwargs
            init_kwargs = result
            comp = component_class(**init_kwargs)
    elif hasattr(spec, 'options') and spec.options:
        options_dict = (spec.options if isinstance(spec.options, dict)
                       else spec.options.model_dump(exclude_defaults=True))
        init_kwargs = options_dict
        comp = component_class(**init_kwargs)
    else:
        init_kwargs = {}
        comp = component_class(**init_kwargs)

    # Wrap the setup method to use spec configuration
    _wrap_setup_method(comp, spec)

    # Apply system-level properties from SystemSpec
    _apply_system_spec_properties(comp, spec)

    return comp


def _instantiate_group(spec):
    """
    Instantiate a group from spec.

    The group's setup() method is wrapped so that the spec configures
    subsystems, connections, and promotions during setup rather than
    during instantiation. This matches the pattern used for components.
    """
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

    # Wrap the setup method to use spec configuration
    _wrap_setup_method(group, spec)

    # Apply system-level properties from SystemSpec
    _apply_system_spec_properties(group, spec)

    return group


def _extract_promote_names(promotes_specs):
    """
    Convert list of PromotesSpec objects to simple names/tuples for add_subsystem().

    Parameters
    ----------
    promotes_specs : list[PromotesSpec] or None
        List of PromotesSpec objects

    Returns
    -------
    list or None
        List of strings or tuples (old_name, new_name) suitable for add_subsystem()
    """
    if not promotes_specs:
        return None

    result = []
    for pspec in promotes_specs:
        result.append(pspec.name)
    return result if result else None


def _apply_promotes_call(group : Group | ParallelGroup, subsys_name : str,
                         promotes_specs : PromotesSpec):
    """
    Apply a set of PromotesSpec objects to a group via Group.promotes().

    Groups promotes by io_type and calls Group.promotes() appropriately.

    Parameters
    ----------
    group : Group
        The group to apply promotions to
    subsys_name : str
        Name of the subsystem to promote from
    promotes_specs : list[PromotesSpec]
        List of PromotesSpec objects to apply
    """
    # Group promotes by io_type
    any_promotes = []
    input_promotes = []
    output_promotes = []

    for pspec in promotes_specs:
        if pspec.io_type == 'any':
            any_promotes.append(pspec)
        elif pspec.io_type == 'input':
            input_promotes.append(pspec)
        elif pspec.io_type == 'output':
            output_promotes.append(pspec)

    # Apply promotes calls with appropriate parameters
    if any_promotes:
        _do_promotes_call(group, subsys_name, any_promotes, io_type='any')

    if input_promotes:
        _do_promotes_call(group, subsys_name, input_promotes, io_type='input')

    if output_promotes:
        _do_promotes_call(group, subsys_name, output_promotes, io_type='output')


def _do_promotes_call(group, subsys_name, promotes_specs, io_type):
    """
    Make a single Group.promotes() call for a specific io_type.

    Parameters
    ----------
    group : Group
        The group to apply promotions to
    subsys_name : str
        Name of the subsystem to promote from
    promotes_specs : list[PromotesSpec]
        List of PromotesSpec objects with the same io_type
    io_type : {'any', 'input', 'output'}
        The type of promotion
    """
    # Extract simple names/tuples (the variable names and renames)
    names = [pspec.name for pspec in promotes_specs]

    # Build kwargs for group.promotes() call
    promotes_kwargs = {}

    if io_type == 'any':
        promotes_kwargs['any'] = names
    elif io_type == 'input':
        promotes_kwargs['inputs'] = names
    elif io_type == 'output':
        promotes_kwargs['outputs'] = names

    # Handle src_indices and src_shape
    # Check if all specs have the same src_indices and src_shape (simple case)
    # Otherwise, we need to call promotes() multiple times or handle per-variable
    all_same_src_indices = all(pspec.src_indices == promotes_specs[0].src_indices
                                for pspec in promotes_specs)
    all_same_src_shape = all(pspec.src_shape == promotes_specs[0].src_shape
                              for pspec in promotes_specs)

    if all_same_src_indices and promotes_specs[0].src_indices is not None:
        src_indices_value = promotes_specs[0].src_indices.value
        if src_indices_value is not None:
            promotes_kwargs['src_indices'] = src_indices_value

    if all_same_src_shape and promotes_specs[0].src_shape is not None:
        promotes_kwargs['src_shape'] = promotes_specs[0].src_shape

    # Make the promotes call
    group.promotes(subsys_name, **promotes_kwargs)


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


def _extract_promotions(group, subsys_name, io_type) -> list:
    """
    Extract promotion list for a subsystem as PromotesSpec objects.

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
    list[PromotesSpec]
        List of PromotesSpec objects representing promotions
    """
    from openmdao.specs.promotes_spec import PromotesSpec

    promotes_specs = []

    if not hasattr(group, '_var_allprocs_abs2prom'):
        return promotes_specs

    abs2prom = group._var_allprocs_abs2prom[io_type]

    for abs_name, prom_name in abs2prom.items():
        # Check if this absolute name belongs to the subsystem
        # Format: "groupname.subsysname.varname"
        prefix = f"{group.name}.{subsys_name}." if group.name else f"{subsys_name}."

        if abs_name.startswith(prefix):
            # Extract the original variable name from the absolute name
            # Format is "groupname.subsysname.varname" -> extract "varname"
            var_name = abs_name[len(prefix):]

            # Check if the variable was renamed during promotion
            if prom_name != var_name:
                # Variable was renamed
                name = (var_name, prom_name)
            else:
                # Simple promotion without renaming
                name = var_name

            # Create PromotesSpec with appropriate io_type
            pspec = PromotesSpec(name=name, io_type=io_type)
            promotes_specs.append(pspec)

    return promotes_specs


def _extract_connections(group) -> list[ConnectionSpec]:
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
    connections = []

    # Extract from the connections dict
    if hasattr(group, '_conn_abs_in2out'):
        for tgt_abs, src_abs in group._conn_abs_in2out.items():
            # Convert absolute names to relative names within the group
            src_rel = group._resolver.abs2rel(src_abs)
            tgt_rel = group._resolver.abs2rel(tgt_abs)

            connections.append(ConnectionSpec(src=src_rel, tgt=tgt_rel))

    return connections
