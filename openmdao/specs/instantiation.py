from typing import cast, TYPE_CHECKING
from openmdao.specs.system_spec import SystemSpec
from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.group_spec import GroupSpec

if TYPE_CHECKING:
    from openmdao.specs.exec_comp_spec import ExecCompSpec


# Solver registry mapping solver_type strings to module paths
_SOLVER_REGISTRY = {
    # Linear solvers
    'DirectSolver': 'openmdao.solvers.linear.direct.DirectSolver',
    'LinearBlockGS': 'openmdao.solvers.linear.linear_block_gs.LinearBlockGS',
    'LinearBlockJac': 'openmdao.solvers.linear.linear_block_jac.LinearBlockJac',
    'LinearRunOnce': 'openmdao.solvers.linear.linear_runonce.LinearRunOnce',
    'ScipyKrylov': 'openmdao.solvers.linear.scipy_iter_solver.ScipyKrylov',
    'PETScKrylov': 'openmdao.solvers.linear.petsc_ksp.PETScKrylov',

    # Nonlinear solvers
    'NewtonSolver': 'openmdao.solvers.nonlinear.newton.NewtonSolver',
    'BroydenSolver': 'openmdao.solvers.nonlinear.broyden.BroydenSolver',
    'NonlinearBlockGS': 'openmdao.solvers.nonlinear.nonlinear_block_gs.NonlinearBlockGS',
    'NonlinearBlockJac': 'openmdao.solvers.nonlinear.nonlinear_block_jac.NonlinearBlockJac',
    'NonlinearRunOnce': 'openmdao.solvers.nonlinear.nonlinear_runonce.NonlinearRunOnce',

    # Linesearch solvers
    'ArmijoGoldsteinLS': 'openmdao.solvers.linesearch.backtracking.ArmijoGoldsteinLS',
    'BoundsEnforceLS': 'openmdao.solvers.linesearch.backtracking.BoundsEnforceLS',
}


def _instantiate_solver(solver_spec):
    """
    Instantiate a solver from its spec.

    Parameters
    ----------
    solver_spec : NonlinearSolverSpec or LinearSolverSpec or LinesearchSolverSpec
        Solver specification

    Returns
    -------
    Solver instance
    """
    import importlib

    solver_path = _SOLVER_REGISTRY.get(solver_spec.solver_type)
    if solver_path is None:
        raise ValueError(f"Unknown solver type: {solver_spec.solver_type}")

    module_path, class_name = solver_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    solver_class = getattr(module, class_name)

    # Instantiate solver
    solver = solver_class()

    # Set options
    for key, value in solver_spec.options.items():
        solver.options[key] = value

    # Handle nested linesearch (for nonlinear solvers)
    if hasattr(solver_spec, 'linesearch') and solver_spec.linesearch is not None:
        linesearch = _instantiate_solver(solver_spec.linesearch)
        solver.linesearch = linesearch

    return solver


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


def instantiate_from_spec(spec : SystemSpec | dict | str):
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

    # Apply system-level properties from SystemSpec
    _apply_system_spec_properties(comp, spec)

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
    for conn_spec in spec.connections:
        src_indices = None
        if conn_spec.src_indices.value is not None:
            src_indices = conn_spec.src_indices.value

        group.connect(conn_spec.src, conn_spec.tgt, src_indices=src_indices)

    # Set input defaults
    for indef_spec in spec.input_defaults:
        group.set_input_defaults(name=indef_spec.name, val=indef_spec.val, units=indef_spec.units,
                                 src_shape=indef_spec.src_shape)

    # Add advanced promotions via Group.promotes() method calls
    # Group by subsys_name to collect all promotes for each subsystem
    promotes_by_subsys = {}
    for pspec in spec.promotes:
        if pspec.subsys_name is not None:
            if pspec.subsys_name not in promotes_by_subsys:
                promotes_by_subsys[pspec.subsys_name] = []
            promotes_by_subsys[pspec.subsys_name].append(pspec)

    for subsys_name, promotes_specs in promotes_by_subsys.items():
        _apply_promotes_call(group, subsys_name, promotes_specs)

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


def _apply_promotes_call(group, subsys_name, promotes_specs):
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
