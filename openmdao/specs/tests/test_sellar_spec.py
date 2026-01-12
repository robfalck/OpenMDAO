"""
Create a pydantic spec (schema) for the OpenMDAO Sellar problem.

This script demonstrates how to use GroupSpec, ComponentSpec, and SubsystemSpec
to create a serializable specification of the Sellar MDA problem.

The spec can be serialized to JSON/YAML and used to reconstruct the problem later.
"""
import json
from openmdao.specs import GroupSpec, SubsystemSpec, VariableSpec, ExecCompSpec, OMExplicitComponentSpec, \
    instantiate_from_spec
from openmdao.specs.group_spec import NonlinearSolverSpec, LinearSolverSpec, LinesearchSolverSpec


def create_sellar_spec():
    """
    Create a GroupSpec for the Sellar problem.

    This replicates the SellarDerivatives group from openmdao.test_suite.components.sellar
    but as a pydantic specification that can be serialized.

    Returns
    -------
    GroupSpec
        Specification for the Sellar MDA group
    """

    # Define the two discipline components using existing OpenMDAO components via path
    d1_spec = OMExplicitComponentSpec(
        inputs=[VariableSpec(name='z', shape=(2,)),
                VariableSpec(name='x'),
                VariableSpec(name='y2')],
        outputs=[VariableSpec(name='y1')],
        path='openmdao.test_suite.components.sellar.SellarDis1withDerivatives'
    )

    d2_spec = OMExplicitComponentSpec(
        inputs=[VariableSpec(name='z', shape=(2,)),
                VariableSpec(name='y1')],
        outputs=[VariableSpec(name='y2')],
        path='openmdao.test_suite.components.sellar.SellarDis2withDerivatives'
    )

    response_spec = ExecCompSpec(
        exprs=[
            "obj = x**2 + z[1] + y1 + exp(-y2)",
            "con1 = 3.16 - y1",
            "con2 = y2 - 24.0"
        ],
        inputs=[
            VariableSpec(
                name="x",
                desc="Design variable x"
            ),
            VariableSpec(
                name="z",
                shape=(2,),
                desc="Design variable z (2-element array)"
            ),
            VariableSpec(
                name="y1",
                desc="Coupling variable y1 from discipline 1"
            ),
            VariableSpec(
                name="y2",
                desc="Coupling variable y2 from discipline 2"
            ),
        ],
        outputs=[
            VariableSpec(
                name="obj",
                desc="Objective function value"
            ),
            VariableSpec(
                name="con1",
                desc="Constraint: 3.16 - y1 >= 0",
                upper=0.0  # For inequality constraint con1 <= 0
            ),
            VariableSpec(
                name="con2",
                desc="Constraint: y2 - 24.0 <= 0",
                upper=0.0  # For inequality constraint con2 <= 0
            ),
        ]
    )

    # Create the group spec with promoted subsystems
    # This follows the structure of SellarDerivatives from sellar.py
    sellar_spec = GroupSpec(
        subsystems=[
            SubsystemSpec(
                name='d1',
                system=d1_spec,
                promotes_inputs=['x', 'z', 'y2'],
                promotes_outputs=['y1']
            ),
            SubsystemSpec(
                name='d2',
                system=d2_spec,
                promotes_inputs=['z', 'y1'],
                promotes_outputs=['y2']
            ),
            SubsystemSpec(
                name='obj_cmp',
                system=response_spec,
                promotes_inputs=['x', 'z', 'y1', 'y2'],
                promotes_outputs=['obj', 'con1', 'con2']
            )
        ],
        connections=[
            # Note: In this version with promotes, d1 and d2 are connected via promoted y1 and y2
            # No explicit connections needed
        ],
        # Solver specs - Sellar is a coupled problem that requires a nonlinear solver
        nonlinear_solver=NonlinearSolverSpec(
            solver_type='NewtonSolver',
            options={'maxiter': 10, 'atol': 1e-10, 'rtol': 1e-10}
        ),
        linear_solver=LinearSolverSpec(
            solver_type='DirectSolver',
            options={}
        )
    )

    return sellar_spec


def create_sellar_spec_with_connections():
    """
    Create a GroupSpec for the Sellar problem using explicit connections instead of promotes.

    This version replicates SellarDerivativesConnected from sellar.py.

    Returns
    -------
    GroupSpec
        Specification for the Sellar MDA group with explicit connections
    """
    # Define the two discipline components
    d1_spec = OMExplicitComponentSpec(
        inputs=[VariableSpec(name='z', shape=(2,), units=None),
                VariableSpec(name='x', units=None),
                VariableSpec(name='y2', units=None)],
        outputs=[VariableSpec(name='y1', lower=0.1, upper=1000, units=None, ref=1.0)],
        path='openmdao.test_suite.components.sellar.SellarDis1withDerivatives'
    )

    d2_spec = OMExplicitComponentSpec(
        inputs=[VariableSpec(name='z', shape=(2,), units=None),
                VariableSpec(name='y1', units=None),],
        outputs=[VariableSpec(name='y2', lower=0.1, upper=1000., ref=1.0, units=None)],
        path='openmdao.test_suite.components.sellar.SellarDis2withDerivatives'
    )

    response_spec = ExecCompSpec(
        exprs=[
            "obj = x**2 + z[1] + y1 + exp(-y2)",
            "con1 = 3.16 - y1",
            "con2 = y2 - 24.0"
        ],
        inputs=[
            VariableSpec(
                name="x",
                desc="Design variable x"
            ),
            VariableSpec(
                name="z",
                shape=(2,),
                desc="Design variable z (2-element array)"
            ),
            VariableSpec(
                name="y1",
                desc="Coupling variable y1 from discipline 1"
            ),
            VariableSpec(
                name="y2",
                desc="Coupling variable y2 from discipline 2"
            ),
        ],
        outputs=[
            VariableSpec(
                name="obj",
                desc="Objective function value"
            ),
            VariableSpec(
                name="con1",
                desc="Constraint: 3.16 - y1 >= 0",
                upper=0.0  # For inequality constraint con1 <= 0
            ),
            VariableSpec(
                name="con2",
                desc="Constraint: y2 - 24.0 <= 0",
                upper=0.0  # For inequality constraint con2 <= 0
            ),
        ]
    )

    # Create the group spec with explicit connections
    from openmdao.specs import ConnectionSpec

    sellar_spec = GroupSpec(
        subsystems=[
            SubsystemSpec(
                name='d1',
                system=d1_spec,
                promotes_inputs=['x', 'z']
            ),
            SubsystemSpec(
                name='d2',
                system=d2_spec,
                promotes_inputs=['z']
            ),
            SubsystemSpec(
                name='responses',
                system=response_spec,
                promotes_inputs=['x', 'z'],
                promotes_outputs=['obj', 'con1', 'con2']
            ),
        ],
        connections=[
            # Connect discipline outputs to each other and to constraints/objective
            ConnectionSpec(src='d1.y1', tgt='d2.y1'),
            ConnectionSpec(src='d1.y1', tgt='responses.y1'),
            # ConnectionSpec(src='d1.y1', tgt='con_cmp1.y1'),
            ConnectionSpec(src='d2.y2', tgt='d1.y2'),
            ConnectionSpec(src='d2.y2', tgt='responses.y2'),
            # ConnectionSpec(src='d2.y2', tgt='con_cmp2.y2'),
        ],
        # Solver specs with linesearch
        nonlinear_solver=NonlinearSolverSpec(
            solver_type='NewtonSolver',
            options={'maxiter': 10, 'atol': 1e-10},
            linesearch=LinesearchSolverSpec(
                solver_type='ArmijoGoldsteinLS',
                options={'maxiter': 3}
            )
        ),
        linear_solver=LinearSolverSpec(
            solver_type='ScipyKrylov',
            options={'maxiter': 100}
        )
    )

    return sellar_spec


def main():
    """
    Main function to demonstrate creating and serializing the Sellar spec.
    """
    print("=" * 70)
    print("Creating Sellar Problem Spec (Schema)")
    print("=" * 70)

    # Create the specification (promoted version)
    print("\n1. Creating GroupSpec for Sellar problem (with promotes)...")
    sellar_spec = create_sellar_spec()
    print(f"   - Created spec with {len(sellar_spec.subsystems)} subsystems")
    print(f"   - Type: {sellar_spec.type}")

    # Display subsystem names
    print("\n   Subsystems:")
    for subsys in sellar_spec.subsystems:
        if hasattr(subsys.system, 'path') and subsys.system.path:
            comp_class = subsys.system.path.split('.')[-1]
            print(f"     - {subsys.name}: {comp_class}")
            if subsys.promotes_inputs:
                print(f"       promotes_inputs: {subsys.promotes_inputs}")
            if subsys.promotes_outputs:
                print(f"       promotes_outputs: {subsys.promotes_outputs}")

    # Serialize to JSON
    print("\n2. Serializing spec to JSON...")
    spec_dict = sellar_spec.model_dump()
    spec_json = json.dumps(spec_dict, indent=2)
    print(f"   - Serialized to {len(spec_json)} characters")

    # Save to file
    output_file = 'sellar_spec.json'
    with open(output_file, 'w') as f:
        f.write(spec_json)
    print(f"   - Saved to {output_file}")

    # Demonstrate deserialization
    print("\n3. Deserializing spec from JSON...")
    restored_spec = GroupSpec.model_validate(spec_dict)
    print(f"   - Restored spec with {len(restored_spec.subsystems)} subsystems")
    print(f"   - Validation successful!")

    # Create the connected version
    print("\n4. Creating GroupSpec for Sellar problem (with connections)...")
    sellar_conn_spec = create_sellar_spec_with_connections()
    print(f"   - Created spec with {len(sellar_conn_spec.connections)} connections")

    print("\n   Connections:")
    for conn in sellar_conn_spec.connections:
        print(f"     - {conn.src} -> {conn.tgt}")

    # Serialize connected version
    # Note: Don't use exclude_defaults=True because we need the 'type' field
    # for proper deserialization of ComponentSpec subclasses
    conn_spec_dict = sellar_conn_spec.model_dump()
    conn_output_file = 'sellar_spec_connected.json'
    with open(conn_output_file, 'w') as f:
        json.dump(conn_spec_dict, f, indent=2)
    print(f"   - Saved to {conn_output_file}")


    with open(conn_output_file, 'r') as f:
        loaded_json = json.load(f)

    restored_sellar_conn_spec = GroupSpec.model_validate(loaded_json)

    print("\n5. Verifying deserialized component types...")
    for subsys in restored_sellar_conn_spec.subsystems:
        print(f"   - {subsys.name}: {type(subsys.system).__name__}")


    import openmdao.api as om

    p = om.Problem()
    p.model = instantiate_from_spec(restored_sellar_conn_spec)
    p.setup()
    

    # print("\n" + "=" * 70)
    # print("Missing Schemas / TODO Items")
    # print("=" * 70)
    # print("\nThe following schemas are referenced but not yet fully implemented:")
    # print("\n1. ComponentSpec - Init Arguments:")
    # print("   - ComponentSpec has 'path' to reference existing component classes")
    # print("   - But there's no way to specify __init__ arguments for those classes")
    # print("   - For example, ExecComp needs expression and variable initialization")
    # print("   - Possible solutions:")
    # print("     a) Add an 'init_kwargs' field to ComponentSpec")
    # print("     b) Create a separate ExecCompSpec with expression field")
    # print("     c) Use the ComponentSpec from group_spec.py (has init_kwargs)")
    # print("\n2. SolverSpec:")
    # print("   - Currently only has basic structure (solver_type, options)")
    # print("   - Need to fully specify solver types as enum:")
    # print("     * NewtonSolver, NonlinearBlockGS, NonlinearBlockJac")
    # print("     * DirectSolver, LinearBlockGS, LinearBlockJac, ScipyKrylov, PETScKrylov")
    # print("   - Need to specify valid options for each solver type")
    # print("   - Need to handle nested solvers (e.g., linesearch in Newton)")
    # print("\n3. ResponseSpec (exists in response_spec.py but not integrated):")
    # print("   - add_design_var specifications")
    # print("   - add_objective specifications")
    # print("   - add_constraint specifications")
    # print("   - Should be added to GroupSpec or a ProblemSpec")
    # print("\n4. Input Defaults:")
    # print("   - Need a way to specify set_input_defaults in the GroupSpec")
    # print("   - Could be added as a dict field in GroupSpec")
    # print("\n5. Recorder Specs:")
    # print("   - add_recorder specifications")
    # print("   - recording_options specifications")
    # print("\n6. Configuration:")
    # print("   - approx_totals settings")
    # print("   - set_check_partial_options")
    # print("=" * 70)

    # print("\n" + "=" * 70)
    # print("Summary")
    # print("=" * 70)
    # print(f"Created serializable specs for Sellar problem:")
    # print(f"  - {output_file} (promoted version)")
    # print(f"  - {conn_output_file} (connected version)")
    # print("\nThese specs can be:")
    # print("  - Serialized to JSON/YAML/other formats")
    # print("  - Stored in version control")
    # print("  - Used to regenerate the OpenMDAO model")
    # print("  - Validated against the pydantic schema")
    # print("=" * 70)


if __name__ == '__main__':
    main()
