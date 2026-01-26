from pydantic import BaseModel


# Global registry for all system spec types
_SOLVER_SPEC_REGISTRY: dict[str, type[BaseModel]] = {}

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


def register_solver_spec(spec_class):
    """
    Register a solver spec type.

    Extracts the type name from the class's 'solver_type' field default value.

    Examples
    --------
    @register_solver_spec
    class MyCustomSpec(NonlinearSolverSpec):
        solver_type: Literal['NewtonSolver'] = 'NewtonSolver'
    """
    # Get the type name from the solver_type field's default
    type_field = spec_class.model_fields.get('solver_type')
    if type_field and hasattr(type_field, 'default'):
        type_value = type_field.default
    else:
        raise ValueError(
            f"{spec_class.__name__} must have a 'solver_type' field with a default value"
        )

    _SOLVER_SPEC_REGISTRY[type_value] = spec_class
    return spec_class
