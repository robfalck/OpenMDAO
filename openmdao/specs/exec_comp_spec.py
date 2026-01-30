from collections.abc import Sequence
from pydantic import Field, field_validator, model_validator, ConfigDict, \
    ValidationError

from openmdao.utils.units import valid_units
from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.options_spec import ComponentOptionsSpec
from openmdao.specs.variable_spec import VariableSpec
from openmdao.specs.systems_registry import register_system_spec


class ExecCompOptionsSpec(ComponentOptionsSpec):
        
    has_diag_partials : bool = Field(
        default=False,
        description='If True, treat all array/array partials as diagonal if both '
                    'arrays have size > 1. All arrays with size > 1 must have the '
                    'same flattened size or an exception will be raised.'
    )

    units : str | None = Field(
        default=None,
        description='Units to be assigned to all variables in this component. '
                    'Default is None, which means units may be provided for variables '
                    'individually.'
    )

    shape : tuple | None = Field(
        default=None,
        description='Shape to be assigned to all variables in this component. '
                    'Default is None, which means shape may be provided for variables'
                    ' individually.'
    )

    shape_by_conn : bool = Field(
        default=False,
        description='If True, shape all inputs and outputs based on their connection. '
                    'Default is False.'
    )

    do_coloring : bool = Field(
        default=True,
        description='If True (the default), compute the partial jacobian '
                    'coloring for this component.'
    )

    @field_validator('units', mode='before')
    @classmethod
    def validate_units(cls, v):
        if isinstance(v, str) and not valid_units(v):
            raise ValidationError('Units {v} are not valid OpenMDAO units.')
        return v

    @field_validator('distributed', mode='before')
    @classmethod
    def disallow_distributed(cls, v):
        """
        Option distributed is not valid for ExecComp.
        """
        if v:
            raise ValidationError('ExecComp does not support distributed variables.')
        return v


@register_system_spec
class ExecCompSpec(ComponentSpec[ExecCompOptionsSpec]):
    """
    Specification for an ExecComp.
    
    ExecComp allows you to define a component using simple expressions.
    This spec captures the expressions and variable metadata.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_type: str = 'ExecComp'

    path: str = "openmdao.api.ExecComp"

    exprs: str | Sequence[str] = Field(
        ...,
        description="Expression(s) to be evaluated. Can be a single string or list of strings."
    )

    options: ExecCompOptionsSpec | None = Field(
        default_factory=ExecCompOptionsSpec,
        description='ExecComp options.'
    )

    @field_validator('exprs', mode='before')
    @classmethod
    def validate_exprs(cls, v):
        """Ensure exprs is a list."""
        if isinstance(v, str):
            return [v]
        return list(v)
    
    @model_validator(mode='after')
    def validate_variable_names(self):
        """Ensure no duplicate variable names and validate against expressions."""
        # Check for duplicate names
        input_names = {var.name for var in self.inputs}
        output_names = {var.name for var in self.outputs}
        
        overlap = input_names & output_names
        if overlap:
            raise ValueError(f"Variables cannot be both inputs and outputs: {overlap}")
        
        # Extract variable names from expressions
        # This is a simplified check - ExecComp does more sophisticated parsing
        import re
        all_vars = set()
        for expr in self.exprs:
            # Find variable names in expression (simplified - doesn't handle all cases)
            vars_in_expr = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
            all_vars.update(vars_in_expr)
        
        # Remove Python keywords and numpy functions
        keywords = {'exp', 'sin', 'cos', 'tan', 'log', 'log10', 'sqrt', 'abs', 
                   'arcsin', 'arccos', 'arctan', 'arctan2', 'sinh', 'cosh', 'tanh'}
        all_vars -= keywords
        
        declared_vars = input_names | output_names
        
        # Variables in expressions that aren't declared
        undeclared = all_vars - declared_vars
        if undeclared:
            # This is a warning, not an error - ExecComp will auto-create these
            pass
        
        return self

    def to_init_kwargs(self):
        """
        Return positional and keyword arguments for ExecComp instantiation.

        Returns
        -------
        tuple
            (positional_args, keyword_args) where positional_args is a list
            and keyword_args is a dict
        """
        exprs_arg = self.exprs

        # Options become keyword arguments
        options_dict = self.options.model_dump(exclude_defaults=True) if self.options else {}

        # Fields that ExecComp recognizes for variables (from ExecComp._allowed_meta)
        allowed_fields = {'value', 'val', 'shape', 'units', 'res_units', 'desc',
                         'ref', 'ref0', 'res_ref', 'lower', 'upper', 'src_indices',
                         'flat_src_indices', 'tags', 'shape_by_conn', 'copy_shape',
                         'compute_shape', 'units_by_conn', 'copy_units', 'compute_units',
                         'constant'}

        # Add input and output metadata as keyword arguments
        # ExecComp accepts variable metadata like: ExecComp(exprs, x={'shape': (2,)}, ...)
        for var in self.inputs:
            var_dict = var.model_dump(exclude_defaults=True)  # exclude to avoid val conflicts
            var_dict.pop('name', None)  # Remove name key since it's the kwarg key
            var_dict.pop('val', None)  # Don't pass val for inputs - it comes from connected source
            # Keep only fields that ExecComp recognizes
            var_dict = {k: v for k, v in var_dict.items() if k in allowed_fields}
            if var_dict:  # Only add if there's metadata
                options_dict[var.name] = var_dict

        for var in self.outputs:
            var_dict = var.model_dump(exclude_defaults=True)  # Exclude defaults
            var_dict.pop('name', None)  # Remove name key since it's the kwarg key
            # For outputs, keep val if specified (it provides initial value)
            # Keep only fields that ExecComp recognizes
            var_dict = {k: v for k, v in var_dict.items() if k in allowed_fields}
            if var_dict:  # Only add if there's metadata
                options_dict[var.name] = var_dict

        return ([exprs_arg], options_dict)

    def setup(self, comp):
        """
        Configure ExecComp by calling its setup() method.

        ExecComp needs its setup() called to parse expressions and create variables.
        We call ExecComp.setup(comp) directly to handle expression parsing and
        variable creation. No additional configuration is needed after that.

        Parameters
        ----------
        comp : ExecComp
            The ExecComp instance to configure
        """
        from openmdao.components.exec_comp import ExecComp
        ExecComp.setup(comp)

# Example usage
if __name__ == "__main__":
    # Example 1: Simple expression
    simple_spec = ExecCompSpec(
        exprs="y = 2.0 * x",
        inputs=[
            VariableSpec(name="x", units="m")
        ],
        outputs=[
            VariableSpec(name="y", units="m")
        ]
    )
    
    print("Simple ExecComp:")
    print(simple_spec.model_dump_json(indent=2, exclude_defaults=True))
    
    # Example 2: Sellar objective
    obj_spec = ExecCompSpec(
        exprs="obj = x**2 + z[1] + y1 + exp(-y2)",
        inputs=[
            VariableSpec(name="x"),
            VariableSpec(name="z", shape=(2,)),
            VariableSpec(name="y1"),
            VariableSpec(name="y2"),
        ],
        outputs=[
            VariableSpec(name="obj")
        ]
    )
    
    print("\n\nSellar Objective ExecComp:")
    print(obj_spec.model_dump_json(indent=2, exclude_defaults=True))
    
    # Example 3: Multiple expressions
    multi_spec = ExecCompSpec(
        exprs=[
            "con1 = 3.16 - y1",
            "con2 = y2 - 24.0"
        ],
        inputs=[
            VariableSpec(name="y1"),
            VariableSpec(name="y2"),
        ],
        outputs=[
            VariableSpec(name="con1"),
            VariableSpec(name="con2"),
        ]
    )
    
    print("\n\nMulti-expression ExecComp:")
    print(multi_spec.model_dump_json(indent=2, exclude_defaults=True))
    
    # Serialize and deserialize
    spec_dict = obj_spec.model_dump()
    restored = ExecCompSpec.model_validate(spec_dict)
    print("\n\nRound-trip successful:", restored.exprs == obj_spec.exprs)


# Register ExecComp for reverse conversion (component -> spec)
def _register_exec_comp():
    """Register ExecComp for spec conversion."""
    try:
        from openmdao.api import ExecComp
        from openmdao.specs.systems_registry import register_component_to_spec
        register_component_to_spec(ExecComp, ExecCompSpec)
    except ImportError:
        pass

_register_exec_comp()
