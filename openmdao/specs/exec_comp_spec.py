from typing import Literal
from collections.abc import Sequence
from pydantic import Field, field_validator, model_validator, ConfigDict


from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.variable_spec import VariableSpec


class ExecCompSpec(ComponentSpec):
    """
    Specification for an ExecComp.
    
    ExecComp allows you to define a component using simple expressions.
    This spec captures the expressions and variable metadata.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal['ExecComp'] = 'ExecComp'
    
    exprs: str | Sequence[str] = Field(
        ...,
        description="Expression(s) to be evaluated. Can be a single string or list of strings."
    )
    
    # ExecComp-specific options
    has_diag_partials: bool = Field(
        default=False,
        description="If True, treat all array variables as diagonal partials"
    )
    
    units: str | None = Field(
        default=None,
        description="Default units to apply to all variables."
    )
    
    shape: tuple[int, ...] | int | None = Field(
        default=None,
        description="Default shape to apply to all variables."
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
    
    def to_init_kwargs(self) -> dict:
        """
        Convert this spec to ExecComp __init__ kwargs.
        
        Returns
        -------
        dict
            Dictionary of kwargs suitable for ExecComp.__init__
        """
        kwargs = {
            'exprs': self.exprs,
            'has_diag_partials': self.has_diag_partials,
        }
        
        # Add default units and shape if specified
        if self.units is not None:
            kwargs['units'] = self.units
        if self.shape is not None:
            kwargs['shape'] = self.shape
        
        # Add individual variable specs
        for var in self.inputs + self.outputs:
            var_kwargs = {}
            if var.val is not None:
                var_kwargs['val'] = var.val
            if var.shape is not None:
                var_kwargs['shape'] = var.shape
            if var.units is not None:
                var_kwargs['units'] = var.units
            if var.desc:
                var_kwargs['desc'] = var.desc
            if var.lower is not None:
                var_kwargs['lower'] = var.lower
            if var.upper is not None:
                var_kwargs['upper'] = var.upper
            if var.ref != 1.0:
                var_kwargs['ref'] = var.ref
            if var.ref0 != 0.0:
                var_kwargs['ref0'] = var.ref0
            if var.tags is not None:
                var_kwargs['tags'] = var.tags
            
            if var_kwargs:
                kwargs[var.name] = var_kwargs
        
        return kwargs


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