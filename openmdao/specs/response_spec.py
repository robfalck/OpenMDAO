from collections.abc import Sequence
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import numpy as np

from openmdao.specs.indices_spec import IndicesSpec


class ResponseSpec(BaseModel):
    """Base specification for optimization responses (constraints and objectives)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(
        description='Name of the response variable.'
    )
    
    ref: None | float | np.ndarray = Field(
        default=None,
        description='Value of response variable that scales to 1.0 in the driver.'
    )
    
    ref0: None | float | np.ndarray = Field(
        default=None,
        description='Value of response variable that scales to 0.0 in the driver.'
    )
    
    adder: None | float | np.ndarray = Field(
        default=None,
        description='Value to add to the model value to get the scaled value. '
        'Adder is first in precedence.'
    )
    
    scaler: None | float | np.ndarray = Field(
        default=None,
        description='Value to multiply the model value to get the scaled value. '
        'Scaler is second in precedence.'
    )
    
    parallel_deriv_color: None | str = Field(
        default=None,
        description='If specified, this color will be used to determine which responses '
        'are grouped when doing simultaneous derivative solves.'
    )
    
    cache_linear_solution: bool = Field(
        default=False,
        description='If True, cache the linear solution vector for this response.'
    )
    
    alias: None | str = Field(
        default=None,
        description='Alias for this response.'
    )


class ConstraintSpec(ResponseSpec):
    """Specification for a constraint."""
    
    lower: None | float | Sequence[float] | np.ndarray = Field(
        default=None,
        description='Lower bound of the constraint.'
    )
    
    upper: None | float | Sequence[float] | np.ndarray = Field(
        default=None,
        description='Upper bound of the constraint.'
    )
    
    equals: None | float | Sequence[float] | np.ndarray = Field(
        default=None,
        description='Desired value, if an equality constraint.'
    )
    
    indices: IndicesSpec = Field(
        default_factory=lambda: IndicesSpec(value=None),
        description='Indices of the variable to which this constraint applies.'
    )
    
    linear: bool = Field(
        default=False,
        description='If True, treat this constraint as linear. Its derivatives will only be '
        'computed once and cached.'
    )
    
    flat_indices: bool = Field(
        default=False,
        description='If True, interpret indices as flattened indices.'
    )
    
    @field_validator('indices', mode='before')
    @classmethod
    def validate_indices(cls, v):
        """Convert various index inputs to IndicesSpec."""
        # Already an IndicesSpec
        if isinstance(v, IndicesSpec):
            return v
        
        # None, tuple, ndarray, list, slice, int - wrap in IndicesSpec
        # These will be validated by IndicesSpec.validate_indices
        return IndicesSpec(value=v)
    
    @model_validator(mode='after')
    def validate_bounds(self):
        """Ensure equals is mutually exclusive with lower and upper."""
        if self.equals is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    "Constraint cannot have both 'equals' and 'lower'/'upper' bounds. "
                    "Use 'equals' for equality constraints or 'lower'/'upper' for "
                    "inequality constraints."
                )
        return self


class ObjectiveSpec(ResponseSpec):
    """Specification for an objective."""
    
    index: None | int = Field(
        default=None,
        description='If not None, specifies a single index for objectives that are arrays.'
    )
