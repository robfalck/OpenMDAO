from collections.abc import Sequence
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np

from openmdao.specs.indices_spec import IndicesSpec


class DesignVarSpec(BaseModel):
    """Base specification for optimization responses (constraints and objectives)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(
        description='Name of the response variable.'
    )

    lower: None | float | Sequence[float] | np.ndarray = Field(
        default=None,
        description='Lower bound of the constraint.'
    )
    
    upper: None | float | Sequence[float] | np.ndarray = Field(
        default=None,
        description='Upper bound of the constraint.'
    )
    
    ref: None | float | np.ndarray = Field(
        default=None,
        description='Value of response variable that scales to 1.0 in the driver.'
    )
    
    ref0: None | float | np.ndarray = Field(
        default=None,
        description='Value of response variable that scales to 0.0 in the driver.'
    )

    indices: IndicesSpec = Field(
        default_factory=lambda: IndicesSpec(value=None),
        description='Indices of the variable to which this constraint applies.'
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

    units : str | None = Field(
        default=None,
        description='Units to be assigned to all variables in this component. '
                    'Default is None, which means units may be provided for variables '
                    'individually.'
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

        # Dict (from deserialization) - use model_validate
        if isinstance(v, dict):
            return IndicesSpec.model_validate(v)

        # None, tuple, ndarray, list, slice, int - wrap in IndicesSpec
        # These will be validated by IndicesSpec.validate_indices
        return IndicesSpec(value=v)
