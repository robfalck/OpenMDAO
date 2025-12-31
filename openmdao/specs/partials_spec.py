from typing import Literal, Sequence
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import numpy as np


class PartialsSpec(BaseModel):
    """
    Specification for declaring partial derivatives.
    """
    
    model_config = ConfigDict(
        # Allow numpy arrays in place of Sequences
        arbitrary_types_allowed=True,
    )
    
    # Required: what derivatives are being declared
    of: str | list[str] = Field(
        ...,
        description="Name(s) of residual/output that derivatives are computed for."
        "May contain glob pattern."
    )
    wrt: str | list[str] = Field(
        ...,
        description="Name(s) of variables that derivatives are taken with respect to."
        "May contain glob pattern."
    )
    
    # Dependency specification
    dependent: bool = Field(
        default=True,
        description="If False, specifies no dependence between output(s) and input(s). "
                    "Used to mark independence in sparse global jacobian."
    )
    
    # Sparse jacobian specification - accepts any sequence including numpy arrays
    rows: Sequence[int] | np.ndarray | None = Field(
        default=None,
        description="Row indices for each nonzero entry. For sparse subjacobians only. "
                    "Can be list, tuple, numpy array, or any sequence."
    )

    cols: Sequence[int] | np.ndarray | None = Field(
        default=None,
        description="Column indices for each nonzero entry. For sparse subjacobians only. "
                    "Can be list, tuple, numpy array, or any sequence."
    )
    
    # Value specification - accepts scalars, sequences, nested sequences, or numpy arrays
    val: float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = Field(
        default=None,
        description="Value of subjacobian. Can be scalar, 1D array/sequence, 2D array/sequence, "
                    "numpy array, or sparse matrix data. If rows and cols provided, contains "
                    "values at each (row, col) location."
    )
    
    # Approximation method
    method: Literal["exact", "fd", "cs"] = Field(
        default="exact",
        description="Approximation method: 'exact' for analytic derivatives, "
                    "'fd' for finite difference, 'cs' for complex step."
    )
    
    # Finite difference / complex step options
    step: float | None = Field(
        default=None,
        description="Step size for approximation. If None, method provides default."
    )
    
    form: Literal["forward", "backward", "central"] | None = Field(
        default=None,
        description="Form for finite difference: 'forward', 'backward', or 'central'. "
                    "If None, method provides default."
    )
    
    step_calc: Literal["abs", "rel", "rel_avg", "rel_element", "rel_legacy"] | None = Field(
        default=None,
        description="Step type for finite difference size: 'abs' for absolute, "
                    "'rel_avg' for relative to vector norm, 'rel_element' for relative to each "
                    "element, 'rel_legacy' for legacy relative. 'rel' is equivalent to 'rel_avg'. "
                    "If None, method provides default."
    )
    
    minimum_step: float | None = Field(
        default=None,
        description="Minimum step size when using relative step_calc options."
    )
    
    # Matrix structure
    diagonal: bool | None = Field(
        default=None,
        description="If True, the subjacobian is a diagonal matrix."
    )
    
    @field_validator('of', 'wrt')
    @classmethod
    def normalize_variable_names(cls, v):
        """Ensure variable names are always stored as lists internally."""
        if isinstance(v, str):
            return [v]
        return list(v)
    
    @field_validator('rows', 'cols', mode='before')
    @classmethod
    def convert_array_to_list(cls, v):
        """Convert numpy arrays to lists for JSON serialization."""
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Sequence) and not isinstance(v, str):
            return list(v)
        return v
    
    @field_validator('val', mode='before')
    @classmethod
    def convert_val_to_serializable(cls, v):
        """Convert numpy arrays and sequences to serializable format."""
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (int, float)):
            return float(v)
        # Handle sequences (but not strings)
        if isinstance(v, Sequence) and not isinstance(v, str):
            # Check if it's nested
            if len(v) > 0 and isinstance(v[0], (Sequence, np.ndarray)) \
                and not isinstance(v[0], str):
                # 2D sequence
                return [list(row) if isinstance(row, np.ndarray) else list(row) for row in v]
            # 1D sequence
            return list(v)
        return v
    
    @field_validator('step', 'minimum_step')
    @classmethod
    def validate_positive_step(cls, v):
        """Ensure step sizes are positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Step sizes must be positive")
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate logical consistency after initialization."""
        # If rows or cols provided, both must be provided
        if (self.rows is None) != (self.cols is None):
            raise ValueError("Both 'rows' and 'cols' must be provided together for "
                             "sparse specification")
        
        # If rows/cols provided and val is a list, lengths must match
        if self.rows is not None and isinstance(self.val, list):
            if not isinstance(self.val[0], list):  # 1D list
                if len(self.val) != len(self.rows):
                    raise ValueError(f"Length of val ({len(self.val)}) must match length "
                                     f"of rows/cols ({len(self.rows)})")
        
        # FD/CS options only make sense with appropriate method
        if self.method == "exact":
            if self.step is not None or self.form is not None or self.step_calc is not None:
                raise ValueError("step, form, and step_calc are only valid for 'fd' "
                                 "or 'cs' methods")
        
        # form is only for finite difference
        if self.form is not None and self.method != "fd":
            raise ValueError("form option is only valid for 'fd' method")
        
        return self


# Example usage demonstrating different input types
if __name__ == "__main__":
    import numpy as np
    
    # Example 1: Using numpy arrays
    partials1 = PartialsSpec(
        of="g",
        wrt="x",
        rows=np.array([0, 2, 1, 3]),
        cols=np.array([0, 0, 1, 1]),
        val=np.array([1.0, 1.0, 1.0, 1.0])
    )
    print("With numpy arrays:")
    print(partials1.model_dump_json(indent=2))
    print(f"Stored rows type: {type(partials1.rows)}")
    
    # Example 2: Using tuples
    partials2 = PartialsSpec(
        of="g",
        wrt="x",
        rows=(0, 2, 1, 3),
        cols=(0, 0, 1, 1),
        val=(1.0, 1.0, 1.0, 1.0)
    )
    print("\nWith tuples:")
    print(partials2.model_dump_json(indent=2))
    
    # Example 3: 2D numpy array for val
    partials3 = PartialsSpec(
        of="g",
        wrt="y1",
        val=np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    )
    print("\n2D numpy array:")
    print(partials3.model_dump_json(indent=2))
    
    # Example 4: Mixed - numpy for val, list for rows/cols
    partials4 = PartialsSpec(
        of="output",
        wrt="input",
        rows=[0, 1, 2],
        cols=[0, 1, 2],
        val=np.ones(3)
    )
    print("\nMixed types:")
    print(partials4.model_dump_json(indent=2))
    
    # Example 5: Demonstrate that it works with OpenMDAO-style arrays
    jacobian_data = np.random.rand(4, 4)
    row_indices = np.where(jacobian_data > 0.5)[0]
    col_indices = np.where(jacobian_data > 0.5)[1]
    values = jacobian_data[jacobian_data > 0.5]
    
    partials5 = PartialsSpec(
        of="output",
        wrt="input",
        rows=row_indices,
        cols=col_indices,
        val=values
    )
    print("\nFrom sparse jacobian extraction:")
    print(f"Number of nonzeros: {len(partials5.rows)}")
    print(partials5.model_dump_json(indent=2))
    
    # Example 6: Finite difference with all options
    partials6 = PartialsSpec(
        of="residual",
        wrt="state",
        method="fd",
        form="central",
        step=1e-6,
        step_calc="rel_avg",
        minimum_step=1e-12
    )
    print("\nFinite difference with options:")
    print(partials6.model_dump_json(indent=2))