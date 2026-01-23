"""
Specifications for variable promotions in OpenMDAO specs.
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from openmdao.specs.indices_spec import SrcIndicesSpec


class PromotesSpec(BaseModel):
    """
    Specification for promoting a single variable in a group or subsystem.

    Represents a single variable promotion, supporting both simple promotion
    and advanced features like src_indices and src_shape.

    A promoted variable can be specified as:
    - A string name: 'x' (simple promotion)
    - A tuple (old_name, new_name): ('x', 'X') (promotion with renaming)

    Parameters
    ----------
    name : str or tuple[str, str]
        Variable name to promote. Can be a string for simple promotion,
        or a tuple (old_name, new_name) for renaming.

    io_type : {'input', 'output', 'any'}
        Whether this promotes an input, output, or either. Default is 'any'.

    src_indices : SrcIndicesSpec, optional
        Indices into source variable. Only applicable for promoted inputs.

    src_shape : tuple or int, optional
        Assumed shape of connected source. Only applicable for promoted inputs.

    subsys_name : str, optional
        Name of the subsystem to promote from. If None, applies to current system
        (used for simple promotions via add_subsystem). If specified, indicates
        this promotion should be done via a Group.promotes() call.

    Examples
    --------
    # Simple promotion of variable 'x' in current system
    pspec1 = PromotesSpec(name='x', io_type='input')

    # Promotion with renaming
    pspec2 = PromotesSpec(name=('old_x', 'new_x'), io_type='input')

    # Promotion with array indexing from a specific subsystem
    pspec3 = PromotesSpec(
        name='z',
        io_type='input',
        src_indices=[0, 1, 2],
        src_shape=(3,),
        subsys_name='comp1'
    )

    # Promotion from current system with advanced features
    pspec4 = PromotesSpec(
        name='y',
        io_type='input',
        subsys_name=None  # None means current system
    )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | tuple[str, str] = Field(
        ...,
        description="Variable name to promote. Can be a string for simple promotion, "
                    "or a tuple (old_name, new_name) for renaming."
    )

    io_type: Literal['input', 'output', 'any'] = Field(
        default='any',
        description="Whether this promotes an input, output, or either."
    )

    src_indices: SrcIndicesSpec | None = Field(
        default=None,
        description="Indices into source variable. Only applicable for promoted inputs."
    )

    src_shape: tuple[int, ...] | int | None = Field(
        default=None,
        description="Assumed shape of connected source. Only applicable for promoted inputs."
    )

    subsys_name: str | None = Field(
        default=None,
        description="Name of subsystem to promote from. None means current system."
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate name is string or 2-tuple of strings."""
        if isinstance(v, str):
            return v
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, str) for x in v):
            return v
        raise ValueError(
            "name must be a string or a 2-tuple of strings (old_name, new_name)"
        )

    @field_validator('src_shape', mode='before')
    @classmethod
    def normalize_src_shape(cls, v):
        """Normalize src_shape to tuple."""
        if v is None:
            return None
        if isinstance(v, int):
            return (v,)
        if isinstance(v, (tuple, list)):
            return tuple(v)
        return v

    @model_validator(mode='after')
    def validate_indexing_only_for_inputs(self):
        """Ensure src_indices/src_shape only used for inputs."""
        if (self.io_type == 'output' and
                (self.src_indices is not None or self.src_shape is not None)):
            raise ValueError(
                "src_indices and src_shape are only valid for promoted inputs, "
                "not outputs"
            )
        return self
