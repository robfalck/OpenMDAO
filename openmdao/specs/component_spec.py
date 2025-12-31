from collections.abc import Sequence

from pydantic import BaseModel, Field, field_validator

from openmdao.specs.partials_spec import PartialsSpec
from openmdao.specs.variable_spec import VariableSpec


class ComponentSpec(BaseModel):

    inputs : Sequence[VariableSpec] = Field(
        ...,
        description='Input variables for this component.'
    )

    outputs : Sequence[VariableSpec] = Field(
        ...,
        description='Output variables for this component.'
    )

    path : str | None = Field(
        default=None,
        description='If given, the dotted path to the component class.'
    )

    compute : str | None = Field(
        default=None,
        description="If given, the dotted path to the component's compute function."
    )

    compute_partials : str | None = Field(
        default=None,
        description="If given, the dotted path to the component's compute_partials function."
    )

    compute_jacvec_prod : str | None = Field(
        default=None,
        description="If given, the dotted path to the component's compute_jacvec_prod function.")

    partials : list[PartialsSpec] = Field(
        default_factory=list,
        description="Partial derivatives declared for the outputs of this component."
    )

    @field_validator('partials', mode='before')
    @classmethod
    def convert_partials_to_list(cls, v):
        """Convert any collection of PartialsSpec to a list."""
        if isinstance(v, list):
            return v
        return list(v)
