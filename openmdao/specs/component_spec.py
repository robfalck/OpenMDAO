from collections.abc import Sequence, Collection

from pydantic import BaseModel, Field

from openmdao.specs import PartialsSpec, VariableSpec


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

    partials : Collection[PartialsSpec] = Field(
        default_factory=list,
        description="Partial derivatives declared for the outputs of this component."
    )
