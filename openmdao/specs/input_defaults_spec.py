
from pydantic import BaseModel, Field, ConfigDict


class InputDefaultsSpec(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name : str = Field(
        ...,
        description=('The promoted input name for which defaults are being set.'))
    
    val : float | complex | list[float | complex | int] | None = Field(
        default=None,
        description='The default initial value of the input.'
    )

    units : str | None = Field(
        default=None,
        description='The default units associated with this input.')
    
    src_shape : tuple[int, ...] | None = Field(
        default=None,
        description='The default shape of the output connected to this input' )
