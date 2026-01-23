
from pydantic import BaseModel, Field, ConfigDict, field_validator
from openmdao.core.constants import _UNDEFINED


class InputDefaultsSpec(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name : str = Field(
        ...,
        description=('The promoted input name for which defaults are being set.'))
    
    val : float | complex | list[float | complex | int] | None | object = Field(
        default=None,
        description='The default initial value of the input.'
    )

    units : str | None = Field(
        default=None,
        description='The default units associated with this input.')
    
    src_shape : tuple[int, ...] | None = Field(
        default=None,
        description='The default shape of the output connected to this input' )

    @field_validator('val', mode='before')
    @classmethod
    def validate_val(cls, v):
        if v is _UNDEFINED or isinstance(v, (float, complex, list, type(None))):
            return v
        raise ValueError(f'val must be a numeric value, None, or _UNDEFINED, got {type(v)}')