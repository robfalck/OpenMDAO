
from pydantic import BaseModel, Field, ConfigDict


class VariableSpec(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name : str = Field(
        ...,
        description=('The name of the variable in a component.'))
    units : str | None = Field(
        default=None,
        description='The units of the variable for within a component.')
    shape : tuple[int, ...] | None = Field(
        default=None,
        description='Shape of this variable, only required if val is '
        'not an array. Default is None.')
    desc : str = Field(
        default='',
        description='A description of the variable.')
    tags : set[str] = Field(
        default_factory=set,
        description='Tags used to denote characteristics of the Varaible.')
    shape_by_conn : bool = Field(
        default=False,
        description='If True, shape this variable to match its connected input or output.')
    copy_shape : str | None = Field(
        default=None,
        description='If a str, that str is the name of a variable. '
        'Shape this input to match that of the named variable.')
    compute_shape : str | None = Field(
        default=None,
        description="Dotted path to a function taking a dict arg "
        "containing names and shapes of this component's "
        "outputs and returning the shape of this input.")
    units_by_conn : bool = Field(
        default=False,
        description='If True, set units of this input to match its connected output.')
    copy_units : str | None = Field(
        default=None,
        description='If True, set units of this input to match its connected output.')
    compute_units : str | None = Field(
        default=None,
        description="Dotted path to a function taking a dict arg containing names and "
        "PhysicalUnits of this component's outputs and returning the PhysicalUnits of "
        "this input.")
    require_conection : bool = Field(
        default=False,
        description='For inputs, if True and this is not a ' \
        'design variable, it must be connected to an output.')
    distributed : bool = Field(
        default=False,
        description='If True, this variable is a distributed variable,'
        'so it can have different sizes/values across MPI processes.')
    primal_name : str | None = Field(
        default=None,
        description="Valid python name to represent the variable in" \
        "compute_primal if 'name' is not a valid python name.")
