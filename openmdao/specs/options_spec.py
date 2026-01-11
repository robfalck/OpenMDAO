from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


ComponentOptionsT = TypeVar('ComponentOptionsT', bound='ComponentOptionsSpec')


class RecordingOptionsSpec(BaseModel):

    record_inputs: bool = Field(
        default=True,
        description='Set to True to record inputs at the system level'
    )
    
    record_outputs: bool = Field(
        default=True,
        description='Set to True to record outputs at the system level'
    )
    
    record_residuals: bool = Field(
        default=True,
        description='Set to True to record residuals at the system level'
    )
    
    includes: list[str] = Field(
        default=['*'],
        description='Patterns for variables to include in recording. Uses fnmatch wildcards'
    )
    
    excludes: list[str] = Field(
        default=[],
        description='Patterns for vars to exclude in recording (processed post-includes). '
                    'Uses fnmatch wildcards'
    )
    
    options_excludes: list[str] = Field(
        default=[],
        description='User-defined metadata to exclude in recording'
    )


class SystemOptionsSpec(BaseModel):

    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    assembled_jac_type: Literal['csc', 'csr', 'dense'] | None = Field(
        default=None,
        description='Linear solver in this group or implicit component, '
                    'if using an assembled jacobian, will use this type.'
    )

    derivs_method: Literal['jax', 'cs', 'fd'] | None = Field(
        default=None,
        description='The method to use for computing derivatives.'
    )


class ComponentOptionsSpec(SystemOptionsSpec):
    """
    Base class for component options.

    These are options on all components.
    """
    
    distributed: bool = Field(
        default=False,
        description='If True, set all variables in this component as distributed '
                    'across multiple processes'
    )

    run_root_only: bool = Field(
        default=False,
        description='If True, call compute, compute_partials, linearize, '
                    'apply_linear, apply_nonlinear, solve_linear, solve_nonlinear, '
                    'and compute_jacvec_product only on rank 0 and broadcast the '
                    'results to the other ranks.'
    )

    always_opt: bool = Field(
        default=False,
        description='If True, force nonlinear operations on this component to be '
                    'included in the optimization loop even if this component is not '
                    'relevant to the design variables and responses.'
    )

    use_jit: bool = Field(
        default=True,
        description='If True, attempt to use jit on compute_primal, assuming jax '
                    'or some other AD package capable of jitting is active.'
    )

    default_shape: tuple[int, ...] = Field(
        default=(1,),
        description='Default shape for variables that do not set val to a non-scalar '
                    'value or set shape, shape_by_conn, copy_shape, or compute_shape.'
                    ' Default is (1,).'
    )
