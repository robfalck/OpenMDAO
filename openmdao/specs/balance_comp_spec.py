"""Specification for BalanceComp."""

from pydantic import BaseModel, Field

from openmdao.specs.component_spec import ComponentSpec
from openmdao.specs.options_spec import ComponentOptionsSpec
from openmdao.specs.systems_registry import register_system_spec


class BalanceOptionsSpec(ComponentOptionsSpec):
    """Options for BalanceComp."""

    guess_func: str | None = Field(
        default=None,
        description='Dotted path to a callable function for initial guess. '
                    'Function should have signature f(inputs, outputs, residuals).'
    )


class BalanceSpec(BaseModel):
    """
    Configuration for a single balance equation in a BalanceComp.

    This corresponds to a single call to BalanceComp.add_balance().
    """

    name: str = Field(
        ...,
        description='Name of the state variable to be created'
    )

    eq_units: str | None = Field(
        default=None,
        description='Units for the left-hand-side and right-hand-side of the equation'
    )

    lhs_name: str | None = Field(
        default=None,
        description="Optional name for the LHS variable. Default: 'lhs:{name}'"
    )

    rhs_name: str | None = Field(
        default=None,
        description="Optional name for the RHS variable. Default: 'rhs:{name}'"
    )

    rhs_val: float | list[float] = Field(
        default=0.0,
        description='Default value for the RHS'
    )

    use_mult: bool = Field(
        default=False,
        description='Specifies whether the LHS multiplier is to be used'
    )

    mult_name: str | None = Field(
        default=None,
        description="Optional name for the LHS multiplier. Default: 'mult:{name}'"
    )

    mult_val: float | list[float] = Field(
        default=1.0,
        description='Default value for the LHS multiplier'
    )

    normalize: bool = Field(
        default=True,
        description='Specifies whether residual should be normalized'
    )

    # State variable properties (for add_output)
    val: float | list[float] | None = Field(
        default=None,
        description='Initial value for the state variable'
    )

    shape: tuple | None = Field(
        default=None,
        description='Shape of the state variable'
    )

    units: str | None = Field(
        default=None,
        description='Units for the state variable'
    )

    lower: float | None = Field(
        default=None,
        description='Lower bound for the state variable'
    )

    upper: float | None = Field(
        default=None,
        description='Upper bound for the state variable'
    )

    ref: float | None = Field(
        default=None,
        description='Reference value for scaling'
    )

    ref0: float | None = Field(
        default=None,
        description='Zero-reference value for scaling'
    )


@register_system_spec
class BalanceCompSpec(ComponentSpec[BalanceOptionsSpec]):
    """
    Specification for BalanceComp.

    BalanceComp solves implicit equations of the form:
        lhs * mult - rhs = 0

    Each balance creates a state variable and associated LHS/RHS inputs.
    """

    system_type: str = 'BalanceComp'

    path: str = Field(
        default='openmdao.components.balance_comp.BalanceComp',
        description='Path to BalanceComp class'
    )

    balances: list[BalanceSpec] = Field(
        default_factory=list,
        description='List of balance equations to configure'
    )

    # Override inputs/outputs - BalanceComp creates these automatically
    inputs: list = Field(
        default_factory=list,
        description='Inputs are created automatically from balances'
    )

    outputs: list = Field(
        default_factory=list,
        description='Outputs (state variables) are created automatically from balances'
    )

    def setup(self, comp):
        """
        Configure BalanceComp by calling add_balance for each balance.

        BalanceComp.add_balance() creates the state variable output and
        the associated LHS, RHS, and optionally MULT inputs.

        Parameters
        ----------
        comp : BalanceComp
            The BalanceComp instance to configure
        """
        for balance_spec in self.balances:
            # Convert balance spec to kwargs for add_balance
            kwargs = balance_spec.model_dump(
                exclude={'name'},  # name is positional
                exclude_none=True   # don't pass None values
            )

            # Call add_balance on the component
            comp.add_balance(name=balance_spec.name, **kwargs)


# Register BalanceComp for reverse conversion (component -> spec)
def _register_balance_comp():
    """Register BalanceComp for spec conversion."""
    try:
        from openmdao.components.balance_comp import BalanceComp
        from openmdao.specs.systems_registry import register_component_to_spec
        register_component_to_spec(BalanceComp, BalanceCompSpec)
    except ImportError:
        pass


_register_balance_comp()
