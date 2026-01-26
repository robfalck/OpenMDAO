"""Base solver options specification."""
from pydantic import BaseModel, Field


class SolverOptionsSpec(BaseModel):
    """Base specification for solver options (common to both linear and nonlinear)."""

    maxiter : int = Field(
        default=10,
        description='maximum number of iterations')

    atol : float = Field(
        default=1.0E-10,
        description='absolute error tolerance')

    rtol : float = Field(
        default=1.0E-10,
        description='relative error tolerance')

    iprint : int = Field(
        default=1,
        description='whether to print output')

    err_on_non_converge : bool = Field(
        default=False,
        description="When True, AnalysisError will be raised if we do't converge.")
