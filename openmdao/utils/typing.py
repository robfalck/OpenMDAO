from typing import Type, TypeVar

from openmdao.core.system import System

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent


SystemDerivedType = TypeVar('SystemDerivedType', bound=System)