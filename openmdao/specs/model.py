"""
openmdao.specs.model

Model is a Pydantic BaseModel-derived class that is the equivalent of a top-level Group upon which
setup() has been called. This is stored for efficiency but in general users shouldn't be dealing
with Model.
"""

from pydantic import BaseModel
from .group_spec import GroupSpec


class Model(GroupSpec):

    pathname : str = ''
    pre_components : list[str] | None = None
    post_components : list[str] | None = None
