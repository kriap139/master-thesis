from .space import Categorical, Integer, Real
from typing import Dict, Union
from numbers import Number

TY_DIM = Union[Integer, Real, Categorical]
TY_SPACE = Dict[str, TY_DIM]