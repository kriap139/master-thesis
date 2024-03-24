import numpy as np
import pandas as pd
from typing import Iterable, Union, Dict, Optional
from numbers import Number
from abc import ABC, abstractclassmethod
from ..Util import Integer, Categorical, Real

TY_DIM = Union[Integer, Real, Categorical]
TY_RETURN = Union[pd.Series, float, int, str]
TY_X = Union[pd.Series, float, str]

class KSpace:
    def __init__(self, k_space: Dict[TY_DIM], k:  Union[Number, dict] = None, x_in_search_space=False):
        self.k_space = k_space
        self.k = k
        self.mapping_funcs = {}
        self.x_in_search_space = x_in_search_space

        if isinstance(k, dict):
            self._kmap = k
            for param, _k in self._kmap.items():
                if self.k_space[param].is_type(Categorical):
                    raise ValueError(f"Categorical parameter ({param}) is not yet supported!")
                elif not isinstance(_k, Number):
                    raise ValueError(f"passed k value for {param} is not a number: {_k}")
        elif isinstance(k, Number):
            self._kmap =  {param: k for param, space in self.k_space.items() if not space.is_type(Categorical)}
        else:
            raise ValueError(f"k argument is not of supported types ('int', 'float', 'dict'): {type(k)}")
        
        for param, _ in self._kmap.items():
            if self.k_space[param].is_type(Real):
                self.mapping_funcs[param] = lambda param, x: (
                    self.f(x, self.k_space[param].low, self.k_space[param].high, self._kmap[param])
                )
            elif self.k_space[param].is_type(Real) and x_in_search_space:
                self.mapping_funcs[param] = self._rescale_wrapper
            elif self.k_space[param].is_type(Integer):
                self.mapping_funcs[param] = lambda param, x: (
                    self._int_wrapper(
                        self.f(x, self.k_space[param].low, self.k_space[param].high, self._kmap[param])
                    )
                )
            elif self.k_space[param].is_type(Integer) and x_in_search_space:
                self.mapping_funcs[param] = lambda param, x: (
                    self._int_wrapper(
                        self._rescale_wrapper(param, x)
                    )
                )
            elif self.k_space[param].is_type(Categorical):
                pass
    
    def h(self, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return (1 - x) * (y_u / np.exp(abs(k) * x)) + y_l
    
    def g(self, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return y_u - self.h((1 - x), y_u, y_l, k)
    
    def f(self, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        if k <= 0:
            return self.h(x, y_l, y_u, k)
        else:
            return self.g(x, y_l, y_u)
    
    @classmethod
    def _int_wrapper(cls, x: TY_X) -> TY_RETURN:
        if type(x) == pd.Series:
            return x.round().astype(int)
        elif isinstance(x, Number):
            return round(x)
        return x
    
    def _rescale_wrapper(self, param: str, y: TY_X) -> TY_RETURN:
        y_u, y_l = self.k_space[param].high, self.k_space[param].low
        x = (y - y_l) / (y_u - y_l)
        return self.f(x, y_l, y_u, self._kmap[param])

    def f_cat(self, param: str, x: TY_CAT) -> TY_RETURN:
        NotImplemented

    def kmap(self, param: str, x: TY_X, default=None) -> TY_RETURN:
        f = self.mapping_funcs.get(param, None)
        if f is not None:
            return f(param, x)
        elif default == 'raise':
            raise ValueError(f"param({param}) have no associated kmap function!")
        elif callable(default):
            return default()
        return default



