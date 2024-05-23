import numpy as np
import pandas as pd
from typing import Iterable, Union, Dict, Optional
from numbers import Number
from abc import ABC, abstractclassmethod
from Util import Integer, Categorical, Real, TY_SPACE
from typing import Type

TY_RETURN = Union[pd.Series, float, int, str]
TY_X = Union[pd.Series, float, str]

class KSpace:
    def __init__(self, k_space: TY_SPACE, k:  Union[Number, dict] = None, x_in_search_space=False):
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
            if self.k_space[param].is_type(Real) and x_in_search_space:
                self.mapping_funcs[param] = self._rescale_wrapper
            elif self.k_space[param].is_type(Real):
                self.mapping_funcs[param] = lambda param, x: (
                    self.f(x, self.k_space[param].low, self.k_space[param].high, self._kmap[param])
                )
            elif self.k_space[param].is_type(Integer) and x_in_search_space:
                self.mapping_funcs[param] = lambda param, x: (
                    self._int_wrapper(
                        self._rescale_wrapper(param, x)
                    )
                )
            elif self.k_space[param].is_type(Integer):
                self.mapping_funcs[param] = lambda param, x: (
                    self._int_wrapper(
                        self.f(x, self.k_space[param].low, self.k_space[param].high, self._kmap[param])
                    )
                )
            elif self.k_space[param].is_type(Categorical):
                pass
    
    @classmethod
    def h(cls, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return (1 - x) * (y_u / np.exp(abs(k) * x)) + y_l
    
    @classmethod
    def g(cls, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return y_u - cls.h((1 - x), y_l, y_u, k)
    
    @classmethod
    def f(cls, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        if k <= 0:
            return cls.h(x, y_l, y_u, k)
        else:
            return cls.g(x, y_l, y_u, k)
    
    @classmethod
    def _int_wrapper(cls, x: TY_X) -> TY_RETURN:
        if type(x) == pd.Series:
            return x.round().astype(int)
        elif isinstance(x, Number):
            return round(x)
        elif isinstance(x, np.ndarray):
            return x.round()
        else:
            return x
    
    @classmethod
    def _rescale(cls, y_u: Number, y_l: Number, y: TY_X) -> TY_X:
        slope = 1 / (y_u - y_l)
        return slope * (y - y_l)

    def _rescale_wrapper(self, param: str, y: TY_X) -> TY_RETURN:
        y_u, y_l = self.k_space[param].high, self.k_space[param].low
        x = self._rescale(y_u, y_l, y)
        ky = self.f(x, y_l, y_u, self._kmap[param])
        #print(f"param={param}, x_inn={y}, x={x}, y={ky}")
        return ky

    def kmap(self, param: str, x: TY_X, default=None) -> TY_RETURN:
        f = self.mapping_funcs.get(param, None)
        if f is not None:
            return f(param, x)
        elif default == 'raise':
            raise ValueError(f"param({param}) have no associated kmap function!")
        elif callable(default):
            return default()
        return default

class KSpaceV2(KSpace):
    @classmethod
    def h(cls, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return x * (y_u / np.exp(abs(k) * (1 - x))) + y_l

class KSpaceV3(KSpace):
    @classmethod
    def h(cls, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return x * ((y_u - y_l) / np.exp(abs(k) * (1 - x))) + y_l
    
    @classmethod
    def g(cls, x: TY_X, y_l: Number, y_u: Number, k: Number) -> TY_RETURN:
        return (y_u + y_l) - cls.h((1 - x), y_l, y_u, k)


def get_kspace_ver(k_space_ver: int) -> Type[KSpace]:
    if k_space_ver == 1:
        return KSpace
    else:
        import sys
        this_module = sys.modules[__name__]

        cls = getattr(k_space, "KSpaceV" + str(k_space_ver), None)
        if cls is None:
            raise RuntimeError(f"Invalid kspace implementation version: {k_space_ver}")
        return cls