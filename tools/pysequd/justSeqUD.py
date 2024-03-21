import time
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from matplotlib import pylab as plt
from sklearn.model_selection import cross_val_score
from typing import Iterable, Callable
import math
import pyunidoe as pydoe
from itertools import chain
from sequd import SeqUD, MappingData, EPS
from typing import Iterable, Union
from numbers import Number

class JustSeqUD(SeqUD):
    def __init__(self, para_space, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=True, random_state=0, verbose=0, error_score='raise', k:  Union[Number, dict] =None):
        super().__init__(para_space, n_runs_per_stage, max_runs, max_search_iter, n_jobs, estimator, cv, scoring, refit, random_state, verbose, error_score)
        self.k = k
        self.mapping_funcs = {}

        if isinstance(k, dict):
            self._kmap = k
            for param, _k in self._kmap.items():
                if self.para_space[param]['Type'] == 'categorical':
                    raise ValueError(f"Categorical parameter ({param}) is not yet supported!")
                elif not isinstance(_k, Number):
                    raise ValueError(f"passed k value for {param} is not a number: {_k}")
        elif isinstance(k, Number):
            self._kmap =  {param: k for param, info in self.para_space.items() if info['Type'] != 'categorical'}
        else:
            raise ValueError(f"k argument is not of supported types ('int', 'float', 'dict'): {type(k)}")
        
        for param, _ in self._kmap.items():
            if self.para_space[param]["Type"] == "continuous":
                self.mapping_funcs[param] = self._map_float
            elif self.para_space[param]["Type"] == "integer":
                self.mapping_funcs[param] = self._map_int

    def h(self, x: float, y_l: Number, y_u: Number, k: Number) -> Number:
        return (1 - x) * (y_u / np.exp(abs(k) * x)) + y_l
    
    def g(self, x: float, y_l: Number, y_u: Number, k: Number) -> Number:
        return y_u - self.h((1 - x), y_u, y_l, k)
    
    def f(self, x: float, y_l: Number, y_u: Number, k: Number) -> Number:
        if k <= 0:
            return self.h(x, y_l, y_u, k)
        else:
            return self.g(x, y_l, y_u)
    
    def _map_int(self, param: str, x: float) -> pd.Series:
        series = self.f(x, self.para_space[param]["Mapping"][0], self.para_space[param]["Mapping"][-1], self._kmap[param])
        return  series.round().astype(int)
    
    def _map_float(self, param: str, x: float) -> pd.Series:
        return self.f(x, self.para_space[param]["Range"][0], self.para_space[param]["Range"][1], self._kmap[param])
    
    def _map_cat(self, param: str, x: float) -> pd.Series:
        pass
    
    def _passtrough_mapping(self, para_set: pd.DataFrame, para_set_ud: pd.DataFrame, param: str, info: dict):
        if info['Type'] == "continuous":
            para_set[param] = info['Wrapper'](
                para_set_ud[param + "_UD"] * (info['Range'][1] - info['Range'][0]) + info['Range'][0]
            )
        elif info['Type'] == "integer":
            temp = np.linspace(0, 1, len(info['Mapping']) + 1)
            for j in range(1, len(temp)):
                para_set.loc[
                    (para_set_ud[param + "_UD"] >= (temp[j - 1] - EPS)) & (para_set_ud[param + "_UD"] < (temp[j] + EPS)), param
                ] = info['Mapping'][j - 1]

            para_set.loc[np.abs(para_set_ud[param + "_UD"] - 1) <= EPS, param] = info['Mapping'][-1]
            para_set[param] = para_set[param].round().astype(int)
        elif info['Type'] == "categorical":
            column_bool = [
                param == para_name[::-1].split("DU_", maxsplit=1)[1][::-1] for para_name in self.para_ud_names
            ]
            col_index = np.argmax(para_set_ud.loc[:, column_bool].values, axis=1).tolist()
            para_set[param] = np.array(info['Mapping'])[col_index]

    def _para_mapping(self, para_set_ud, log_append=True):
        para_set = pd.DataFrame(np.zeros((para_set_ud.shape[0], self.factor_number)), columns=self.para_names)

        for param, info in self.para_space.items():
            f = self.mapping_funcs.get(param, None)

            if f is not None:
                para_set[param] = f(param, para_set_ud[param + '_UD'])
            else:
                self._passtrough_mapping(para_set, para_set_ud, param, info)

        return MappingData(para_set, logs_append=None)