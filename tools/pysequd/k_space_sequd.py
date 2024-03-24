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
from .k_space import KSpace, Integer, Real, Categorical

class KSpaceSeqUD(SeqUD):
    def __init__(self, para_space, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=True, random_state=0, verbose=0, error_score='raise', k:  Union[Number, dict] = None):
        super(SeqUD, self).__init__(para_space, n_runs_per_stage, max_runs, max_search_iter, n_jobs, estimator, cv, scoring, refit, random_state, verbose, error_score)
        self.kspace = KSpace(self._create_k_space(para_space), k)
    
    def _create_k_space(self, para_space: dict) -> dict:
        space = {}
        for k, v in search_space.items():
            if v['Type'] == 'continuous':
                space[k] = Real(v["Range"][0], v["Range"][0], name=k)
            elif isinstance(v, Integer):
                space[k] = Integer(v["Mapping"][0], v["Mapping"][-1], name=k)
            elif isinstance(v, Categorical):
                space[k] = Categorical(v["Mapping"], name=k)
            else:
                raise ValueError(f"search space contains unsupported type for '{k}': {type(v)}")
        return space
    
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
            value = self.kspace.kmap(param, x=para_set_ud[param + '_UD'])
            if value is not None:
                para_set[param] = value
            else:
                self._passtrough_mapping(para_set, para_set_ud, param, info)
        return MappingData(para_set, logs_append=None)