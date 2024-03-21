import time
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from matplotlib import pylab as plt
from sklearn.model_selection import cross_val_score
from typing import Iterable
import math
import pyunidoe as pydoe
from itertools import chain
from sequd import SeqUD, MappingData, EPS
from typing import Iterable
from numbers import Number


class JustSeqUD(SeqUD):
    def __init__(self, para_space, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=True, random_state=0, verbose=0, error_score='raise', k=0, just_params: Iterable =None):
        super().__init__(para_space, n_runs_per_stage, max_runs, max_search_iter, n_jobs, estimator, cv, scoring, refit, random_state, verbose, error_score)
        self.k = k
        self.just_params = just_params

        if self.just_params is None:
            self.just_params = [k for k, v in self.para_space.items() if v['Type'] != 'categorical']
        else:
            for k, v in para_space.items():
                if v['Type'] == 'categorical':
                    raise ValueError(f"Categorical parameter ({k}) is not yet supported!")
        
        if self.k <= 0:
            self.f = self.h
        else:
            self.f = self.g

    def h(self, x: float, y_l: Number, y_u: Number) -> Number:
        return (1 - x) * (y_u / np.exp(abs(self.k) * x)) + y_l
    
    def g(self, x: float, y_l: Number, y_u: Number) -> Number:
        return y_u - self.h((1 - x), y_u, y_l)
    
    def _passtrough_mapping(self):
        pass

    def _para_mapping(self, para_set_ud, log_append=True):
        para_set = pd.DataFrame(np.zeros((para_set_ud.shape[0], self.factor_number)), columns=self.para_names)

        for item, values in self.para_space.items():
            if item in self.just_params and (values['Type'] == 'continuous'):
                para_set[item] = self.f(para_set_ud[item + '_UD'], values['Range'][0], values['Range'][1])
            elif item in self.just_params and (values['Type'] == 'integer'):
                para_set[item] = self.f(para_set_ud[item + '_UD'], values['Mapping'][0], values['Mapping'][-1])
                para_set[item] = para_set[item].round().astype(int)
            elif (values['Type'] == "continuous"):
                para_set[item] = values['Wrapper'](
                    para_set_ud[item + "_UD"] * (values['Range'][1] - values['Range'][0]) + values['Range'][0])
            elif (values['Type'] == "integer"):
                temp = np.linspace(0, 1, len(values['Mapping']) + 1)
                for j in range(1, len(temp)):
                    para_set.loc[
                        (para_set_ud[item + "_UD"] >= (temp[j - 1] - EPS)) & (para_set_ud[item + "_UD"] < (temp[j] + EPS)), item
                    ] = values['Mapping'][j - 1]

                para_set.loc[np.abs(para_set_ud[item + "_UD"] - 1) <= EPS, item] = values['Mapping'][-1]
                para_set[item] = para_set[item].round().astype(int)
            elif (values['Type'] == "categorical"):
                column_bool = [
                    item == para_name[::-1].split("DU_", maxsplit=1)[1][::-1] for para_name in self.para_ud_names]
                col_index = np.argmax(
                    para_set_ud.loc[:, column_bool].values, axis=1).tolist()
                para_set[item] = np.array(values['Mapping'])[col_index]

        return MappingData(para_set, logs_append=None)