from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, json_to_str, save_csv
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import inspect

class NOSearch(BaseSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # needed for compatibility with older scikit learn version on slurm cluster!
        self.useFitParamsParameter = True
        signature = inspect.signature(cross_val_score)
        # TODO: Fix this!

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        # needed for compatibility with older scikit learn version on slurm cluster!
        if self.useFitParamsParameter: 
            result = cross_val_score(self._model, x_train, y_train, n_jobs=self.n_jobs, scoring=self.scoring, cv=self.inner_cv, fit_params=fixed_params)
        else:
            result = cross_val_score(self._model, x_train, y_train, n_jobs=self.n_jobs, scoring=self.scoring, cv=self.inner_cv, params=fixed_params)
        
        model = clone(self._model).fit(x_train, y_train, **fixed_params)

        params = model.get_params()
        best_params = {k:v for (k,v) in params.items() if k in search_space}

        return InnerResult(0, best_params, result.mean(), pd.DataFrame(), model)