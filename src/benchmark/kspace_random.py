from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv, TY_SPACE
import lightgbm as lgb
from typing import Callable, Iterable, Dict, Union, Sequence
from numbers import Number
import time
import numpy as np
import json
from optuna import Trial
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator
from kspace import KSpaceRandom
        
class KSpaceRandomSearchV3(BaseSearch):
    def __init__(self, k:  Union[Number, dict] = None, *args, **kwargs):
        self.k = k
        self.x_in_search_space = True
        self.kspace_ver = 3
        super().__init__(*args, **kwargs)

    def _create_save_dir(self) -> str:
        info = dict(kparams=len(self.k.keys())) if isinstance(self.k, dict) else None
        return super()._create_save_dir(info)
    
    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = KSpaceRandom(self._model, search_space, n_iter=self.n_iter, scoring=self.scoring, n_jobs=self.n_jobs, cv=self.inner_cv, refit=self.refit)
        search.set_k(self.k)
        results = search.fit(x_train, y_train, **fixed_params)
        search.process_results()
        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.cv_results_, results.best_estimator_)