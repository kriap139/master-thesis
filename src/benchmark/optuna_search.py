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
from optuna.integration import OptunaSearchCV
from optuna.study import Study, create_study
from optuna.trial import Trial
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.samplers import TPESampler
from kspace import KSpaceStudy, infer_kspace_ver
        
class OptunaSearch(BaseSearch):
    def _create_study(self, search_space: TY_SPACE) -> Study:
        return create_study(sampler=TPESampler(), direction="maximize")
    
    def _encode_search_space(self, search_space: dict) -> dict:
        space = search_space.copy()
        for k in space.keys():
            v = space[k]
            if isinstance(v, Integer):
                space[k] = IntDistribution(v.low, v.high) # log=(v.prior == "log-uniform")
            elif isinstance(v, Real):
                space[k] = FloatDistribution(v.low, v.high) # log=(v.prior == "log-uniform")
            elif isinstance(v, Categorical):
                space[k] = CategoricalDistribution(list(v.categories))
            else:
                raise ValueError(f"search space contains unsupported type for '{k}': {type(v)}")
        return space
    
    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        if isinstance(self.scoring, Sequence) and (not isinstance(self.scoring, str)):
            scoring = self.refit
        else:
            scoring = self.scoring
        
        search = OptunaSearchCV(self._model, search_space, n_trials=self.n_iter, n_jobs=self.n_jobs, cv=self.inner_cv, scoring=scoring, study=self._create_study(search_space), refit=True)
        results = search.fit(x_train, y_train, **fixed_params)
        return InnerResult(results.best_index_, results.best_params_, results.best_score_,  results.trials_dataframe(), results.best_estimator_)

class KSpaceOptunaSearch(OptunaSearch):
    def __init__(self, k:  Union[Number, dict] = None, *args, **kwargs):
        self.k = k
        self.x_in_search_space = True
        self.kspace_ver = infer_kspace_ver(method=self.__class__.__name__)
        super().__init__(*args, **kwargs)

    def _create_save_dir(self) -> str:
        info = dict(kparams=len(self.k.keys())) if isinstance(self.k, dict) else None
        return super()._create_save_dir(info)
    
    def _create_study(self, search_space: TY_SPACE) -> KSpaceStudy:
        return KSpaceStudy.create_study(search_space, self.k, k_space_ver=self.__class__.__name__)

class KSpaceOptunaSearchV2(KSpaceOptunaSearch):
    pass

class KSpaceOptunaSearchV3(KSpaceOptunaSearch):
    pass