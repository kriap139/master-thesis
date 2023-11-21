from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv
import lightgbm as lgb
from typing import Callable
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

        
class OptunaSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, save_inner_history=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir, save_inner_history)
    
    def _get_inner_history_head(self, search_space: dict) -> list:
        head = ["outer_iter"]
        head.extend([name for name, v in search_space.items()])
        head.append("train_score")
        return head
    
    def __update_inner_history(self, search_iter: int, clf: OptunaSearchCV):
        rows = []
        for trial in clf.study_.get_trials():
            row = trial.params.copy()
            row["outer_iter"] = search_iter
            row["train_score"] = trial.value
            rows.append(row)
        save_csv(self._inner_history_fp, self.inner_history_head, rows)
    
    def _encode_search_space(self, search_space: dict) -> dict:
        space = search_space.copy()
        for k in space.keys():
            v = space[k]
            if isinstance(v, Integer):
                space[k] = IntDistribution(v.low, v.high, log=(v.prior == "log-uniform"))
            elif isinstance(v, Real):
                space[k] = FloatDistribution(v.low, v.high, log=(v.prior == "log-uniform"))
            elif isinstance(v, Categorical):
                space[k] = CategoricalDistribution(list(v.categories))
            else:
                raise ValueError(f"search space contains unsupported type for '{k}': {type(v)}")
        return space
    
    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        study = create_study(sampler=TPESampler(), direction="maximize")
        search = OptunaSearchCV(self._model, search_space, n_trials=self.n_iter, n_jobs=self.n_jobs, cv=self.inner_cv, scoring=self.scoring, study=study, refit=True)
        results = search.fit(x_train, y_train, **fixed_params)

        if self.save_inner_history:
            self.__update_inner_history(search_iter, search)

        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)