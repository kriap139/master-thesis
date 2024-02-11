from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import json
from sequd import SeqUD
import pandas as pd
from itertools import chain

class SeqUDSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, 
                 n_runs_per_stage=20, max_search_iter=100, save_inner_history=True, max_outer_iter: int = None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir, save_inner_history, max_outer_iter)
        self.n_runs_per_stage = n_runs_per_stage
        self.max_search_iter = max_search_iter
    
    def _get_search_method_info(self) -> dict:
        return dict(
            n_runs_per_stage=self.n_runs_per_stage,
            max_search_iter=self.max_search_iter
        )
    
    def _get_inner_history_head(self, search_space: dict) -> list:
        params = [name for name in search_space.keys()]
        params_ud = [f"{name}_UD" for name in search_space.keys()]
        head = list(chain.from_iterable([("outer_iter",), params, params_ud, ("score", "stage")]))
        return head
    
    def __update_inner_history(self, search_iter: int, clf: SeqUD):
        rows = clf.logs.to_dict(orient="records")
        for row in rows:
            row["outer_iter"] = search_iter
        save_csv(self._inner_history_fp, self.inner_history_head, rows)
    
    def _encode_search_space(self, search_space: dict) -> dict:
        space = {}
        for k, v in search_space.items():
            if isinstance(v, Real):
                space[k] = dict(Type='continuous', Range=[v.low, v.high], Wrapper=lambda x: x)
            elif isinstance(v, Integer):
                space[k] = dict(Type='integer', Mapping=tuple(range(v.low, v.high + 1)))
            elif isinstance(v, Categorical):
                space[k] = dict(Type='categorical', Mapping=list(v.categories))
            else:
                raise ValueError(f"search space contains unsupported type for '{k}': {type(v)}")
        return space

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = SeqUD(search_space, self.n_runs_per_stage, self.n_iter, self.max_search_iter, self.n_jobs, self._model, self.cv, self.scoring, refit=True, verbose=2)
        search.fit(x_train, y_train, fixed_params)

        if self.save_inner_history:
            self.__update_inner_history(search_iter, search)

        return InnerResult(search.best_index_, search.best_params_, search.best_score_, search.best_estimator_)