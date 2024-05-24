from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv
import lightgbm as lgb
from typing import Callable, Iterable
import time
import numpy as np
import json
from sequd import SeqUD
import pandas as pd
from itertools import chain
from kspace import KSpaceSeqUD
from numbers import Number

class SeqUDSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save=False, 
                 n_runs_per_stage=20, max_search_iter=100, save_inner_history=True, max_outer_iter: int = None, refit=True, add_save_dir_info: dict = None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save, save_inner_history, max_outer_iter, refit, add_save_dir_info)
        self.n_runs_per_stage = n_runs_per_stage
        self.max_search_iter = max_search_iter
    
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
        search = SeqUD(search_space, self.n_runs_per_stage, self.n_iter, self.max_search_iter, self.n_jobs, self._model, self.cv, self.scoring, refit=self.refit, verbose=2)
        search.fit(x_train, y_train, **fixed_params)
        return InnerResult(search.best_index_, search.best_params_, search.best_score_, search.logs, search.best_estimator_)


class KSpaceSeqUDSearch(SeqUDSearch):
    def __init__(
            self, 
            model, 
            train_data: Dataset, 
            test_data: Dataset = None,
            n_iter=100, 
            n_jobs=None, 
            cv: TY_CV = None, 
            inner_cv: TY_CV = None, 
            scoring = None, 
            save=False, 
            n_runs_per_stage=20, 
            max_search_iter=100, 
            save_inner_history=True, 
            max_outer_iter: int = None,
            refit=True,
            add_save_dir_info: dict = None,
            k=0
        ):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save, n_runs_per_stage, max_search_iter, save_inner_history, max_outer_iter, refit, add_save_dir_info)
        self.k = k
        self.x_in_search_space = False
        self.error_score = 'raise'
    
    def _create_save_dir(self) -> str:
        if isinstance(self.k, Number):
            k_mask = self.k
            k_params = None
        elif isinstance(self.k, dict):
            k_mask = sum(self.k.values())
            k_params = len(self.k.keys())

        info = dict(kmask=k_mask)
        if k_params is not None:
            info['kparams'] = k_params
        return super()._create_save_dir(info)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = KSpaceSeqUD(
            search_space, self.n_runs_per_stage, self.n_iter, self.max_search_iter, self.n_jobs, self._model, self.cv, 
            self.scoring, refit=self.refit, verbose=2, k=self.k, error_score=self.error_score
        )
        search.fit(x_train, y_train, **fixed_params)
        return InnerResult(search.best_index_, search.best_params_, search.best_score_, search.logs, search.best_estimator_)
