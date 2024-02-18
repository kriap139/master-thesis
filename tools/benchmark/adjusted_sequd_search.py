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
from .sequd_search import SeqUDSearch
from pysequd import AdjustedSequd

class AdjustedSeqUDSearch(SeqUDSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, 
                 n_runs_per_stage=20, max_search_iter=100, save_inner_history=True, max_outer_iter: int = None,
                 adjust_method='linear', t=0.25, exp_step=0.18):
        super().__init__(
            model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir, 
            n_runs_per_stage, max_outer_iter, save_inner_history, max_outer_iter
        )
        
        self.t = t
        self.exp_step = exp_step
        self.adjust_method = adjust_method
    
    def _get_search_method_info(self) -> dict:
        info = super()._get_search_method_info()
        info["adjust_method"] = self.adjust_method
        info["t"] = self.t
        info["exp_step"] = self.exp_step
        return info

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = AdjustedSequd(
            search_space, self.n_runs_per_stage, self.n_iter, self.max_search_iter, self.n_jobs, self._model, self.cv, 
            self.scoring, refit=True, verbose=2, adjust_method=self.adjust_method, t=self.t, exp_step=self.exp_step
        )
        search.fit(x_train, y_train, fixed_params)

        if self.save_inner_history:
            self.__update_inner_history(search_iter, search)

        return InnerResult(search.best_index_, search.best_params_, search.best_score_, search.best_estimator_)
