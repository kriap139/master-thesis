from .base_search import BaseSearch
from Util import Dataset, TY_CV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import json
from sequd import SeqUD


class SeqUDSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, 
                 n_runs_per_stage=20, max_runs=100, max_search_iter=100):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir)
        self.n_runs_per_stage = n_runs_per_stage
        self.max_runs = max_runs
        self.max_search_iter = max_search_iter
    
    def _get_search_method_info(self) -> dict:
        return dict(
            n_runs_per_stage=self.n_runs_per_stage,
            max_runs=self.max_runs
        )

    def _inner_search(self, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = sequd.SeqUD(params, self.n_runs_per_stage, self.max_runs, self.max_search_iter, self.n_jobs, self._model, self.inner_cv, self.scoring, refit=True) 
        results = search.fit(x_train, y_train, **fixed_params)
        return InnerResult(results.best_params_, results.best_score_, results.best_estimator_)