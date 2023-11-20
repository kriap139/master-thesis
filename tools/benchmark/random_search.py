from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, json_to_str
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import pandas as pd

class RandomSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir)

    def _inner_search(self, search_id: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = RandomizedSearchCV(self._model, search_space, n_iter=self.n_iter, n_jobs=self.n_jobs, cv=self.inner_cv, refit=True)
        results = search.fit(x_train, y_train, **fixed_params)
        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)