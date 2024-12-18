from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, json_to_str, save_csv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import pandas as pd

class RandomSearch(BaseSearch):
    def __init__(self, verbose=0, *args, **kwargs):
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = RandomizedSearchCV(self._model, search_space, n_iter=self.n_iter, scoring=self.scoring, n_jobs=self.n_jobs, cv=self.inner_cv, refit=self.refit, verbose=self.verbose)
        results = search.fit(x_train, y_train, **fixed_params)
        return InnerResult(results.best_index_, results.best_params_, results.best_score_, pd.DataFrame(results.cv_results_), results.best_estimator_)

class GridSearch(BaseSearch):
    def __init__(self, verbose=0, *args, **kwargs):
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = GridSearchCV(self._model, search_space, n_jobs=self.n_jobs, scoring=self.scoring, cv=self.inner_cv, refit=self.refit, verbose=self.verbose)
        results = search.fit(x_train, y_train, **fixed_params)
        return InnerResult(results.best_index_, results.best_params_, results.best_score_, pd.DataFrame(results.cv_results_), results.best_estimator_)