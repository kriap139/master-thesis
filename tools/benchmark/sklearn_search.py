from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, json_to_str, save_csv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import pandas as pd

class SklearnSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save=False, save_inner_history=True, verbose=2, max_outer_iter: int = None, refit=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save, save_inner_history, max_outer_iter, refit)
        self.verbose = verbose
    
    def _update_inner_history(self, search_iter: int, clf: GridSearchCV):
        result = pd.DataFrame(clf.cv_results_)
        result["outer_iter"] = search_iter
        head = list(result.columns)
        save_csv(self._inner_history_fp, head, rows)

class RandomSearch(SklearnSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save=False, save_inner_history=True, verbose = 0, max_outer_iter: int = None, refit=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save, save_inner_history, verbose, max_outer_iter, refit)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = RandomizedSearchCV(self._model, search_space, n_iter=self.n_iter, scoring=self.scoring, n_jobs=self.n_jobs, cv=self.inner_cv, refit=self.refit, verbose=self.verbose)
        results = search.fit(x_train, y_train, **fixed_params)

        if self.save_inner_history:
            self._update_inner_history(search_iter, search)

        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)

class GridSearch(SklearnSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save=False, save_inner_history=True, verbose = 0, max_outer_iter: int = None, refit=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save, save_inner_history, verbose, max_outer_iter, refit)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = GridSearchCV(self._model, search_space, n_jobs=self.n_jobs, scoring=self.scoring, cv=self.inner_cv, refit=self.refit, verbose=self.verbose)
        results = search.fit(x_train, y_train, **fixed_params)

        if self.save_inner_history:
            self._update_inner_history(search_iter, search)

        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)