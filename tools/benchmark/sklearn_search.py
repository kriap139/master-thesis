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
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, save_inner_history=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir, save_inner_history)
    
    def _get_inner_history_head(self, search_space: dict) -> list:
        head = ["outer_iter"]
        head.extend([name for name, v in search_space.items()])
        head.append("train_score")
        return head
    
    def _update_inner_history(self, search_iter: int, clf: GridSearchCV):
        params = clf.cv_results_["params"].to_numpy()
        train_scores = clf.cv_results_["mean_test_score"].to_numpy()
        rows = [
            dict(outer_iter=search_iter, param=param, train_score=score) 
            for param, score in  zip(params, train_scores, strict=True)
        ]
        save_csv(self._inner_history_fp, self.inner_history_head, rows)

class RandomSearch(SklearnSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, save_inner_history=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir, save_inner_history)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = RandomizedSearchCV(self._model, search_space, n_iter=self.n_iter, scoring=self.scoring, n_jobs=self.n_jobs, cv=self.inner_cv, refit=True)
        results = search.fit(x_train, y_train, **fixed_params)

        if self.save_inner_history:
            self._update_inner_history(search_iter, search)

        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)

class GridSearch(SklearnSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None, save_inner_history=True):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir, save_inner_history)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        search = GridSearchCV(self._model, search_space, n_jobs=self.n_jobs, scoring=self.scoring, cv=self.inner_cv, refit=True)
        results = search.fit(x_train, y_train, **fixed_params)

        if self.save_inner_history:
            self._update_inner_history(search_iter, search)

        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)