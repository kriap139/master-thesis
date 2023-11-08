from .base_search import BaseSearch
from Util import Dataset, TY_CV
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import json

class RandomSearch(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_path=None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_path)
        self.result = None

    def search(self, params: dict, fixed_params: dict) -> 'RandomSearch':
        if self._save:
            self.init_save()
        
        if not self.train_data.has_saved_folds():
            print(f"Saveing folds for dataset {self.train_data.name}")
            self.train_data.save_folds(self.cv)
        
        folds = self.train_data.load_saved_folds()
        print("starting search")
        start = time.perf_counter()
        
        for i, (train_idx, test_idx) in enumerate(folds):
            search = RandomizedSearchCV(self._model, params, n_iter=self.n_iter, n_jobs=self.n_jobs, cv=self.inner_cv, refit=True, error_score='raise')
            
            x_train, x_test = self.train_data.x.iloc[train_idx, :], self.train_data.x.iloc[test_idx, :]
            y_train, y_test = self.train_data.y[train_idx], self.train_data.y[test_idx]

            results = search.fit(x_train, y_train, **fixed_params)
            best = search.best_estimator_

            acc = self.score(best, x_test, y_test)
            self._add_iteration_stats(search.best_params_, search.best_score_, acc)

            print(f"{i}: best_score={round(search.best_score_, 4)}, test_score={round(acc, 4)}, params={json.dumps(search.best_params_)}")

        end = start - time.perf_counter()
        self._set_result(end)
        self._flush_iterations_stats(update_keys=dict(result=self.result), clear_cache=False)

        return self