from .base_search import BaseSearch
from Util import Dataset, TY_CV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import json
import sequd


class SeqUD(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_path=None, 
                 n_runs_per_stage=20, max_runs=100, max_search_iter=100):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_path)
        self.result = None
        self.n_runs_per_stage = n_runs_per_stage
        self.max_runs = max_runs
        self.max_search_iter = max_search_iter

    def search(self, params: dict, fixed_params: dict) -> 'SeqUD':
        if self._save:
            self.init_save(info_append=dict(
                n_runs_per_stage=self.n_runs_per_stage,
                max_runs=self.max_runs,
                max_search_iter=self.max_search_iter
            ))
        
        if not self.train_data.has_saved_folds():
            print(f"Saveing folds for dataset {self.train_data.name}")
            self.train_data.save_folds(self.cv)
        
        folds = self.train_data.load_saved_folds()
        print("starting search")
        start = time.perf_counter()
        
        for i, (train_idx, test_idx) in enumerate(folds):

            search = sequd.SeqUD(params, self.n_runs_per_stage, self.max_runs, self.max_search_iter, self.n_jobs, 
                self._model, self.inner_cv, self.scoring, refit=True, verbose=True
            ) 
            
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