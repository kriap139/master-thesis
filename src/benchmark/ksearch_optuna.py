from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv, TY_SPACE, load_csv, load_json
from typing import Callable, Iterable, Dict
import time
import numpy as np
from sequd import SeqUD
import pandas as pd
from itertools import chain
from numbers import Number
from .optuna_search import BaseSearch, InnerResult, KSpaceOptunaSearchV3
from optuna.distributions import FloatDistribution
from optuna.study import Study, create_study
from sklearn.base import clone
from optuna.samplers import TPESampler
import os
import logging
import sys

class KSearchOptuna(BaseSearch):    
    def __init__(self, ksearch_iter: int = 100, resume_dir: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs, save_dir=resume_dir)

        self.ksearch_iter = ksearch_iter
        self._passed_kwargs = kwargs
        self._study = create_study(sampler=TPESampler(), direction="maximize")
        self._iter = range(self.ksearch_iter)
        self._searches_dir = None

        #FIXME Have this here, as it is unkown how positional arguments will affect the 
        # propegation of arguments trough inherited classes. 
        assert len(args) == 0

        if 'model' in self._passed_kwargs:
            kwargs.pop('model')

        if resume_dir is not None:
            self._resume_search(resume_dir)
    
    def _resume_search(self, resume_dir: str):
        if os.path.exists(resume_dir):
            data = load_csv(resume_dir)
            self.history_head = list(data.columns)

            rows = data.shape[0]
            delta = rows - self.ksearch_iter

            if delta >= rows:
                self._iter = range(rows, delta)
            else:
                self._iter = range(rows, rows + self.ksearch_iter)
            
            self._searches_dir = os.path.join(self._save_dir, "searches")
            assert os.path.exists(self._searches_dir)
        else:
            raise ValueError(f"resume_dir doesn't exist: {resume_dir}")
                
    
    def _calc_result(self):
        self.result = dict(
            best_number=self._study.best_trial.number,
            best_score=self._study.best_value,
            best_params=self._study.best_params
        )

        if self._save:
            _data = load_json(self._result_fp, default={})
            _data["result"] = self.result
            save_json(self._result_fp, _data, overwrite=True)
    
    def init_save(self, search_space: dict):
        self._init_save_paths(create_dirs=True)

        self._searches_dir = os.path.join(self._save_dir, "searches")
        os.makedirs(self._searches_dir, exist_ok=True)

        data = load_json(self._result_fp, default={})
        data = self._get_search_attrs(search_space, current_attrs=data)

        save_json(self._result_fp, data)
    
    def update_history(self, row: dict):
        if self.history_head is None:
            self.history_head = list(data.keys())
            save_csv(self._history_fp, self.history_head)
        save_csv(self._history_fp, self.history_head, row)

    def create_kspace_distributions(self, search_space: TY_SPACE) -> Dict[str, FloatDistribution]:
        names = list(search_space.keys())
        zero = np.nextafter(0, 1.0)
        return {name: FloatDistribution(zero, 1.0) for name in names}
    
    def search(self, search_space: dict, fixed_params: dict) -> 'BaseSearch':
        if self._save:
            self.init_save(search_space)

        space = self.create_kspace_distributions(search_space)

        for i in self._iter:
            trial = self._study.ask(space)
            k = trial.params.copy()

            tuner = KSpaceOptunaSearchV3(k=k, model=clone(self._model), **self._passed_kwargs, root_dir=self._searches_dir)
            tuner.search(search_space, fixed_params)

            result = dict(search_dir=tuner._save_dir)
            result.update({f"k_{key}": v for key, v in k.items()})
            result.update(tuner.result)

            self._study.tell(trial, result["mean_test_acc"])

            if self._save:
                self.update_history(result)

            print(f"{i}: train_score={round(result['mean_train_acc'], 4)}, test_score={round(result['mean_test_acc'], 4)}, params={json_to_str(k, indent=None)}")
        
        print(f"best_number={self._study.best_trial.number}, best_score={self._study.best_value}, best_params={self._study.best_params}")
        self._calc_result()            
        return self