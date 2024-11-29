from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv, TY_SPACE, load_csv, load_json, save_json, json_to_str
from Util.compat import removeprefix
from typing import Callable, Iterable, Dict, Union, List
import time
import numpy as np
from sequd import SeqUD
import pandas as pd
from itertools import chain
from numbers import Number
from .optuna_search import BaseSearch, InnerResult, KSpaceOptunaSearchV3
from .kspace_random import KSpaceRandomSearchV3
from optuna.distributions import FloatDistribution
from optuna.study import Study, create_study
from optuna.trial import Trial, create_trial
from sklearn.base import clone
from optuna.samplers import TPESampler
import os
import logging
import sys
import gc

class KSearchOptuna(BaseSearch):    
    def __init__(self, ksearch_iter: int = 100, search_method=None, *args, **kwargs):
        self.search_method = search_method
        self.ksearch_iter = ksearch_iter
        self._passed_kwargs = kwargs
        self._study = create_study(sampler=TPESampler(), direction="maximize")
        self._iter = range(self.ksearch_iter)
        self._searches_dir = None
        
        #FIXME Have this here, as it is unknown how positional arguments will affect the 
        # propagation of arguments trough inherited classes. 
        assert len(args) == 0

        if self.search_method:
            this_module = sys.modules[__name__]
            self._method = getattr(this_module, self.search_method, None)
            if self._method is None:
                raise ValueError(f"Unable to find selected search_method: {self.search_method}")
            if 'kspace' not in self._method.__name__.lower():
                raise ValueError(f"Selected search method not a kspace method: {self.search_method}")
            logging.info(f"Using search method {self._method.__name__}")
        else:
            self._method = KSpaceOptunaSearchV3

        super().__init__(*args, **kwargs)
        
        if 'model' in self._passed_kwargs:
            kwargs.pop('model')
        if self._resume:
            self._resume_search()

    def _create_save_dir(self) -> str:
        method = self._method.__name__ 
        method = removeprefix(method, "KSpace")
        method = method.split("Search", maxsplit=1)[0]
        info = dict(method=method)
        return super()._create_save_dir(info)
    
    def _set_history_head(self, search_space: dict):
        # Is set after first history update 
        self.history_head = None
    
    def _resume_search(self):
        data = load_csv(self._history_fp)
        self.history_head = list(data.columns)

        rows = data.shape[0]
        delta = rows - self.ksearch_iter

        if delta >= rows:
            self._iter = range(rows, delta)
        else:
            self._iter = range(rows, rows + self.ksearch_iter)
        
        self._searches_dir = os.path.join(self._save_dir, "searches")
        assert os.path.exists(self._searches_dir)

        # Adding previous trials!
        params = [param[len("k_"):] for param in self.history_head if param.startswith("k_")]

        for row in data.iterrows():
            trial = create_trial(
                params={param: row[param] for param in params},
                distributions=self.create_kspace_distributions(params),
                value=row["mean_test_acc"]
            )
            self._study.add_trial(trial)
    
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
        self._searches_dir = os.path.join(self._save_dir, "searches")
        os.makedirs(self._searches_dir, exist_ok=True)
        super().init_save(search_space)
    
    def update_history(self, row: dict):
        if self.history_head is None:
            self.history_head = list(row.keys())
            save_csv(self._history_fp, self.history_head)
        save_csv(self._history_fp, self.history_head, row)

    def create_kspace_distributions(self, search_space: Union[TY_SPACE, List[str]]) -> Dict[str, FloatDistribution]:
        if type(search_space) != list:
            names = list(search_space.keys())
        else:
            names = search_space

        zero = np.nextafter(0, 1.0)
        return {name: FloatDistribution(zero, 1.0) for name in names}
    
    def search(self, search_space: dict, fixed_params: dict) -> 'BaseSearch':
        if self._save:
            self.init_save(search_space)
        
        space = self.create_kspace_distributions(search_space)

        for i in self._iter:
            trial = self._study.ask(space)
            k = trial.params.copy()

            tuner = self._method(k=k, model=clone(self._model), **self._passed_kwargs, root_dir=self._searches_dir)
            tuner.search(search_space, fixed_params)

            search_dir = tuner._save_dir.split(os.path.sep)[-1]
            result = dict(search_dir=search_dir)
            result.update({f"k_{key}": v for key, v in k.items()})
            result.update(tuner.result)

            self._study.tell(trial, result["mean_test_acc"])

            if self._save:
                self.update_history(result)

            print(f"{i}: train_score={round(result['mean_train_acc'], 4)}, test_score={round(result['mean_test_acc'], 4)}, params={json_to_str(k, indent=None)}", flush=True)
            
            del tuner
            gc.collect()
        
        print(f"best_number={self._study.best_trial.number}, best_score={self._study.best_value}, best_params={self._study.best_params}")
        self._calc_result()            
        return self