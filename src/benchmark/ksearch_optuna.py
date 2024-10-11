from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv, TY_SPACE
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

class KSearchOptuna(BaseSearch):    
    def __init__(self, ksearch_iter: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ksearch_iter = ksearch_iter
        self._passed_kwargs = kwargs
        self._study = create_study(sampler=TPESampler(), direction="maximize")

        #FIXME Have this here, as it is unkown how positional arguments will affect the 
        # propegation of arguments trough inherited classes. 
        assert len(args) == 0

        if 'model' in self._passed_kwargs:
            kwargs.pop('model')
    
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
    
    def update_history(self, row: dict):
        if self.history_head is None:
            self._init_save_paths(create_dirs=True)
            self.history_head = list(data.keys())

            data = load_json(self._result_fp, default={})
            data = self._get_search_attrs(search_space, current_attrs=data)
            save_json(self._result_fp, data)

        save_csv(self._history_fp, self.history_head, row)

    def create_kspace_distributions(self, search_space: TY_SPACE) -> Dict[str, FloatDistribution]:
        names = list(search_space.keys())
        zero = np.nextafter(0, 1.0)
        return {name: FloatDistribution(zero, 1.0)}
    
    def search(self, search_space: dict, fixed_params: dict) -> 'BaseSearch':
        self.init_save(search_space, fixed_params)    
        space = self.create_optuna_kspace(search_space)

        for i in range(self.ksearch_iter):
            trial = self._study.ask(space)
            k = trial.params.copy()

            tuner = KSpaceOptunaSearchV3(k=k, model=clone(self._model), **self._passed_kwargs)
            tuner.search(search_space, fixed_params)

            result = dict(search_dir=tuner._save_dir)
            result.update({f"k_{key}": v for key, v in k.items()})
            result.update(tuner.result)

            self._study.tell(trial, result["mean_test_acc"])

            if self._save:
                self.update_history(result)

            print(f"{i}: train_score={round(result["mean_train_acc"], 4)}, test_score={round(result["mean_test_acc"], 4)}, params={json_to_str(k, indent=None)}")
        
        print(f"best_number={self._study.best_trial.number}, best_score={self._study.best_value}, best_params={self._study.best_params}")
        self._calc_result()            
        return self