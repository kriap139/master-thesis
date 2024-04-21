from .base_search import BaseSearch, InnerResult
from Util import Dataset, TY_CV, Integer, Real, Categorical, save_csv
import lightgbm as lgb
from typing import Callable, Iterable, Dict, Union
from numbers import Number
import time
import numpy as np
import json
from optuna import Trial
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator
from optuna.integration import OptunaSearchCV
from optuna.study import Study, create_study
from optuna.trial import Trial
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.samplers import TPESampler
from kspace import KSpaceStudy
        
class OptunaSearch(BaseSearch):
    def __init__(
        self, 
        model, 
        train_data: Dataset, 
        test_data: Dataset = None,
        n_iter=100, 
        n_jobs=None, 
        cv: TY_CV = None, 
        inner_cv: TY_CV = None, 
        scoring = None, 
        save=False, 
        save_inner_history=True, 
        max_outer_iter: int = None, 
        refit=True, 
        add_save_dir_info: dict = None,
        study=None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save, save_inner_history, max_outer_iter, refit, add_save_dir_info)
        self._study = study
    
    def __update_inner_history(self, search_iter: int, clf: OptunaSearchCV):
        df = clf.trials_dataframe()
        df["outer_iter"] = search_iter
        head = list(df.columns)
        save_csv(self._inner_history_fp, head, df.to_dict(orient="records"))
    
    def _encode_search_space(self, search_space: dict) -> dict:
        space = search_space.copy()
        for k in space.keys():
            v = space[k]
            if isinstance(v, Integer):
                space[k] = IntDistribution(v.low, v.high) # log=(v.prior == "log-uniform")
            elif isinstance(v, Real):
                space[k] = FloatDistribution(v.low, v.high) # log=(v.prior == "log-uniform")
            elif isinstance(v, Categorical):
                space[k] = CategoricalDistribution(list(v.categories))
            else:
                raise ValueError(f"search space contains unsupported type for '{k}': {type(v)}")
        return space
    
    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        if self._study is None:
            study = create_study(sampler=TPESampler(), direction="maximize")
        else:
            study = self._study

        search = OptunaSearchCV(self._model, search_space, n_trials=self.n_iter, n_jobs=self.n_jobs, cv=self.inner_cv, scoring=self.scoring, study=study, refit=True)
        results = search.fit(x_train, y_train, **fixed_params)
        if self.save_inner_history:
            self.__update_inner_history(search_iter, search)
        return InnerResult(results.best_index_, results.best_params_, results.best_score_, results.best_estimator_)

class KSpaceOptunaSearch(OptunaSearch):
    def __init__(
            self, 
            model, 
            train_data: Dataset, 
            test_data: Dataset = None,
            n_iter=100, 
            n_jobs=None, 
            cv: TY_CV = None, 
            inner_cv: TY_CV = None, 
            scoring = None, 
            save=False, 
            save_inner_history=True, 
            max_outer_iter: int = None, 
            refit=True, 
            add_save_dir_info: dict = None,
            k:  Union[Number, dict] = None
        ):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, False, save_inner_history, max_outer_iter, refit, add_save_dir_info, study=None)
        self.k = k
        self.x_in_search_space = True
        self._pre_init_save(save)
        self._inner_func = self._init_study

    def _create_save_dir(self) -> str:
        if isinstance(self.k, Number):
            info = None
        elif isinstance(self.k, dict):
            info = dict(kparams=len(self.k.keys()))
        return super()._create_save_dir(info)
    
    def _get_search_method_info(self) -> dict:
        info = super()._get_search_method_info()
        info["k"] = self.k
        info["x_in_search_space"] = self.x_in_search_space
        return info
    
    def _init_study(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict, k_space_ver: int = 1) -> InnerResult:
        self._study = KSpaceStudy.create_study(search_space, self.k, k_space_ver=k_space_ver)
        self._inner_func = super()._inner_search
        return super()._inner_search(search_iter, x_train, y_train, search_space, fixed_params)

    def _inner_search(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        return self._inner_func(search_iter, x_train, y_train, search_space, fixed_params)

class KSpaceOptunaSearchV2(KSpaceOptunaSearch):
    def _init_study(self, search_iter: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict, k_space_ver: int = 1) -> InnerResult:
        return super()._init_study(search_iter, x_train, y_train, search_space, fixed_params, k_space_ver=2)