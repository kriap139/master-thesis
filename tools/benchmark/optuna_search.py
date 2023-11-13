from .base_search import BaseSearch
from Util import Dataset, TY_CV
import lightgbm as lgb
from typing import Callable
import time
import numpy as np
import json
import sequd
from ..Util import Integer, Real
from optuna import Trial
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator
from optuna.integration import LightGBMTunerCV
from optuna.distributions import FloatDistribution, IntDistribution, IntLogUniformDistribution, IntUniformDistribution



def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy
        
class OPTUNA(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_path=None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_path)
        self.result = None

    def search(self, params: dict, fixed_params: dict) -> 'OPTUNA':
        if self._save:
            self.init_save(info_append=dict(
                n_runs_per_stage=self.n_runs_per_stage,
                max_runs=self.max_runs,
                max_search_iter=self.max_search_iter
            ))
        
        if not self.train_data.has_saved_folds():
            print(f"Saveing folds for dataset {self.train_data.name}")
            self.train_data.save_folds(self.cv)

        for k in _params.keys():
            v = _params[k]
            if isinstance(v, Integer):
                _params[k] = IntDistribution(v.name, v.low, v.high, log=(v.prior == "log-uniform"))
            elif isinstance(v, Real):
                _params[k] = FloatDistribution(v.name, v.low, v.high, log=(v.prior == "log-uniform")) 

        folds = self.train_data.load_saved_folds()
        print("starting search")
        start = time.perf_counter()
        
        for i, (train_idx, test_idx) in enumerate(folds):
            x_train, x_test = self.train_data.x.iloc[train_idx, :], self.train_data.x.iloc[test_idx, :]
            y_train, y_test = self.train_data.y[train_idx], self.train_data.y[test_idx]

            
            
            best = search.best_estimator_

            acc = self.score(best, x_test, y_test)
            self._add_iteration_stats(search.best_params_, search.best_score_, acc)

            print(f"{i}: best_score={round(search.best_score_, 4)}, test_score={round(acc, 4)}, params={json.dumps(search.best_params_)}")

        end = start - time.perf_counter()
        self._set_result(end)
        self._flush_iterations_stats(update_keys=dict(result=self.result), clear_cache=False)

        return self