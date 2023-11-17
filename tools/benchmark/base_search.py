from Util import Dataset, find_dir_ver, load_csv, save_csv, Integer, Real, Categorical, find_file_ver
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold
from sklearn.metrics import get_scorer, get_scorer_names
from typing import Sequence, Union, Optional, Iterable
from Util.dataset import Dataset, Builtin, Task, TY_CV
from Util.io_util import load_json, save_json, data_dir, json_to_str
import gc
from typing import Callable
import os
from datetime import datetime
import logging
from dataclasses import dataclass
import time

class InnerResult:
    def __init__(self, best_params: dict, best_score: float, best_model):
        self.best_params = best_params
        self.best_score = best_score
        self.best_model = best_model


class BaseSearch:

    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring=None, save_dir=None):
        self.train_data = train_data
        self.test_data = test_data
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.cv = cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self._model = model

        self._save_dir = save_dir
        self._result_fp = None
        self._history_fp = None
        self._models_dir = None
        self.history_head = None

        if self._save_dir is None:
            raise RuntimeError("save_dir argument is required!")
        else:
            log = os.path.exists(self._save_dir)
            self.__init_save_paths()
            if log:
                logging.debug(f"Save directory already exists, saving to alternative directory: {self._save_dir}")

        self._result = None
    
    def __init_save_paths(self):
        if os.path.exists(self._save_dir):
            self._save_dir = find_dir_ver(self._save_dir)
        self._history_fp = os.path.join(self._save_dir, "history.csv")
        self._result_fp = os.path.join(self._save_dir, "result.json")
        self._models_dir = os.path.join(self._save_dir, "models")
    
    def init_save(self, search_space: dict, exstra_keys: dict = None, info_append: dict = None):
        # os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._models_dir, exist_ok=True)
        data = load_json(self._result_fp, default={})

        if hasattr(self._model, "__qualname__"):
            model = self._model.__qualname__
        else:
            model = self._model.__class__.__qualname__

        data = dict(
            info=dict(
                model=model,
                dataset=self.train_data.name,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                cv=Dataset.get_cv_info(self.cv) if self.cv is not None else None,
                inner_cv=Dataset.get_cv_info(self.inner_cv) if self.inner_cv is not None else None,
                method_params=self._get_search_method_info()
            )
        )

        if callable(self.scoring):
            data["scoring"] = self.scoring.__name__
        else:
            data["scoring"] = self.scoring
        save_json(self._result_fp, data)

        self.history_head = [name for name, v in search_space.items()]
        self.history_head.extend(("train_score", "test_score", "time"))
        save_csv(self._history_fp, self.history_head)


    def update_history(self, params: dict, train_score: float, test_score: float, time: float):
        row = params.copy()
        row["train_score"] = train_score
        row["test_score"] = test_score
        row["time"] = time
        save_csv(self._history_fp, self.history_head, row)
    
    def save_model(self, model, outer_id: int = None, inner_id: int = None):
        if inner_id is None and outer_id is None:
            raise RuntimeError(f"Both outer_id and inner_id cant't be None")
        elif inner_id is None:
            name = f"model_outer{outer_id}.txt"
        elif outer_id is None:
            name = f"model_inner{inner_id}.txt"
        else:
            name = f"model_inner{inner_id}_outer{outer_id}.txt"

        fp = os.path.join(self._models_dir, name)
        if os.path.exists(fp):
            fp = find_file_ver(self._models_dir, name)
            logging.warn(f"Selected model save name ({name})  exists, using new name: {os.path.basename(fp)}")

        if hasattr(model, "save_model"):
            model.save_model(fp)
        elif hasattr(model, "booster_") and hasattr(model.booster_, "save_model"):
            model.booster_.save_model(fp)
        else:
            raise RuntimeError(f"Unable to save model ({type(model)}) without a save_model method!")
                
    
    def _calc_result(self):
        data = load_csv(self._history_fp)

        train_scores = data["train_score"]
        test_scores = data["test_score"]
        times = data["time"]
        time = np.sum(times)
        
        self.result = dict(
            mean_train_acc=np.mean(train_scores),
            std_train_acc=np.std(train_scores),
            mean_test_acc=np.mean(test_scores),
            std_test_acc=np.std(test_scores),
            time=time,
            mean_fold_time=np.mean(times),
            std_fold_time=np.std(time)
        )
        _data = load_json(self._result_fp, default={})
        _data["result"] = self.result
        save_json(self._result_fp, _data, overwrite=True)
    
    def score(self, model, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        if self.scoring is not None:
            scorer = get_scorer(self.scoring)
            return scorer(model, x_test, y_test)
        elif hasattr(model, "score"):
            return model.score(x_test, y_test)
        else:
            raise ValueError("No valid scoring function prestent")

    def test_model(self, params: dict, fixed_params: dict):
        if self.train_data.has_test_set():
            if self.test_data is None:
                test_data = self.train_data.load_test_dataset()
        else:
            return
        model = lgb.LGBMClassifier(**params, **fixed_params)
        model.fit(X=self.train_data.x, y=self.train_data.y, categorical_feature=self.train_data.cat_features)
        return self.score(model, self.test_data.x, self.test_data.y)
    
    def _get_search_method_info(self) -> dict:
        return {}
    
    def _inner_search(self, search_id: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        raise RuntimeError("Unimplemented")
    
    def search(self, search_space: dict, fixed_params: dict) -> 'BaseSearch':
        self.init_save(search_space)
        
        if not self.train_data.has_saved_folds():
            print(f"Saveing folds for dataset {self.train_data.name}")
            self.train_data.save_folds(self.cv)
        
        folds = self.train_data.load_saved_folds()
        print("starting search")
        start = 0

        for i, (train_idx, test_idx) in enumerate(folds):
            start = time.perf_counter()

            x_train, x_test = self.train_data.x.iloc[train_idx, :], self.train_data.x.iloc[test_idx, :]
            y_train, y_test = self.train_data.y[train_idx], self.train_data.y[test_idx]

            result = self._inner_search(i, x_train, y_train, search_space, fixed_params)
            acc = self.score(result.best_model, x_test, y_test)
            end = time.perf_counter() - start

            self.update_history(result.best_params, result.best_score, acc, end)
            self.save_model(result.best_model, outer_id=i)
            print(f"{i}: best_score={round(result.best_score, 4)}, test_score={round(acc, 4)}, params={json_to_str(result.best_params, indent=None)}")

        end = time.perf_counter() - start
        self._calc_result(end)
        return self


