from Util import Dataset, find_dir_ver, load_csv, save_csv, Integer, Real, Categorical, find_file_ver
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold
from sklearn.metrics import get_scorer, get_scorer_names
from typing import Sequence, Union, Optional, Iterable, List
from Util.dataset import Dataset, Builtin, Task, TY_CV
from Util.io_util import load_json, save_json, data_dir, json_to_str
import gc
from typing import Callable
import os
from datetime import datetime
import logging
from dataclasses import dataclass
import time
from scipy.stats import mode
import sys
from scipy.sparse import coo_matrix

class InnerResult:
    def __init__(self, best_index: int, best_params: dict, best_score: float, best_model):
        self.best_params = best_params
        self.best_score = best_score
        self.best_model = best_model
        self.best_index = best_index

class BaseSearch:
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring=None, save_dir=None, save_inner_history=True, max_outer_iter: int = None):
        self.train_data = train_data
        self.test_data = test_data
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.max_outer_iter = max_outer_iter
        self.cv = cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self._model = model

        self._save_dir = save_dir
        self._result_fp = None
        self._history_fp = None
        self._inner_history_fp = None
        self._models_dir = None
        self.history_head = None
        self.inner_history_head = None
        self.save_inner_history = save_inner_history

        if self.max_outer_iter is None:
            self.max_outer_iter = sys.maxsize

        if self._save_dir is None:
            raise RuntimeError("save_dir argument is required!")
        else:
            log = os.path.exists(self._save_dir)
            self.__init_save_paths()
            if log:
                logging.debug(f"Save directory already exists, saving to alternative directory: {self._save_dir}")

        self.result = None
    
    def __init_save_paths(self):
        if os.path.exists(self._save_dir):
            self._save_dir = find_dir_ver(self._save_dir)
        self._history_fp = os.path.join(self._save_dir, "history.csv")
        self._inner_history_fp = os.path.join(self._save_dir, "inner_history.csv")
        self._result_fp = os.path.join(self._save_dir, "result.json")
        self._models_dir = os.path.join(self._save_dir, "models")
    
    def init_save(self, search_space: dict, fixed_params: dict):
        # os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._models_dir, exist_ok=True)
        data = load_json(self._result_fp, default={})

        if hasattr(self._model, "__qualname__"):
            name = self._model.__qualname__
        else:
            name = self._model.__class__.__qualname__

        params = self._model.get_params() if hasattr(self._model, "get_params") else None

        data = dict(
            info=dict(
                model=dict(
                    name=name,
                    params=params
                ),
                space=search_space, 
                fixed_params=fixed_params,
                dataset=self.train_data.name,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                max_outer_iter=self.max_outer_iter if (self.max_outer_iter != sys.maxsize) else None,
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

        self.history_head = ["inner_index"]
        self.history_head.extend([name for name, v in search_space.items()])
        self.history_head.extend(("train_score", "test_score", "time"))
        save_csv(self._history_fp, self.history_head)

        if self.save_inner_history:
            self.inner_history_head = self._get_inner_history_head(search_space)
            save_csv(self._inner_history_fp, self.inner_history_head)
    
    def _get_inner_history_head(self, search_space: dict) -> list:
        return []
    
    def _update_info(self, update_keys: dict):
         data = load_json(self._result_fp, default=dict(info={}))
         data["info"].update(update_keys)
         save_json(self._result_fp, data)

    def update_history(self, inner_index: int, params: dict, train_score: float, test_score: float, time: float):
        row = dict(inner_index=inner_index, train_score=train_score, test_score=test_score, time=time)
        row.update(params)
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
    
    @classmethod
    def time_to_str(cls, secs: float) -> str:
        secs = int(secs)
        days, r = divmod(secs, 86400)
        hours, r = divmod(r, 3600)
        minutes, secs = divmod(r, 60)
        if days == 0:
            return "{:02}:{:02}:{:02}".format(hours, minutes, secs)
        else:
            return "{:02}:{:02}:{:02}:{:02}".format(days, hours, minutes, secs)
    
    @classmethod
    def recalc_results(cls, result_dir: str) -> dict:
        search = BaseSearch(None, None, save_dir=os.path.join(result_dir, "dummy"))
        search._history_fp = os.path.join(result_dir, "history.csv")
        search._result_fp = os.path.join(result_dir, "results.tmp.json")
        search._calc_result()

        _data = load_json(os.path.join(result_dir, "result.json"), default={})
        _data["result"] = search.result
        return dict(result=search.result)

    def _calc_result(self):
        data = load_csv(self._history_fp)

        train_scores = data["train_score"]
        test_scores = data["test_score"]
        times = data["time"]

        time = np.sum(times)
        mean_fold_time=np.mean(times)
        std_fold_time=np.std(times)
        
        self.result = dict(
            mean_train_acc=np.mean(train_scores),
            std_train_acc=np.std(train_scores),
            min_train_acc=np.min(train_scores),
            max_train_acc=np.max(train_scores),
            mode_train_acc=mode(train_scores, keepdims=True)[0][0],
            meadian_train_acc=np.median(train_scores),
            mean_test_acc=np.mean(test_scores),
            std_test_acc=np.std(test_scores),
            min_test_acc=np.min(test_scores),
            max_test_acc=np.max(test_scores),
            mode_test_acc=mode(test_scores, keepdims=True)[0][0],
            meadian_test_acc=np.median(test_scores),
            time=time,
            mean_fold_time=mean_fold_time,
            std_fold_time=std_fold_time,
            time_str=self.time_to_str(time),
            mean_fold_time_str=self.time_to_str(mean_fold_time),
            std_fold_time_str=self.time_to_str(std_fold_time)
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
            raise ValueError("No valid scoring function present")

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
    
    def _encode_search_space(self, search_space: dict) -> dict:
        return search_space
    
    def _inner_search(self, search_id: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        raise RuntimeError("Unimplemented")
    
    def search(self, search_space: dict, fixed_params: dict) -> 'BaseSearch':
        self.init_save(search_space, fixed_params)
        
        if not self.train_data.has_saved_folds():
            print(f"Saveing folds for dataset {self.train_data.name}")
            self.train_data.save_folds(self.cv)
        
        data = self.train_data.load_saved_folds_file()
        folds = data["folds"]
        assert data["info"] == Dataset.get_cv_info(self.cv)
        
        search_space = self._encode_search_space(search_space)
        is_sparse = self.train_data.x.dtypes.apply(pd.api.types.is_sparse).all()
        print("starting search")

        if is_sparse:
            train_x: coo_matrix = self.train_data.x.sparse.to_coo()
            train_x = train_data.tocsr()
        else:
            train_x = None
        
        for i, (train_idx, test_idx) in enumerate(folds):
            if i > self.max_outer_iter:
                print(f"Max number of outer fold iterations reached ({self.max_outer_iter}), terminating!")
                break

            start = time.perf_counter()

            if is_sparse:
                x_train, x_test = train_x[train_idx, :], train_x[test_idx, :]
            else:
                x_train, x_test = self.train_data.x.iloc[train_idx, :], self.train_data.x.iloc[test_idx, :]
            
            y_train, y_test = self.train_data.y[train_idx], self.train_data.y[test_idx]

            result = self._inner_search(i, x_train, y_train, search_space, fixed_params)
            acc = self.score(result.best_model, x_test, y_test)
            end = time.perf_counter() - start

            self.update_history(result.best_index, result.best_params, result.best_score, acc, end)
            self.save_model(result.best_model, outer_id=i)
            print(f"{i}: best_score={round(result.best_score, 4)}, test_score={round(acc, 4)}, params={json_to_str(result.best_params, indent=None)}")

        self._calc_result()
        return self


