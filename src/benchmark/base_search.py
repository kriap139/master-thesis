from Util import Dataset, find_dir_ver, load_csv, save_csv, Integer, Real, Categorical, find_file_ver, CVInfo
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
import shutil
from itertools import chain

class InnerResult:
    def __init__(self, best_index: int, best_params: dict, best_score: float, inner_history: pd.DataFrame, best_model):
        self.best_params = best_params
        self.best_score = best_score
        self.best_model = best_model
        self.best_index = best_index
        self.inner_history = inner_history

class BaseSearch:
    def __init__(
        self, model, train_data: Dataset, test_data: Dataset = None, n_iter=100, n_jobs=None, cv: TY_CV = None, 
        inner_cv: TY_CV = None, scoring=None, save=False, save_inner_history=True, max_outer_iter: int = None, refit=True, 
        add_save_dir_info: dict = None, save_best_models=False, save_dir: str = None, root_dir: str = None):

        self.train_data = train_data
        self.test_data = test_data
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.max_outer_iter = max_outer_iter
        self.cv = cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self._model = model
        self.refit = refit

        self.add_save_dir_info = add_save_dir_info
        self.history_head = None
        self.save_inner_history = save_inner_history
        self.save_best_models = save_best_models
        self._save = save

        self._result_fp = None
        self._history_fp = None
        self._inner_history_fp = None
        self._models_dir = None
        self._save_dir = None
        self._root_dir = root_dir
        self._passed_dir = save_dir

        self.result = None

        if self.max_outer_iter is None:
            self.max_outer_iter = sys.maxsize
        if not self._save:
            self.save_inner_history = False
            
        if self._save:
            self._init_save_paths(create_dirs=False)
    
    def _create_save_dir(self, info: dict = None) -> str:
        if self._passed_dir is not None:
            if os.path.exists(self._save_dir):
                return self._save_dir
            else:
                raise ValueError(f"Passed save directory doesn't exist: {self._save_dir}")

        if self.add_save_dir_info is not None:
            if info is None:
                info = self.add_save_dir_info
            else:
                _info = self.add_save_dir_info.copy()
                _info.update(info)
                info = _info
            
        if info is not None:
            info_str = ",".join([f"{k}={v}" for k, v in info.items()])
            dirname = f"{self.__class__.__name__}[{self.train_data.name};{info_str}]"
        else:
            dirname = f"{self.__class__.__name__}[{self.train_data.name}]"

        if self._root_dir is None:
            return data_dir(f"test_results/{dirname}", make_add_dirs=False) 
        else:
            return os.path.join(self._root_dir, dirname)
    
    def _init_save_paths(self, create_dirs=False):
        self._save_dir = self._create_save_dir()

        if os.path.exists(self._save_dir):
            old_dir = self._save_dir
            self._save_dir = find_dir_ver(self._save_dir)
            logging.debug(f"Save directory already exists ({old_dir}), saving to alternative directory: {self._save_dir}")   

        self._history_fp = os.path.join(self._save_dir, "history.csv")
        self._inner_history_fp = os.path.join(self._save_dir, "inner_history.csv")
        self._result_fp = os.path.join(self._save_dir, "result.json")
        self._models_dir = os.path.join(self._save_dir, "models")  

        if create_dirs:
            os.makedirs(self._save_dir, exist_ok=True)
            if self.save_best_models:
                os.makedirs(self._models_dir, exist_ok=True)


    def _get_search_attrs(self, search_space: dict, current_attrs: dict = None) -> dict:
        ignore_attrs =("result")
        params = self._model.get_params() if hasattr(self._model, "get_params") else None

        if hasattr(self._model, "__qualname__"):
            name = self._model.__qualname__
        else:
            name = self._model.__class__.__qualname__
        
        info = dict(
            search_method=self.__class__.__name__,
            model=dict(
                name=name,
                params=params
            ),
            space=search_space, # named space for compatibility with old format!
            dataset=self.train_data.name,
            method_params={}
        )

        base_attrs = BaseSearch.__init__.__code__.co_names
        for k, v in self.__dict__.items():
            if k.startswith("_") or k.startswith("__") or (k in ignore_attrs):
                continue
            elif k in ("inner_cv", "cv"):
                info[k] = CVInfo(v).to_dict() if v is not None else None
            elif k == "max_outer_iter":
                info[k] = v if (self.max_outer_iter != sys.maxsize) else None
            elif isinstance(v, (pd.DataFrame, Dataset)):
                logging.debug(f"{self.__class__.__name__}: Attrvalue {v} of type '{type(v)}' skipped for save info!")
            elif k in base_attrs:
                info[k] = v if not callable(v) else v.__name__
            else:
                info["method_params"][k] = v if not callable(v) else v.__name__
        
        if current_attrs is not None:
            if current_attrs.get("info", None) is None:
                current_attrs["info"] = info
            else:
                current_attrs["info"].update(info)
            return current_attrs

        return dict(info=info)
                
    def init_save(self, search_space: dict, fixed_params: dict):
        if not self._save:
            return
        self._init_save_paths(create_dirs=True)

        self.history_head = list(chain.from_iterable([
            ("inner_index", ), [name for name, v in search_space.items()], ("train_score", "test_score", "time")
        ]))

        data = load_json(self._result_fp, default={})
        data = self._get_search_attrs(search_space, current_attrs=data)

        save_json(self._result_fp, data)
        save_csv(self._history_fp, self.history_head)
    
    def _update_info(self, update_keys: dict):
         data = load_json(self._result_fp, default=dict(info={}))
         data["info"].update(update_keys)
         save_json(self._result_fp, data)

    def update_history(self, inner_index: int, params: dict, train_score: float, test_score: float, time: float):
        row = dict(inner_index=inner_index, train_score=train_score, test_score=test_score, time=time)
        row.update(params)
        save_csv(self._history_fp, self.history_head, row)
    
    def update_inner_history(self, outer_iter: int, inner_history: pd.DataFrame): 
        inner_history["outer_iter"] = outer_iter
        rows = inner_history.to_dict(orient="records")
        head = list(inner_history.columns)
        #head.sort()
        save_csv(self._inner_history_fp, head, rows)
    
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
            return "{}d,{:02}:{:02}:{:02}".format(days, hours, minutes, secs)
    
    @classmethod
    def recalc_results(cls, result_dir: str) -> dict:
        search = BaseSearch(None, None)
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

        if self._save:
            _data = load_json(self._result_fp, default={})
            _data["result"] = self.result
            save_json(self._result_fp, _data, overwrite=True)
    
    def score(self, model, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        if (self.scoring is not None) and isinstance(self.scoring, str):
            scorer = get_scorer(self.scoring)
            return scorer(model, x_test, y_test)
        elif (self.scoring is not None) and isinstance(self.refit, str):
            scorer = get_scorer(self.refit)
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
    
    def _encode_search_space(self, search_space: dict) -> dict:
        return search_space
    
    def _inner_search(self, search_id: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        raise RuntimeError("Unimplemented")
    
    def search(self, search_space: dict, fixed_params: dict) -> 'BaseSearch':
        self.init_save(search_space, fixed_params)
        
        if not self.train_data.has_saved_folds(self.cv):
            print(f"Saving {CVInfo(self.cv)} folds for dataset {self.train_data.name}")
            self.train_data.save_folds(self.cv)
        
        data = self.train_data.load_saved_folds_file(self.cv)
        folds = data["folds"]
        assert CVInfo(data["info"]) == CVInfo(self.cv)
        
        search_space = self._encode_search_space(search_space)
        is_sparse = self.train_data.x.dtypes.apply(pd.api.types.is_sparse).all()
        print("starting search")

        if is_sparse:
            train_x: coo_matrix = self.train_data.x.sparse.to_coo()
            train_x = train_x.tocsr()
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

            if self._save:
                self.update_history(result.best_index, result.best_params, result.best_score, acc, end)
                if self.save_best_models:
                    self.save_model(result.best_model, outer_id=i)
                if self.save_inner_history:
                    self.update_inner_history(outer_iter=i, inner_history=result.inner_history)

            print(f"{i}: best_score={round(result.best_score, 4)}, test_score={round(acc, 4)}, params={json_to_str(result.best_params, indent=None)}")

        self._calc_result()
        return self


