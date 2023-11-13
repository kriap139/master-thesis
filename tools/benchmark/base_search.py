from Util import Dataset
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold
from sklearn.metrics import get_scorer, get_scorer_names
from typing import Sequence, Union, Optional, Iterable
from Util.dataset import Dataset, Builtin, Task, TY_CV
from Util.io_util import load_json, save_json, data_dir
from skopt.space import Real, Integer
import gc
from typing import Callable
import os
from datetime import datetime
import logging

class BaseSearch:
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring=None, save_path=None):
        self.train_data = train_data
        self.test_data = test_data
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.cv = cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self._model = model

        self._save_timestamp = datetime.now().strftime("%d-%m-%Y@%H:%M:%S")
        self._save_path = save_path
        self._save = save_path is not None
        
        if self._save_path is not None and os.path.exists(self._save_path):
            logging.debug("Save file already exists!")

        # save file data
        self._params = []
        self._train_scores = []
        self._test_scores = []
        self._result = None

    
    def init_save(self, exstra_keys: dict = None, info_append: dict = None):
        data = load_json(self._save_path, default={})

        data = dict(
            info=dict(
                dataset=self.train_data.name,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                cv=Dataset.get_cv_info(self.cv) if self.cv is not None else None,
                inner_cv=Dataset.get_cv_info(self.inner_cv) if self.inner_cv is not None else None
            ),
            params=[],
            train_scores=[],
            test_scores=[]
        )
        if exstra_keys is not None:
            data.update(exstra_keys)

        if info_append:
            for k, v in info_append.items():
                data["info"][k] = v
        
        if callable(self.scoring):
            data["scoring"] = self.scoring.__name__
        else:
            data["scoring"] = self.scoring

        save_json(self._save_path, data)
    
    def update_save(self, append_keys: dict = None, replace_keys: dict = None, update_keys: dict = None):
        data = load_json(self._save_path, default={})
        
        if replace_keys is not None:
            for k, v in replace_keys.items():
                data[k] = v
        
        if append_keys is not None:
            for k, v in append_keys.items():
                if isinstance(v, Iterable) and isinstance(data[k], Iterable):
                    data[k].extend(v)
        
        if update_keys is not None:   
            data.update(update_keys)

        save_json(self._save_path, data, overwrite=True)
    
    def _add_iteration_stats(self, params: dict, train_score: float, test_score: float):
        self._params.append(params)
        self._train_scores.append(train_score)
        self._test_scores.append(test_score)
    
    def _clear_iterations_stats_cache(self):
        self._params.clear()
        self._test_scores.clear()
        self._train_scores.clear()
    
    def _flush_iterations_stats(self, append_keys: dict = None, replace_keys: dict = None, update_keys: dict = None, clear_cache=True):
        if append_keys is None:
            append_keys = {}

        append_keys["params"] = self._params
        append_keys["train_scores"] = self._train_scores
        append_keys["test_scores"] = self._test_scores

        self.update_save(append_keys, replace_keys, update_keys)

        if clear_cache:
            self._clear_iterations_stats_cache()
    
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
    
    def _set_result(self, time: float):
        self.result = dict(
            mean_train_acc=np.mean(self._train_scores),
            std_train_acc=np.std(self._train_scores),
            mean_test_acc=np.mean(self._test_scores),
            std_test_acc=np.std(self._test_scores),
            time=time
        )

    def search(self, params: dict, fixed_params: dict):
        raise RuntimeError("Not implemented")
