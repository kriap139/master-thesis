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

        
class OPTUNA(BaseSearch):
    def __init__(self, model, train_data: Dataset, test_data: Dataset = None,
                 n_iter=100, n_jobs=None, cv: TY_CV = None, inner_cv: TY_CV = None, scoring = None, save_dir=None):
        super().__init__(model, train_data, test_data, n_iter, n_jobs, cv, inner_cv, scoring, save_dir)
    
    def _inner_search(self, search_id: int, x_train: pd.DataFrame, y_train: pd.DataFrame, search_space: dict, fixed_params: dict) -> InnerResult:
        space = search_space.copy()

        for k in space.keys():
            v = space[k]
            if isinstance(v, Integer):
                space[k] = IntDistribution(v.name, v.low, v.high, log=(v.prior == "log-uniform"))
            elif isinstance(v, Real):
                space[k] = FloatDistribution(v.name, v.low, v.high, log=(v.prior == "log-uniform"))