import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, ParameterSampler
from itertools import chain
from typing import Iterable, Union, Iterable, Callable
from numbers import Number
from .k_space import KSpaceV3
from Util import TY_SPACE

class KSpaceRandom(RandomizedSearchCV):
    
    def __init__(self, 
        estimator, 
        param_distributions: TY_SPACE, 
        n_iter=10, 
        scoring=None, 
        n_jobs=None, 
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False):
        super().__init__(
            estimator=estimator, 
            param_distributions=param_distributions, 
            n_iter=n_iter, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            refit=refit, 
            cv=cv, 
            verbose=verbose, 
            pre_dispatch=pre_dispatch, 
            random_state=random_state, 
            error_score=error_score, 
            return_train_score=return_train_score
        )

        self.kspace = None
        self.original_candidates = None
    
    def set_k(self, k:  Union[Number, dict] = None):
        self.kspace = KSpaceV3(self.param_distributions, k, x_in_search_space=True)
    
    def process_results(self):
        frame = pd.DataFrame(self.cv_results_)
        originals = pd.DataFrame(self.original_candidates).to_dict(orient="series")

        for column, values in originals.items():
            frame[column + '_original'] = values
        self.cv_results_ = frame
    
    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        sampled = ParameterSampler(
            self.param_distributions, self.n_iter, random_state=self.random_state
        )

        candidates = []
        for sample in sampled:
            candidates.append(
                {param: self.kspace.kmap(param, v, default=v) for param, v in sample.items()}
            )
        
        self.original_candidates = sampled
        evaluate_candidates(candidates)
        