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
        model, 
        k_space: TY_SPACE, 
        k:  Union[Number, dict] = None, 
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
            estimator=model, param_distributions=k_space, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs, 
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state, 
            error_score=error_score, return_train_score=return_train_score
        )
        self.kspace = KSpaceV3(k_space, k, x_in_search_space=True)
    
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

        evaluate_candidates(candidates)
        frame = pd.DataFrame(self.cv_results_)

        for column, values in pd.DataFrame(sampled).to_dict(orient="list"):
            frame[column + '_original'] = values
        self.cv_results_ = frame

        self.cv_results_ = results
    
