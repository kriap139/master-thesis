import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score
from sequd import SeqUD
from scipy.sparse import coo_matrix
from sklearn.metrics import get_scorer, get_scorer_names

from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold, SeqUDSearch, OptunaSearch, AdjustedSeqUDSearch
from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, has_csv_header, CVInfo
import lightgbm as lgb
from search import get_sklearn_model, get_cv, build_cli, search, calc_n_lgb_jobs, get_search_space, MAX_SEARCH_JOBS, CPU_CORES
import logging
import csv
import argparse

def cli(method: str, dataset: Builtin, max_lgb_jobs=None, n_jobs=None) -> argparse.Namespace:
    if (max_lgb_jobs is not None) and n_jobs is not None:
        args = build_cli(method, dataset, max_lgb_jobs, n_jobs)
    elif max_lgb_jobs is not None:
        args = build_cli(method, dataset, test_max_lgb_jobs=max_lgb_jobs)
    elif n_jobs is not None:
        args = build_cli(method, dataset, n_jobs=n_jobs)
    else:
        args = build_cli(method, test_dataset)
    return args

def search_test(method: str, dataset: Builtin, max_lgb_jobs=None, n_jobs=None):
    args = cli(method, dataset, max_lgb_jobs, n_jobs)
    search(args)

def plain_test(bn: Builtin, max_lgb_jobs=None, n_jobs=None):
    args = cli(AdjustedSeqUDSearch.__name__, bn, max_lgb_jobs, n_jobs)
    dataset = Dataset(bn).load()
    search_space = get_search_space(args)
    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")

    fixed_params = dict(
        #objective=OBJECTIVES[dataset.get_builtin()],
        #metric=METRICS[dataset.get_builtin()]
        categorical_feature=dataset.cat_features,
    )

    cv = get_cv(dataset, args.n_folds, args.n_repeats, args.random_state)
    inner_cv = get_cv(dataset, args.inner_n_folds, 0, args.inner_random_state, args.inner_shuffle)

    for i, (train_idx, test_idx) in enumerate(cv.split()):
        x_train, x_test = dataset.x.iloc[train_idx, :], dataset.x.iloc[test_idx, :]
        y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
        tuner = RandomizedSearchCV(get_sklearn_model(dataset), search_space, n_iter=100, n_jobs=args.n_jobs, random_state=9, cv=inner_cv)
        test_acc = tuner.best_estimator_.score(x_test, y_test)
        print(f"{i}: train={tuner.best_score_}, test={test_acc}")


if "__main__" == __name__:
    #search_test(AdjustedSeqUDSearch.__name__, Builtin.ACCEL, max_lgb_jobs=6, n_jobs=1)
    plain_test(Builtin.ACCEL, max_lgb_jobs=6, n_jobs=1)