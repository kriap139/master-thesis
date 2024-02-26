import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from scipy.stats import uniform
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score
from sequd import SeqUD
from scipy.sparse import coo_matrix
from sklearn.metrics import get_scorer, get_scorer_names
from typing import Iterable, Callable, Tuple, Dict, Union, List

from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold, SeqUDSearch, OptunaSearch, AdjustedSeqUDSearch, RandomSearch
from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, has_csv_header, CVInfo, save_json, TY_CV, load_json, find_files
import lightgbm as lgb
from search import get_sklearn_model, get_cv, build_cli, search, calc_n_lgb_jobs, get_search_space, MAX_SEARCH_JOBS, CPU_CORES
import logging
import csv
import argparse
import psutil
from dataclasses import dataclass
import os
import gc
import random
import re
import time

@dataclass
class TestResult:
    train_score: Union[float, Iterable]
    test_score: Union[float, Iterable]
    info: dict
    means: Dict[str, float] = None
    is_inner: bool = False

def cli(method: str = None, dataset: Builtin = None, max_lgb_jobs=None, n_jobs=None) -> argparse.Namespace:
    if (max_lgb_jobs is not None) and n_jobs is not None:
        args = build_cli(method, dataset, max_lgb_jobs, n_jobs)
    elif max_lgb_jobs is not None:
        args = build_cli(method, dataset, test_max_lgb_jobs=max_lgb_jobs)
    elif n_jobs is not None:
        args = build_cli(method, dataset, n_jobs=n_jobs)
    else:
        args = build_cli(method, dataset)
    return args

def run_basic_tests(tests: List[Callable[[Dataset, int, int], TestResult]], bns: Iterable[Builtin], args: argparse.Namespace, save_fns: List[str]=None):
    if save_fns is not None:
        assert len(tests) == len(save_fns)
    first_test = True

    for bn in bns:
        dataset = Dataset.try_from(bn, load=True)
        if dataset is None:
            continue
        
        print(f"{'-' * 60}{bn.name} Dataset{'-' * 60}")
        for i, test in enumerate(tests):
            print(f"\n{'-' * 29}{test.__name__}{'-' * (29 + len(test.__name__))}")
            
            start = time.perf_counter()
            result = test(args, dataset, print_jobs_info=first_test)
            end = time.perf_counter() - start
            result.info["time"] = dict(secs=end, formated=BaseSearch.time_to_str(end))
            first_test = False

            print(f"Test time={BaseSearch.time_to_str(end)}")

            if save_fns is not None:
                new_data = {bn.name: result}
                fn = save_fns[i]

                data = load_json(os.path.join(data_dir(), fn), default={})
                data.update(new_data)
                save_json(os.path.join(data_dir(), fn), data, overwrite=True)

        del dataset
        gc.collect()

def _cv_test_outer_loop(args: argparse.Namespace, func: Callable[[Dataset, int, int, argparse.Namespace], TestResult], dataset: Dataset, cv: TY_CV, shuffle=False, print_jobs_info=True) -> TestResult:
    search_space = get_search_space(args)
    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    if print_jobs_info:
        print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")
    
    is_sparse = dataset.x.dtypes.apply(lambda dtype: isinstance(dtype, pd.SparseDtype)).all()
    if is_sparse:
        train_x: coo_matrix = dataset.x.sparse.to_coo()
        train_x = train_x.tocsr()
    else:
        train_x = None
    
    train_scores = []
    test_scores = []
    info = CVInfo(cv).to_dict()

    for i, (train_idx, test_idx) in enumerate(cv.split(dataset.x, dataset.y)):
        if shuffle:
            random.shuffle(train_idx)
            random.shuffle(test_idx)

        if is_sparse:
            x_train, x_test = train_x[train_idx, :], train_x[test_idx, :]
        else:
            x_train, x_test = dataset.x.iloc[train_idx, :], dataset.x.iloc[test_idx, :]
        
        y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
        result = func(dataset, x_train, y_train, x_test, y_test, args, search_n_jobs, n_jobs)

        if result.is_inner:
            if 'inner_cv' not in info:
                info['inner_cv'] = result.info

        print(f"Fold {i}: train={result.train_score}, test={result.test_score}")
        train_scores.append(result.train_score)
        test_scores.append(result.test_score)
    
    mean_train = np.mean(train_scores)
    mean_test = np.mean(test_scores)
    print(f"Mean scores: train={mean_train}, test={mean_test}")
    
    return TestResult(train_scores, test_scores, info, means=dict(train=mean_train, test=mean_test))

def _basic_cv_test_func(dataset: Dataset, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, args: argparse.Namespace, n_search_jobs=None, n_lgb_jobs=None) -> TestResult:
    model = get_sklearn_model(dataset, verbose=-1, n_jobs=n_lgb_jobs)
    model.fit(x_train, y_train, categorical_feature=dataset.cat_features)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    return TestResult(train_score, test_score, {})

def _basic_inner_cv_test_func(dataset: Dataset, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, args: argparse.Namespace, n_search_jobs=None, n_lgb_jobs=None) -> TestResult:
    model = get_sklearn_model(dataset, verbose=-1, n_jobs=n_lgb_jobs)
    cv = get_cv(dataset, args.inner_n_folds, 0, args.inner_random_state, args.inner_shuffle)
    search = RandomizedSearchCV(model, get_search_space(args), n_iter=50, cv=cv, n_jobs=n_search_jobs)
    search.fit(x_train, y=y_train, categorical_feature=dataset.cat_features)

    train_score = search.best_score_
    test_score = search.best_estimator_.score(x_test, y_test)
    return TestResult(train_score, test_score, CVInfo(cv).to_dict(), is_inner=True)

def basic_cv_test(args: argparse.Namespace, dataset: Dataset, print_jobs_info=True) -> TestResult:
    cv = get_cv(dataset, args.inner_n_folds, 0, args.inner_random_state, args.inner_shuffle)
    return _cv_test_outer_loop(args, _basic_cv_test_func, dataset, cv, print_jobs_info=print_jobs_info)

def basic_cv_repeat_test(args: argparse.Namespace, dataset: Dataset, print_jobs_info=True) -> TestResult:
    cv = get_cv(dataset, args.n_folds, args.n_repeats, args.random_state)
    return _cv_test_outer_loop(args, _basic_cv_test_func, dataset, cv, print_jobs_info=print_jobs_info)
    
def basic_inner_cv_test(args: argparse.Namespace, dataset: Dataset, print_jobs_info=True) -> TestResult:
    cv = get_cv(dataset, args.n_folds, args.n_repeats, args.random_state)
    return _cv_test_outer_loop(args, _basic_inner_cv_test_func, dataset, cv, print_jobs_info=print_jobs_info)

def basic_no_repeat_inner_cv_test(args: argparse.Namespace, dataset: Dataset, print_jobs_info=True) -> TestResult:
    cv = get_cv(dataset, args.n_folds, 0, args.random_state, shuffle=args.inner_shuffle)
    return _cv_test_outer_loop(args, _basic_inner_cv_test_func, dataset, cv, print_jobs_info=print_jobs_info)

def basic_test(args: argparse.Namespace, dataset: Dataset, print_jobs_info=True) -> TestResult:
    search_space = get_search_space(args)
    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    if print_jobs_info:
        print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")

    is_sparse = dataset.x.dtypes.apply(lambda dtype: isinstance(dtype, pd.SparseDtype)).all()

    train_x = dataset.x
    if is_sparse:
        train_x: coo_matrix = dataset.x.sparse.to_coo()
        train_x = train_x.tocsr()
        
    x_train, x_test, y_train, y_test = train_test_split(train_x, dataset.y, test_size=0.30, random_state=9)

    model = get_sklearn_model(dataset, verbose=-1, n_jobs=n_jobs)
    model.fit(x_train, y_train, categorical_feature=dataset.cat_features)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print(f"Results: train={train_score}, test={test_score}")
    return TestResult(train_score, test_score, {})

def print_basic_test_results():
    files = find_files(os.path.join(data_dir(), f"basic_*.json"))
    names = [re.sub(r'basic_|.json', '', os.path.basename(fn)) for fn in files]
    datas = [load_json(file) for file in files]

    for i, data in enumerate(datas):
        print(f"{names[i]}:")
        for dataset, result in data.items():
            if 'means' in result.keys() and (result['means'] is not None):
                train_score = result['means']['train']
                test_score = result['means']['test']
            elif isinstance(result['train_score'], Iterable):
                train_score = np.mean(result['train_score'])
                test_score = np.mean(result['test_score'])
            else:
                train_score = result['train_score']
                test_score = result['test_score']

            diff_score = np.abs(train_score - test_score)
            print(f"\t{dataset} -> train={round(train_score, 4)}, test={round(test_score, 4)}, diff={round(diff_score, 5)}")

# python tools/test.py --max-lgb-jobs 1 --n-jobs 3 --n-repeats 3 --n-folds 5 --random-state 9 --inner-n-folds 5 --inner-shuffle --inner-random-state 9 --dataset accel
if "__main__" == __name__:
    # Args method is not Used in this script!
    args = cli(RandomSearch.__name__)
    datasets = [args.dataset] if not isinstance(args.dataset, Iterable) else args.dataset

    tests = [
        basic_test, 
        basic_cv_test, 
        basic_cv_repeat_test, 
        #basic_inner_cv_test, 
        basic_no_repeat_inner_cv_test
    ]

    save_fns = [
        f"basic_split_tests.json", 
        f"basic_cv_tests.json", 
        f"basic_cv_repeats_tests.json", 
        #f"basic_inner_cv_tests.json", 
        f"basic_no_outer_repeats_inner_cv_test.json"
    ]

    #run_basic_tests(tests, datasets, args, save_fns)
    print_basic_test_results()














