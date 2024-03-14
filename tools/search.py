from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, SizeGroup, Task
import lightgbm as lgb
import logging
from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold
from typing import Union, Iterable
import benchmark
import psutil
import argparse
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import get_scorer_names
from itertools import chain
import gc

MAX_SEARCH_JOBS = 4
CPU_CORES = psutil.cpu_count(logical=False)


def get_sklearn_model(dataset: Dataset, **params) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        if dataset.task in (Task.MULTICLASS, Task.BINARY):
            return lgb.LGBMClassifier(**params)
        elif dataset.task == Task.REGRESSION:
            return lgb.LGBMRegressor(**params)

def get_cv(dataset: Dataset, n_splits=5, n_repeats=10, random_state=None, shuffle=False, no_stratify=False):
    if dataset.task == Task.REGRESSION or no_stratify:
        if n_repeats == 0:
            return KFold(n_splits=n_splits, random_state=random_state if shuffle else None, shuffle=shuffle)
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
    else:
        if n_repeats == 0:
            return StratifiedKFold(n_splits=n_splits, random_state=random_state if shuffle else None, shuffle=shuffle)
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

def calc_n_lgb_jobs(n_search_jobs: int, max_lgb_jobs: int) -> int:
    n_jobs = int(float(CPU_CORES) / n_search_jobs)
    return min(min(n_jobs, CPU_CORES), max_lgb_jobs)

def build_cli(test_method: str = None, test_dataset: Builtin = None, test_max_lgb_jobs=None, test_n_jobs=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--method", 
        choices=("RandomSearch", "SeqUDSearch", "AdjustedSeqUDSearch", "GridSearch", "OptunaSearch"),
        type=str,
        required=test_method is None,
    )
    parser.add_argument("--dataset",
        action='append',
        nargs='+',
        choices=list(chain.from_iterable([("all", ), tuple(b.name.lower() for b in Builtin)])), 
        required=test_dataset is None
    )
    parser.add_argument("--params",
        type=str,
        default=None
    )
    parser.add_argument("--n-jobs", type=int, default=MAX_SEARCH_JOBS)
    parser.add_argument("--max-lgb-jobs", type=int, default=CPU_CORES)

    parser.add_argument("--max-outer-iter", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=None)

    parser.add_argument("--inner-n-folds", type=int, default=5)
    parser.add_argument("--inner-shuffle", action='store_true')
    parser.add_argument("--inner-random-state", type=int, default=None)
    parser.add_argument("--refit_metric", type=str, default=None)

    parser.add_argument("--scoring", type=str, default=None, choices=get_scorer_names())

    args = parser.parse_args()

    if test_max_lgb_jobs is not None:
        args.max_lgb_jobs = test_max_lgb_jobs
    if test_n_jobs is not None:
        args.n_jobs = test_n_jobs
    if test_method is not None:
        args.method = test_method

    if test_dataset is not None:
        args.dataset = test_dataset
    elif isinstance(args.dataset, Iterable):
        datasets = [a[0] for a in args.dataset]
        args.dataset = [Builtin[dt.strip().upper()] for dt in datasets]
    elif isinstance(args.dataset, str):
        if args.dataset.strip() == 'all':
            args.dataset = Builtin
        else:
            args.dataset = Builtin[args.dataset.strip().upper()]

    if args.scoring is not None and (args.scoring not in get_scorer_names()):
        raise RuntimeError(f"Unnsupported scoring {args.scoring}")
    
    if args.params is not None:
        def convert_param(param: str):
            try:
                return int(param)
            except ValueError:
                try:
                    return float(param)
                except ValueError:
                    return param

        params = args.params.strip().split(",")
        args.params = {tup[0].strip(): convert_param(tup[1].strip()) for tup in param.split("=") for param in params}
    
    return args

def get_search_space(args: argparse.Namespace) -> dict:
    if args.method == "GridSearch":
        return dict(
            n_estimators=[50, 100, 200, 350, 500],
            learning_rate=[0.001, 0.01, 0.05, 0.1, 0.02],
            max_depth=[0, 5, 10, 20, 25],
            num_leaves=[20, 60, 130, 200, 250],
        )
        print(f"GridSearch runs: {len(ParameterGrid(search_space))}")
    else:
        return dict(
            n_estimators=Integer(1, 500, name="n_estimators", prior="log-uniform"),
            learning_rate=Real(0.0001, 1.0, name="learning_rate", prior="log-uniform"),
            max_depth=Integer(0, 30, name="max_depth"),
            num_leaves=Integer(10, 300, name="num_leaves", prior="log-uniform"),
            min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
            feature_fraction=Real(0.1, 1.0, name="feature_fraction", prior="log-uniform")
        )

def search(args: argparse.Namespace):
    logging.getLogger().setLevel(logging.DEBUG)

    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")

    search_space = get_search_space(args)
    dataset = Dataset(args.dataset).load()
    #print(dataset.x.info())
    print(f"args: {args}")
    print(f"column names: {list(dataset.x.columns)}")
    print(f"cat_features: {dataset.cat_features}")
    
    fixed_params = dict(
        #objective=OBJECTIVES[dataset.get_builtin()],
        #metric=METRICS[dataset.get_builtin()]
        categorical_feature=dataset.cat_features,
    )
    params = args.params if args.params is not None else {}

    tuner = getattr(benchmark, args.method)
    model = get_sklearn_model(dataset, verbose=-1, n_jobs=n_jobs)
    refit = args.refit_metric if args.refit_metric is not None else True
    
    cv = get_cv(dataset, args.n_folds, args.n_repeats, args.random_state)
    inner_cv = get_cv(dataset, args.inner_n_folds, 0, args.inner_random_state, args.inner_shuffle)
    
    tuner = tuner(model=model, train_data=dataset, test_data=None, n_iter=100, 
                  n_jobs=search_n_jobs, cv=cv, inner_cv=inner_cv, scoring=args.scoring, save=True, max_outer_iter=args.max_outer_iter, refit=refit, **params)

    print(f"Results saved to: {tuner._save_dir}")
    tuner.search(search_space, fixed_params)

def check_scoring(args: argparse.Namespace, override_current=False) -> tuple:
    if args.dataset.info().task == Task.REGRESSION:
        args.scoring = (
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error"
        )
        args.refit_metric = "r2"
    elif override_current:
        args.scoring = None
        args.refit_metric = None

if __name__ == "__main__":
    args = build_cli()

    if isinstance(args.dataset, Iterable):
        datasets = args.dataset
        for dataset in datasets:
            args.dataset = dataset
            check_scoring(args, override_current=True)
            search(args)
            gc.collect()
    else:
        check_scoring(args)
        search(args)
