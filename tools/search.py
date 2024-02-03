from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical
import lightgbm as lgb
import logging
from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold
from typing import Union
import benchmark
import psutil
import argparse
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import get_scorer_names

MAX_SEARCH_JOBS = 4
CPU_CORES = psutil.cpu_count(logical=False)


OBJECTIVES = {
        Task.BINARY: "binary",
        Task.MULTICLASS: "softmax",
        Task.REGRESSION: "l2"
}

METRICS = {
    Task.BINARY: "binary_logloss",
    Task.MULTICLASS: "multi_logloss",
    Task.REGRESSION: "l2",
}

SCORING = {
    Task.BINARY: "acc"
}

def get_sklearn_model(dataset: Dataset, **params) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        if dataset.task in (Task.MULTICLASS, Task.BINARY):
            return lgb.LGBMClassifier(**params)
        elif dataset.task == Task.REGRESSION:
            return lgb.LGBMRegressor(**params)

def get_cv(dataset: Dataset, no_repeats: bool, n_splits=5, n_repeats=10, random_state=None, shuffle=False, no_stratify=False):
    if dataset.task == Task.REGRESSION or no_stratify:
        if no_repeats:
            return KFold(n_splits=n_splits, random_state=random_state if shuffle else None, shuffle=shuffle)
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
    else:
        if no_repeats:
            return StratifiedKFold(n_splits=n_splits, random_state=random_state if shuffle else None, shuffle=shuffle)
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

def calc_n_lgb_jobs(n_search_jobs: int, max_lgb_jobs: int) -> int:
    n_jobs = int(float(CPU_CORES) / search_n_jobs)
    return min(min(n_jobs, CPU_CORES), max_lgb_jobs)

def build_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--method", 
        choices=("RandomSearch", "SeqUDSearch", "GridSearch", "OptunaSearch"),
        type=str,
        required=True
    )
    parser.add_argument("--dataset",
        choices=tuple(b.name.lower() for b in Builtin), 
        required=True
    )
    parser.add_argument("--n-jobs", type=int, default=MAX_SEARCH_JOBS)
    parser.add_argument("--max-lgb-jobs", type=int, default=CPU_CORES)

    parser.add_argument("--no-repeats", action='store_true')
    parser.add_argument("--max-outer-iter", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)

    parser.add_argument("--inner-n-folds", type=int, default=5)
    parser.add_argument("--inner-shuffle", action='store_true')
    
    parser.add_argument("--scoring", type=str, default=None, choices=get_scorer_names())

    args = parser.parse_args()
    args.dataset = Builtin[args.dataset.upper()]

    if args.scoring is not None and (args.scoring not in get_scorer_names()):
        raise RuntimeError(f"Unnsupported scoring {args.scoring}")
    
    return args

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    args = build_cli()

    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")

    if args.method == "GridSearch":
        search_space = dict(
            n_estimators=[50, 100, 200, 350, 500],
            learning_rate=[0.001, 0.01, 0.05, 0.1, 0.02],
            max_depth=[0, 5, 10, 20, 25],
            num_leaves=[20, 60, 130, 200, 250],
        )
        print(f"GridSearch runs: {len(ParameterGrid(search_space))}")
    else:
        search_space = dict(
            n_estimators=Integer(1, 500, name="n_estimators", prior="log-uniform"),
            learning_rate=Real(0.0001, 1.0, name="learning_rate", prior="log-uniform"),
            max_depth=Integer(0, 30, name="max_depth"),
            num_leaves=Integer(10, 300, name="num_leaves", prior="log-uniform"),
            min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
            feature_fraction=Real(0.1, 1.0, name="feature_fraction", prior="log-uniform")
        )

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

    tuner = getattr(benchmark, args.method)
    save_dir = data_dir(f"test_results/{tuner.__name__}[{dataset.name}]")
    model = get_sklearn_model(dataset, verbose=-1, n_jobs=n_jobs)
    
    cv = get_cv(dataset, args.no_repeats, args.n_folds, n_repeats=args.n_repeats)
    inner_cv = get_cv(dataset, True, args.inner_n_folds, shuffle=args.inner_shuffle)
        
    tuner = tuner(model=model, train_data=dataset, test_data=None, n_iter=100, 
                  n_jobs=search_n_jobs, cv=cv, inner_cv=inner_cv, scoring=args.scoring, save_dir=save_dir, max_outer_iter=args.max_outer_iter)

    print(f"Results saved to: {tuner._save_dir}")
    tuner.search(search_space, fixed_params)
