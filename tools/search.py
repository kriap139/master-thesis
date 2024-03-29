from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, SizeGroup, Task, SK_DATASETS, load_json
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
import sys
import os
import shutil
import re

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
        choices=("RandomSearch", "SeqUDSearch", "AdjustedSeqUDSearch", "GridSearch", "OptunaSearch", "KSpaceSeqUDSearch", "KSpaceOptunaSearch"),
        type=str,
        required=test_method is None,
    )
    parser.add_argument("--dataset",
        action='append',
        nargs='+',
        choices=list(chain.from_iterable([("all", ), tuple(b.name.lower() for b in Builtin), SK_DATASETS])), 
        required=test_dataset is None
    )
    parser.add_argument("--params",
        type=str,
        default=None
    )
    parser.add_argument("--search-space",
        action='append',
        nargs='+',
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
        for i in range(len(datasets)):
            name = datasets[i].strip().upper()
            if name in Builtin._member_names_:
                datasets[i] = Builtin[name]
        args.dataset = datasets if len(datasets) > 1 else datasets[0]
    else:
        raise ValueError(f"--dataset argument is an invalid type({type(args.dataset)})")

    if args.scoring is not None and (args.scoring not in get_scorer_names()):
        raise RuntimeError(f"Unnsupported scoring {args.scoring}")
    
    if args.params is not None:
        def try_number(param: str):
            try:
                return int(param)
            except ValueError:
                try:
                    return float(param)
                except ValueError:
                    return param

        def convert_param(param: str):
            if param.startswith("["):
                param = param[1:len(param) - 1].strip()
                params = param.split(',')
                return [try_number(p) for p in params]
            elif param.startswith("{"):
                dct = {}
                string = param[1:len(param) - 1].strip()
                for param in string.split(','):
                    k, v = param.split("=")
                    dct[k.strip()] = try_number(v.strip())
                return dct
            return try_number(param)
        
        if os.path.exists(args.params):
            args.params = load_json(args.params_file, default={})
        else:
            params = {} 
            comma_pattern = r',(?![^{]*})(?![^\[]*\])'
            eq_pattern = r'=(?![^{]*})(?![^\[]*\])'

            for param in re.split(comma_pattern, args.params):
                name, value = re.split(eq_pattern, param)
                params[name.strip()] = convert_param(value.strip())
            args.params = params

    return args

def copy_slurm_logs(dist_dir: str, copy=True, clear_contents=False):
    sys.stdout.flush()
    sys.stderr.flush()
    job_name, job_id = os.environ.get("SLURM_JOB_NAME", None), os.environ.get("SLURM_JOB_ID", None)
    if job_name is not None and (job_id is not None):
        out_name = f"R-{job_name}.{job_id}.out"
        err_name = f"R-{job_name}.{job_id}.err"
        out_fp = os.path.join(os.getcwd(), out_name)
        err_fp = os.path.join(os.getcwd(), err_name)

        if all(os.path.exists(fp) for fp in (out_fp, err_fp)):
            if not os.path.exists(dist_dir):
                print(f"Log destination dir doesn't exist: {dist_dir}", flush=True)
                return
            if copy:
                shutil.copy2(out_fp, os.path.join(dist_dir, "logs.out"))
                shutil.copy2(err_fp, os.path.join(dist_dir, "logs.err"))
                if clear:
                    for fp in (out_fp, err_fp):
                        with open(fp, mode='w') as f:
                            f.truncate(0)
            else:
                shutil.move(out_fp, os.path.join(dist_dir, "logs.out"))
                shutil.move(err_fp, os.path.join(dist_dir, "logs.err"))
        else:
            print(f"Log files dosen't exists: out={out_fp}, err={err_fp}", flush=True)
    else:
        print(f"Unable to get job id and name from environment", flush=True)
            
            

def get_search_space(args: argparse.Namespace) -> dict:
    if args.method == "GridSearch":
        space = dict(
            n_estimators=[50, 100, 200, 350, 500],
            learning_rate=[0.001, 0.01, 0.05, 0.1, 0.02],
            max_depth=[0, 5, 10, 20, 25],
            num_leaves=[20, 60, 130, 200, 250],
        )
        print(f"GridSearch runs: {len(ParameterGrid(search_space))}")
    else:
        space = dict(
            n_estimators=Integer(1, 500, name="n_estimators", prior="log-uniform"),
            learning_rate=Real(0.0001, 1.0, name="learning_rate", prior="log-uniform"),
            max_depth=Integer(0, 30, name="max_depth"),
            num_leaves=Integer(10, 300, name="num_leaves", prior="log-uniform"),
            min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
            feature_fraction=Real(0.1, 1.0, name="feature_fraction", prior="log-uniform")
        )
    
    if args.search_space is not None:
        if isinstance(args.search_space, str):
            space = {args.search_space: space[args.search_space]}
        elif isinstance(args.search_space, Iterable):
            space = {param: space[param] for param in args.search_space}

    return space

def search(args: argparse.Namespace, override_current_scoring=False) -> BaseSearch:
    logging.getLogger().setLevel(logging.DEBUG)

    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")

    search_space = get_search_space(args)
    dataset = Dataset(args.dataset).load()
    check_scoring(args, dataset.task, override_current=True)
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
    
    tuner = tuner(model=model, train_data=dataset, test_data=None, n_iter=100, add_save_dir_info=dict(nparams=len(search_space)),
                  n_jobs=search_n_jobs, cv=cv, inner_cv=inner_cv, scoring=args.scoring, save=True, max_outer_iter=args.max_outer_iter, refit=refit, **params)

    print(f"Results saved to: {tuner._save_dir}")
    tuner.search(search_space, fixed_params)
    return tuner

def check_scoring(args: argparse.Namespace, task: Task, override_current=False) -> tuple:
    if task == Task.REGRESSION:
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

    if isinstance(args.dataset, (str, Builtin)):
        tuner = search(args)
        copy_slurm_logs(tuner._save_dir, copy=False)
    elif isinstance(args.dataset, Iterable):
        datasets = args.dataset
        for dataset in datasets:
            args.dataset = dataset
            tuner = search(args, override_current_scoring=True)
            gc.collect()
            copy_slurm_logs(tuner._save_dir, copy=True, clear_contents=True)
