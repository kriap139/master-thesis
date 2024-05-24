from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, SizeGroup, Task, SK_DATASETS, load_json, parse_cmd_params, remove_lines_up_to, count_lines, get_search_space, CVInfo
import lightgbm as lgb
import logging
from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold
from typing import Union, Iterable, Tuple
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
import inspect

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
        choices=(
            "RandomSearch", "SeqUDSearch",
            "GridSearch", "OptunaSearch", "KSpaceSeqUDSearch", "KSpaceSeqUDSearchV2", "KSpaceSeqUDSearchV3", 
            "KSpaceOptunaSearch", "KSpaceOptunaSearchV2", "KSpaceOptunaSearchV3"
        ),
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
    parser.add_argument("--search-space", type=str, default=None)
    parser.add_argument("--move-slurm-logs", action='store_true')
    parser.add_argument("--copy-new-slurm-log-lines", action='store_true')
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
        args.params = parse_cmd_params(args.params)
    
    if args.search_space is not None:
        if args.search_space == 'all':
            args.search_space = None
        else:
            args.search_space = [param.strip() for param in args.search_space.split(',')]

    return args

def get_current_slurm_logs_count() -> Tuple[int, int]:
    job_name, job_id = os.environ.get("SLURM_JOB_NAME", None), os.environ.get("SLURM_JOB_ID", None)
    out_name = f"R-{job_name}.{job_id}.out"
    err_name = f"R-{job_name}.{job_id}.err"

    if job_name is not None and (job_id is not None):
        out_fp = os.path.join(os.getcwd(), out_name)
        err_fp = os.path.join(os.getcwd(), err_name)
        return (count_lines(err_fp), count_lines(out_fp))
    else:
        return (0, 0)

def copy_slurm_logs(dist_dir: str, copy=True, copy_err_from_line: int = None, copy_out_from_line: int = None):
    def save_data(fp: str, data: list):
        with open(fp, mode='w') as f:
            f.writelines(data)

    sys.stdout.flush()
    sys.stderr.flush()
    job_name, job_id = os.environ.get("SLURM_JOB_NAME", None), os.environ.get("SLURM_JOB_ID", None)
    if job_name is not None and (job_id is not None):
        out_name = f"R-{job_name}.{job_id}.out"
        err_name = f"R-{job_name}.{job_id}.err"
        out_fp = os.path.join(os.getcwd(), out_name)
        err_fp = os.path.join(os.getcwd(), err_name)

        if all(os.path.exists(fp) for fp in (out_fp, err_fp)):
            dest_err = os.path.join(dist_dir, "logs.err")
            dest_out = os.path.join(dist_dir, "logs.out")

            if not os.path.exists(dist_dir):
                print(f"Log destination dir doesn't exist: {dist_dir}", flush=True)
                return
            if copy:
                if copy_err_from_line is None:
                    shutil.copy2(out_fp, dest_err)
                else:
                    data = remove_lines_up_to(err_fp, copy_err_from_line - 1)
                    print(f"copying err logs from line {copy_err_from_line - 1}", flush=True)
                    save_data(dest_err, data)

                if copy_out_from_line is None:
                    shutil.copy2(out_fp, dest_out)
                else:
                    data = remove_lines_up_to(out_fp, copy_out_from_line - 1)
                    print(f"copying out logs from line {copy_out_from_line - 1}", flush=True)
                    save_data(dest_out, data)
            else:
                shutil.move(out_fp, dest_out)
                shutil.move(err_fp, dest_err)
        else:
            print(f"Log files dosen't exists: out={out_fp}, err={err_fp}", flush=True)
    else:
        print(f"Unable to get job id and name from environment", flush=True)

def search(args: argparse.Namespace, override_current_scoring=False) -> BaseSearch:
    logging.getLogger().setLevel(logging.DEBUG)

    search_n_jobs = min(args.n_jobs, MAX_SEARCH_JOBS)
    n_jobs= calc_n_lgb_jobs(search_n_jobs, args.max_lgb_jobs)
    print(f"CPU Cores: {CPU_CORES}, Logical Cores: {psutil.cpu_count(logical=True)}, lgb_n_jobs={n_jobs}, search_n_jobs={search_n_jobs}")

    search_space = get_search_space(args.method, args.search_space)
    dataset = Dataset(args.dataset).load()
    check_scoring(args, dataset.task, override_current=True)
    #print(dataset.x.info())
    print(f"args: {args}")

    if len(dataset.x.columns) > 199:
        print(f"n columns: {len(dataset.x.columns)}")
    else:
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
    
    save_dir_info = dict(
        nrepeat=CVInfo(cv).get_n_repeats(),
        nparams=len(search_space)
    )
    
    tuner = tuner(model=model, train_data=dataset, test_data=None, n_iter=100, add_save_dir_info=save_dir_info,
                  n_jobs=search_n_jobs, cv=cv, inner_cv=inner_cv, scoring=args.scoring, save=True, max_outer_iter=args.max_outer_iter, refit=refit, **params)

    print(f"Results saved to: {tuner._save_dir}", flush=True)
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

def main():
    args = build_cli()
    if args.params is not None and (not isinstance(args.params, dict)):
        raise ValueError(f"Array of parameters is not supported!: {args.params}")
    elif isinstance(args.dataset, list) or inspect.isclass(args.dataset):
        raise ValueError(f"Dataset argument can't be an iterable of dataset types!: {args.dataset}")

    if args.copy_new_slurm_log_lines:
        print("Counting current log file lines", flush=True)
        (curr_err_line, curr_out_line) = get_current_slurm_logs_count()
    else:
        (curr_err_line, curr_out_line) = None, None

    tuner = search(args)
    copy_slurm_logs(tuner._save_dir, not args.move_slurm_logs, curr_err_line, curr_out_line)
        
if __name__ == "__main__":
    main()
