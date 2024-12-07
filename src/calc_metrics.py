import os
from Util import Dataset, Builtin, data_dir, Task, get_n_search_space
from Util.io_util import load_json, json_to_str, load_csv
from Util.compat import removeprefix
from benchmark import BaseSearch, KSearchOptuna
import benchmark
from typing import List, Dict, Tuple, Callable, Any, Union, Optional
import re
from dataclasses import dataclass
import pandas as pd
import hashlib
from itertools import chain
import itertools
from numbers import Number
from scipy.stats import friedmanchisquare
import sys

@dataclass
class ResultFolder:
    dir_path: str
    dataset: Builtin
    search_method: str
    dir_str: str
    version: int = 0
    info: dict = None
    
    def info_hash(self) -> str:
        js = json_to_str(self.info, indent=None, sort_keys=True).encode()
        dhash = hashlib.md5()
        dhash.update(js)
        return dhash.hexdigest()

def select_version(current: ResultFolder, new: Optional[ResultFolder] = None, select_versions: Dict[str, Dict[str, Union[int, dict]]] = None, return_current_as_default=True) -> Optional[ResultFolder]:
    data = select_versions.get(current.dataset.name, None) if select_versions is not None else None
    select = data.get(current.search_method, None) if data is not None else None

    if select is not None:
        if type(select) == int:
            if new is None and select == current.version:
                return current
            elif select == current.version:
                return current
            elif select == new.version:
                return new
            return
        elif isinstance(select, dict):
            select = select.copy()
            version = select.pop("version", None)
            select = select if len(select) > 0 else None
            test = ResultFolder(current.dir_path, current.dataset, current.search_method, current.dir_str, version, select)

            curr_hash = current.info_hash()
            if new is None:
                if test.info_hash() == curr_hash and (version is None or version == current.version):
                    return current
            elif curr_hash == new.info_hash():
                if version is None:
                    return new if current.version < new.version else current
                else:
                    if new.version == version:
                        return new
                    elif current.version == version:
                        return current
            elif test.info_hash() == curr_hash and (version is None or version == current.version):
                return current
            elif test.info_hash() == new.info_hash() and (version is None or version == current.version):
                return new
            return None
        else:
            raise RuntimeError(f"Folder selection attribute for folder(dataset={current.dataset.name}, method={current.search_method}) is invalid, only version(int) or info(dict) is supported: {select}")
    
    if (new is not None) and current.version < new.version:
        return new
    elif return_current_as_default:
        return current

def sort_folders(folders: Dict[str, Dict[str, ResultFolder]], fn: Callable[[ResultFolder], Any], filter_fn: Callable[[ResultFolder], bool] = None, reverse=False) -> Dict[str, Dict[str, ResultFolder]]:
    joined: List[ResultFolder] = []
    for (dataset, methods) in folders.items():
        for method, dirs in methods.items():
            if isinstance(dirs, list):
                joined.extend(dirs)
            else:
                joined.append(dirs)
    
    if filter_fn is not None:
        joined = list(filter(filter_fn, joined))
        
    joined.sort(key=fn, reverse=reverse)
    folders.clear()

    for folder in joined:
        methods = folders.get(folder.dataset.name, None)
        if methods is None:
            folders[folder.dataset.name] = {folder.search_method: folder}
        else:
            dirs = methods.get(folder.search_method, None)
            if dirs is None:
                methods[folder.search_method] = folder
            elif isinstance(dirs, ResultFolder):
                methods[folder.search_method] = [methods[folder.search_method], folder]
            else:
                dirs.append(folder)
    return folders

def print_result_folders(folders: Dict[str, Union[List[ResultFolder], ResultFolder]]):
    for (dataset, methods) in folders.items():
        for method, folder in methods.items():
            if isinstance(folder, list):
                for sub_folder in folder:
                    print(f"{os.path.split(sub_folder.dir_path)[1]}: {sub_folder.dir_path}")
            else:
                print(f"{method}[{dataset}]: {folder.dir_path}")
    print()

def load_result_folders(
        ignore_datasets: List[Builtin] = None, 
        ignore_methods: List[str] = None,
        select_versions: Dict[str, Dict[str, Union[int, dict]]] = None, 
        print_results=True, 
        sort_fn: Callable[[ResultFolder], Any] = None, 
        filter_fn: Callable[[ResultFolder], bool] = None, 
        reverse=False,
        load_all_unique_info_folders=False,
        load_all_folder_versions=False,
        ignore_with_info_filter: Callable[[dict], bool] = None) -> Dict[str, Dict[str, Union[List[ResultFolder], ResultFolder]]]:

    result_dir = data_dir(add="test_results")
    if select_versions is not None:
        select_versions = {key.upper(): v for key, v in select_versions.items()}
    if ignore_methods is None:
        ignore_methods = tuple()
    ignore_datasets = tuple() if ignore_datasets is None else tuple(map(lambda s: s.upper(), ignore_datasets))
    results: Dict[str, Dict[str, Union[List[ResultFolder], ResultFolder]]] = {}

    for test in os.listdir(result_dir):
        path = os.path.join(result_dir, test)

        array = test.split("[")
        method, remainder = array[0].strip(), array[1].strip()

        array = remainder.split("]")
        info, remainder = array[0].strip(), array[1].strip()

        if ';' in info:
            info = info.split(';')
            dataset, info = info[0].strip().upper(), info[1].strip()
            info = info.split(',')
            info = [v.split('=') for v in info]
            info = {tup[0]: tup[1] for tup in info}
        else:
            dataset = info.strip().upper()
            info = None

        version = re.findall(r'(\d+)', remainder)
        version = int(version[0]) if len(version) else 0

        if (dataset in ignore_datasets) or (method in ignore_methods):
            continue
        elif ignore_with_info_filter is not None and ignore_with_info_filter(info):
            continue

        if method == "KSearchOptuna":
            k_space_method = info["method"]
            method = f"KSearch{k_space_method}"

        new_folder = ResultFolder(path, Builtin[dataset], method, result_dir, version, info)
        dataset_results = results.get(dataset, None)
        
        if dataset_results is None: 
            folder = select_version(new_folder, select_versions=select_versions)
            if folder is not None:
                results[dataset] = {method: [folder]}
            continue
        
        result = dataset_results.get(method, None)

        if result is None:
            folder = select_version(new_folder, select_versions=select_versions)
            if folder is not None:
                dataset_results[method] = [folder]
        elif load_all_unique_info_folders or load_all_folder_versions:
            if load_all_folder_versions:
                result.append(new_folder)
            elif load_all_unique_info_folders:
                dupes = tuple(filter(lambda d: d.info_hash() == new_folder.info_hash(), result))
                if len(dupes) == 1:
                    folder = select_version(dupes[0], new_folder, return_current_as_default=False)
                    if (folder is not None) and (folder != dupes[0]):
                        result.append(folder)
                elif len(dupes) > 1:
                    raise RuntimeError(f"To many non-unique info({new_folder.info}) folders: [{dupes}]")
                else:
                    result.append(new_folder)
        else:
            folder = select_version(result[0], new_folder, select_versions)
            if folder is not None:
                dataset_results[method] = [new_folder]
    
    for (dataset, methods) in results.items():
        for key in methods.keys():
            folders = methods[key]
            if len(folders) == 1:
                methods[key] = methods[key][0]
    
    if sort_fn:
        results = sort_folders(results, fn=sort_fn, filter_fn=filter_fn, reverse=reverse)
    
    if print_results:
        print_result_folders(results)
    return results

@dataclass
class EvalMetrics:
    folders: Dict[str, Dict[str, ResultFolder]]
    results: Dict[str, Dict[str, dict]]
    mean_accs: pd.DataFrame
    max_accs: pd.DataFrame
    normalized_scores: pd.DataFrame
    mean_normalized_scores: pd.DataFrame
    agg_ns_scores: pd.DataFrame
    agg_mns_scors: pd.DataFrame
    nas: pd.DataFrame
    nams: pd.DataFrame
    js: pd.DataFrame
    method_names: List[str]
    w_nas: float
    friedman_subscores: pd.DataFrame
    friedman_stats: Any

    def get_method_names(self) -> List[str]:
        names = set()
        for d in self.results.values():
            for (method, folder) in d.items():
                names.add(method)
        return tuple(names)

    def get_reg_results(self) -> Dict[str, Dict[str, dict]]:
        return {k: v for (k, v) in self.results.items() if Builtin[k].info().task == Task.REGRESSION}
    
    def get_cls_results(self) -> Dict[str, Dict[str, dict]]:
        return {k: v for (k, v) in self.results.items() if Builtin[k].info().task in (Task.BINARY, Task.MULTICLASS)}
    
    def get_reg_folders(self) -> Dict[str, Dict[str, dict]]:
        return {k: v for (k, v) in self.folders.items() if Builtin[k].info().task == Task.REGRESSION}
    
    def get_cls_results(self) -> Dict[str, Dict[str, dict]]:
        return {k: v for (k, v) in self.folders.items() if Builtin[k].info().task in (Task.BINARY, Task.MULTICLASS)}


def calc_eval_metrics(w_nas: float, *load_folder_args, **load_folders_kwargs) -> EvalMetrics:
    data = load_result_folders(*load_folder_args, **load_folders_kwargs)
    results: Dict[str, Dict[str, dict]] = {dataset: {} for dataset in data.keys()}
    normalized_scores: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    mean_normalized_scores: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    mean_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    max_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    datasets_max_acc: Dict[str, float] = {dataset: 0 for dataset in data.keys()}
    datasets_max_mean_acc: Dict[str, float] = {dataset: 0 for dataset in data.keys()}
    dataset_methods_names: Dict[str, list] = {}

    methods_names = []
    printed_newline = False
    for dataset, methods in data.items():
        for (method, folder) in methods.items():
            if isinstance(folder, list):
                raise ValueError(f"Calculate eval metrics with multiple results folders ({folder})")

            file_data = load_json(os.path.join(folder.dir_path, "result.json"))

            if ('result' not in file_data.keys()) or (folder.search_method.startswith("KSearch")):
                if not printed_newline:
                    print()
                    printed_newline = True

                print(f"Result for {method} on the {dataset} dataset, doesn't exist. Computing it")
                searcher = getattr(benchmark, folder.search_method, None)

                if (searcher is None) and folder.search_method.startswith("KSearch"):
                    searcher = KSearchOptuna
                if searcher is None:
                    raise RuntimeError(f"Unable to find search method class {folder.search_method}")
                file_data = searcher.recalc_results(folder.dir_path)
                
            if abs(datasets_max_acc[dataset]) < abs(file_data["result"]["max_test_acc"]):
                datasets_max_acc[dataset] = file_data["result"]["max_test_acc"]
            if abs(datasets_max_mean_acc[dataset]) < abs(file_data["result"]["mean_test_acc"]):
                datasets_max_mean_acc[dataset] = file_data["result"]["mean_test_acc"]
            
            mean_accs[dataset][method] = file_data["result"]["mean_test_acc"]
            max_accs[dataset][method] = file_data["result"]["max_test_acc"]
            results[dataset][method] = file_data

            names = dataset_methods_names.get(dataset, None)
            if names is None:
                names = [method]
                dataset_methods_names[dataset] = names
            else:
                names.append(method)
            
            if len(names) > len(methods_names):
                methods_names = names
    
    if printed_newline:
        print()

    friedman_subscores = {d: {} for d in dataset_methods_names.keys()}

    for dataset, methods in results.items():
        if len(methods) < len(methods_names):
            missing = set(methods_names) - set(methods.keys())
            raise ValueError(f"Dataset {dataset} is missing results from {len(methods)} methods named: {missing}")

        for (method, result) in methods.items():
            max_dataset = datasets_max_acc[dataset]
            max_mean_dataset = datasets_max_mean_acc[dataset]
            max_method = result["result"]["max_test_acc"]
            mean_method = result["result"]["mean_test_acc"]

            ns = max_method / max_dataset
            mns = mean_method / max_mean_dataset
            normalized_scores[dataset][method] = ns
            mean_normalized_scores[dataset][method] = mns

            subset_score = w_nas * ns + (1 - w_nas) *  mns
            friedman_subscores[dataset][method] = subset_score
    
    mean_frame = pd.DataFrame.from_dict(mean_accs, orient='index')
    max_frame = pd.DataFrame.from_dict(max_accs, orient='index')
    norm_frame = pd.DataFrame.from_dict(normalized_scores, orient='index')
    mean_norm_frame = pd.DataFrame.from_dict(mean_normalized_scores, orient='index')

    agg_ns_scores = norm_frame.sum(axis=0)
    agg_mns_scores = mean_norm_frame.sum(axis=0)

    d_n = len(data)
    nas = agg_ns_scores / d_n
    nams = agg_mns_scores / d_n
    js = w_nas * nas + (1 - w_nas) * nams

    subscore_frame = pd.DataFrame.from_dict(friedman_subscores, orient='index')
    subscore_numpy = subscore_frame.to_numpy()

    if subscore_frame.shape[1] > 3:
        friedman_stats = friedmanchisquare(*subscore_numpy)
    else:
        friedman_stats = None

    return EvalMetrics(
        data, results, 
        mean_frame, max_frame, 
        norm_frame, mean_norm_frame,
        agg_ns_scores, agg_mns_scores, 
        nas, nams, js,
        methods_names, w_nas,
        subscore_frame, friedman_stats
    )

def time_frame(data: EvalMetrics) -> pd.DataFrame:
    time_dict = dict()
    for _, results in data.results.items():
        for method, result in results.items():
            method_result = time_dict.get(method, None)
            if method_result is None:
                time_dict[method] = [result["result"]["time"]]
            else:
                method_result.append(result["result"]["time"])
    
    frame = pd.DataFrame.from_dict(time_dict)
    frame.index = tuple(data.results.keys())

    frame = frame.sort_values(by=f"{frame.columns[0]}")
    return frame

def time_frame_pct(data: EvalMetrics) -> pd.DataFrame:
    frame = time_frame(data)
    mins = frame.min(axis=1)
    norm = frame.div(mins, axis='index')
    pct = norm * 100
    return pct

def time_frame_stamps(data: EvalMetrics) -> pd.DataFrame:
    frame = time_frame(data)
    return frame.map(BaseSearch.time_to_str)

def time_frame_deltas(data: EvalMetrics) -> pd.DataFrame:
    frame = time_frame(data)

    # sort rows by by time to train, biggest first
    maxs = list(frame.sum(axis=1).sort_values(ascending=False).index)
    frame = frame.reindex(index=maxs)
    
    columns = [col for col in frame.columns if col != "NOSearch"]
    mins = frame[columns].min(axis=1)

    deltas = frame.copy()
    deltas.drop(columns="NOSearch", inplace=True)
    deltas.columns = [f"{col}_delta" for col in deltas.columns]
    deltas = deltas.subtract(mins, axis=0)

    combined = pd.concat([frame, deltas], axis=1)

    return combined


def friedman_check(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True, confidence: float = 0.05):
    friedman = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, sort_fn=sort_fn, reverse=sort_reverse, print_results=False
    ).friedman_stats

    if friedman.pvalue < confidence:
        print("There is a statistically significant difference between the tested algorithms.")
    else:
        print("There isn't a statistically significant difference between the tested algorithms.")

    
if __name__ == "__main__":
    metrics = calc_eval_metrics(0.5, print_results=False)
    frame = time_frame_deltas(metrics)
    print(frame)



            
            

