import os
from Util import Dataset, Builtin, data_dir, Task
from Util.io_util import load_json
from benchmark import BaseSearch
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
import pandas as pd

@dataclass
class ResultFolder:
    dir_path: str
    dataset: Builtin
    search_method: str
    version: int = 0

def select_version(current: ResultFolder, new_version: int, select_versions: Dict[str, Dict[str, int]] = None) -> bool:
    versions = select_versions.get(current.dataset.name, None) if select_versions is not None else None
    version = versions.get(current.search_method, None) if versions is not None else None
    if versions is not None:
        return new_version == version
    return current.version < new_version
    

def load_result_folders(ignore_datasets: List[Builtin] = None, select_versions: Dict[str, Dict[str, int]] = None) -> Dict[str, Dict[str, ResultFolder]]:
    result_dir = data_dir(add="test_results")
    if select_versions is not None:
        select_versions = {key.upper(): v for key, v in select_versions.items()}

    ignore_datasets = tuple() if ignore_datasets is None else tuple(map(lambda s: s.upper(), ignore_datasets))
    results: Dict[str, Dict[str, ResultFolder]] = {}

    for test in os.listdir(result_dir):
        path = os.path.join(result_dir, test)

        array = test.split("[")
        method, remainder = array[0].strip(), array[1].strip()
        array = remainder.split("]")
        dataset, remainder = array[0].strip().upper(), array[1].strip()
        version = re.findall(r'(\d+)', remainder)
        version = int(version[0]) if len(version) else 0

        dataset_results = results.get(dataset, None)

        if dataset in ignore_datasets:
            continue
        elif dataset_results is None: 
            results[dataset] = {method: ResultFolder(path, Builtin[dataset], method, version)}
        else:
            result = dataset_results.get(method, None)
            if result is None:
                dataset_results[method] = ResultFolder(path, Builtin[dataset], method, version)
            elif select_version(result, version, select_versions):
                result.dir_path = path
                result.version = version
    
    return results

@dataclass
class EvalMetrics:
    folders: Dict[str, Dict[str, ResultFolder]]
    results: Dict[str, Dict[str, dict]]
    mean_accs: pd.DataFrame
    mean_ranks: pd.DataFrame
    max_accs: pd.DataFrame
    max_ranks: pd.DataFrame
    normalized_scores: pd.DataFrame
    agg_scores: pd.DataFrame
    rank_scores: pd.DataFrame
    nas: pd.DataFrame
    nrs: pd.DataFrame
    js: pd.DataFrame

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


def calc_eval_metrics(data: Dict[str, Dict[str, ResultFolder]]) -> EvalMetrics:
    results: Dict[str, Dict[str, dict]] = {dataset: {} for dataset in data.keys()}
    normalized_scores: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    mean_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    max_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    datasets_max_acc: [str, float] = {dataset: 0 for dataset in data.keys()}

    for dataset, methods in data.items():
        for (method, folder) in methods.items():
            file_data = load_json(os.path.join(folder.dir_path, "result.json"))

            if 'result' not in file_data.keys():
                print(f"Result for {method} on the {dataset} dataset, doesn't exist. Computing it")
                file_data = BaseSearch.recalc_results(folder.dir_path)

            if abs(datasets_max_acc[dataset]) < abs(file_data["result"]["max_test_acc"]):
                datasets_max_acc[dataset] = file_data["result"]["max_test_acc"]
            
            mean_accs[dataset][method] = file_data["result"]["mean_test_acc"]
            max_accs[dataset][method] = file_data["result"]["max_test_acc"]
            results[dataset][method] = file_data

    for dataset, methods in results.items():
        for (method, result) in methods.items():
            max_dataset = datasets_max_acc[dataset]
            max_method = result["result"]["max_test_acc"]
            ns = max_method / max_dataset
            normalized_scores[dataset][method] = ns
    
    mean_frame = pd.DataFrame.from_dict(mean_accs, orient='index')
    mean_ranks = mean_frame.rank(axis=1)

    max_frame = pd.DataFrame.from_dict(max_accs, orient='index')
    max_ranks = mean_frame.rank(axis=1)

    norm_frame = pd.DataFrame.from_dict(normalized_scores)

    agg_scores = norm_frame.sum(axis=1)
    agg_min = agg_scores.min()
    rank_scores = mean_ranks.sum(axis=0)
    rank_min = rank_scores.min()

    as_min = agg_scores.copy() - agg_min
    rs_min = rank_scores.copy() -  rank_min

    nas = as_min / agg_scores
    nrs = rs_min / rank_scores

    js = nas + nrs

    return EvalMetrics(
        data,
        results,
        mean_frame, mean_ranks, max_frame, max_ranks,
        norm_frame, agg_scores, rank_scores,
        nas, nrs, js
    )

def time_frame(data: EvalMetrics) -> pd.DataFrame:
    columns = ["Dataset"]
    columns.extend(data.get_method_names())

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
    return frame

def time_frame_pct(data: EvalMetrics) -> pd.DataFrame:
    frame = time_frame(data)
    mins = frame.min(axis=1)
    norm = frame.div(mins, axis='index')
    pct = norm * 100
    return pct

if __name__ == "__main__":
    ignore_datasets = (Builtin.AIRLINES.name, )
    result_folders = load_result_folders(ignore_datasets)

    for method, result in result_folders.items():
        datasets = tuple(result.keys())
        print(f"{method}: {datasets}, len={len(datasets)}")
    
    print()
    
    metrics = calc_eval_metrics(result_folders)
    



            
            

