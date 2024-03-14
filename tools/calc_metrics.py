import os
from Util import Dataset, Builtin, data_dir, Task
from Util.io_util import load_json, json_to_str
from benchmark import BaseSearch, AdjustedSeqUDSearch
from typing import List, Dict, Tuple, Callable, Any, Union
import re
from dataclasses import dataclass
import pandas as pd
import hashlib

@dataclass
class ResultFolder:
    dir_path: str
    dataset: Builtin
    search_method: str
    version: int = 0
    info: dict = None

    def info_hash(self) -> str:
        js = json_to_str(self.info, indent=None, sort_keys=True).encode()
        dhash = hashlib.md5()
        dhash.update(js)
        return dhash.hexdigest()

def select_new_version(current: ResultFolder, new: ResultFolder, select_versions: Dict[str, Dict[str, Union[int, dict]]] = None) -> bool:
    data = select_versions.get(current.dataset.name, None) if select_versions is not None else None
    select = data.get(current.search_method, None) if data is not None else None

    if select is not None:
        if type(select) == int:
            return select == new.version
        elif isinstance(select, dict):
            if 'version' in select.keys():
                select = select.copy()
                version = select.pop('version')
            else:
                version = None

            test = ResultFolder(new.dir_path, new.dataset, new.search_method, new.version, select)
            found = test.info_hash() == new.info_hash()
            if found and (current.info_hash() == new.info_hash()):
                return new == version if version is not None else current.version < new.version
            return found
        else:
            raise RuntimeError(f"Folder selection attribute for folder(dataset={current.dataset.name}, method={current.search_method}) is invalid, only version(int) or info(dict) is supported: {select}")
    else:
        return current.version < new.version

def sort_folders(folders: Dict[str, Dict[str, ResultFolder]], fn: Callable[[ResultFolder], Any], filter_fn: Callable[[ResultFolder], bool] = None, reverse=False) -> Dict[str, Dict[str, ResultFolder]]:
    joined: List[ResultFolder] = []
    for (dataset, methods) in folders.items():
        joined.extend(methods.values())
    
    if filter_fn is not None:
        joined = list(filter(filter_fn, joined))
        
    
    joined.sort(key=fn, reverse=reverse)
    folders.clear()

    for folder in joined:
        methods = folders.get(folder.dataset.name, None)
        if methods is None:
            folders[folder.dataset.name] = {folder.search_method: folder}
        else:
            methods[folder.search_method] = folder
    
    return folders

    

def load_result_folders(
        ignore_datasets: List[Builtin] = None, 
        select_versions: Dict[str, Dict[str, Union[int, dict]]] = None, 
        print_results=True, 
        sort_fn: Callable[[ResultFolder], Any] = None, 
        filter_fn: Callable[[ResultFolder], bool] = None, 
        reverse=False,
        load_all_unique_info_folders=False) -> Dict[str, Dict[str, ResultFolder]]:

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
        info, remainder = array[0].strip().upper(), array[1].strip()

        if ';' in info:
            info = info.split(';')
            dataset, info = info[0].strip(), info[1].strip()
            info = info.split(',')
            info = [v.split('=') for v in info]
            info = {tup[0]: tup[1] for tup in info}
        else:
            dataset = info.strip()
            info = None

        version = re.findall(r'(\d+)', remainder)
        version = int(version[0]) if len(version) else 0

        new_folder = ResultFolder(path, Builtin[dataset], method, version, info)
        dataset_results = results.get(dataset, None)

        if dataset in ignore_datasets:
            continue
        elif dataset_results is None: 
            results[dataset] = {method: new_folder}
        else:
            result = dataset_results.get(method, None)
            if result is None:
                dataset_results[method] = new_folder
            elif load_all_unique_info_folders and (info is not None or isinstance(result, dict)):
                is_dict = isinstance(result, dict)
                new_hash = new_folder.info_hash()

                # The folder does not have identical parameters (info_hashes not equal)
                if not is_dict and (new_hash != result.info_hash()):
                    dataset_results[method] = {result.info_hash(): result, new_hash: new_folder}
                elif not is_dict and (new_hash == result.info_hash()):
                    dataset_results[method] = {result.info_hash(): result} if select_new_version(result, folder, select_versions) else {new_hash: new_folder}
                # The folder with the current info does not already exists
                elif not new_hash in result.keys():
                    result[new_hash] = new_folder
                # The folder with the current info does already exists, so pick the newest (or provided) version!
                elif new_hash in result.keys() and select_new_version(result.get(new_hash), new_folder, select_versions):
                    result[new_hash] = new_folder

            elif select_new_version(result, new_folder, select_versions):
                dataset_results[method] = new_folder

    if sort_fn:
        results = sort_folders(results, fn=sort_fn, filter_fn=filter_fn, reverse=reverse)
    
    if print_results:
        for (dataset, methods) in results.items():
            strings = []
            for method, folder in methods.items():
                if isinstance(folder, dict):
                    sub_strings = []
                    for sub_folders in folder.values():
                        sub_strings.extend(f"\t [{k}={v}]" for k, v in sub_folders.info.items())
                    sub_strings = f'\n'.join(sub_strings)
                    strings.append(f"\t{method}: \t{sub_strings}")
                else:
                    strings.append(f"\t{method}: {folder.dir_path}")
            print(f"{dataset}: \n" + "\n".join(strings))
            strings.clear()
    
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


def calc_eval_metrics(ignore_datasets: List[Builtin] = None) -> EvalMetrics:
    data = load_result_folders(ignore_datasets)
    results: Dict[str, Dict[str, dict]] = {dataset: {} for dataset in data.keys()}
    normalized_scores: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    mean_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    max_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    datasets_max_acc: [str, float] = {dataset: 0 for dataset in data.keys()}

    printed_newline = False
    for dataset, methods in data.items():
        for (method, folder) in methods.items():
            if isinstance(folder, dict):
                raise ValueError(f"Calculate eval metrics with multiple results ({folder})")

            file_data = load_json(os.path.join(folder.dir_path, "result.json"))

            if 'result' not in file_data.keys():
                if not printed_newline:
                    print()
                    printed_newline = True

                print(f"Result for {method} on the {dataset} dataset, doesn't exist. Computing it")
                file_data = BaseSearch.recalc_results(folder.dir_path)

            if abs(datasets_max_acc[dataset]) < abs(file_data["result"]["max_test_acc"]):
                datasets_max_acc[dataset] = file_data["result"]["max_test_acc"]
            
            mean_accs[dataset][method] = file_data["result"]["mean_test_acc"]
            max_accs[dataset][method] = file_data["result"]["max_test_acc"]
            results[dataset][method] = file_data
    
    if printed_newline:
        print()

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

def print_all_adjusted_sequd_results():
    data = load_result_folders(load_all_unique_info_folders=True)
    for dataset, methods in data.items():
        for folders in methods[AdjustedSeqUDSearch.__name__]:
            folders_ = (folders, ) if isinstance(folders, ResultFolder) else folders
        
            for folder in folders_:
                file_data = load_json(os.path.join(folder.dir_path, "result.json"))
                train_ = file_data["result"]["mean_train_acc"]
                test_ = file_data["result"]["mean_test_acc"]
                time_ = file_data["result"]["time"]

                info_str = "[" + f",".join(f"{k}={v}" for k, v in folder.info) + "]" if folder.info is not None else ""
                print(f"{AdjustedSeqUDSearch.__name__}{info_str}: train={train_}, test={test_}, time={time_}")


if __name__ == "__main__":
    ignore_datasets = ()
    #metrics = calc_eval_metrics(ignore_datasets)
    print_all_adjusted_sequd_results()
    



            
            

