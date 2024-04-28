import os
from Util import Dataset, Builtin, data_dir, Task, get_n_search_space
from Util.io_util import load_json, json_to_str, load_csv
from Util.compat import removeprefix
from benchmark import BaseSearch, AdjustedSeqUDSearch
from typing import List, Dict, Tuple, Callable, Any, Union, Optional
import re
from dataclasses import dataclass
import pandas as pd
import hashlib
from itertools import chain
from numbers import Number

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
            test = ResultFolder(current.dir_path, current.dataset, current.search_method, version, select)

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
            dataset = info.strip()
            info = None

        version = re.findall(r'(\d+)', remainder)
        version = int(version[0]) if len(version) else 0

        new_folder = ResultFolder(path, Builtin[dataset], method, version, info)
        dataset_results = results.get(dataset, None)

        if (dataset in ignore_datasets) or (method in ignore_methods):
            continue
        elif ignore_with_info_filter is not None and ignore_with_info_filter(info):
            continue
        elif dataset_results is None: 
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
            folder = select_version(result, new_folder, select_versions)
            if folder is not None:
                dataset_results[method] = new_folder
    
    for (dataset, methods) in results.items():
        for key in methods.keys():
            folders = methods[key]
            if len(folders) == 1:
                methods[key] = methods[key][0]
    
    if sort_fn:
        results = sort_folders(results, fn=sort_fn, filter_fn=filter_fn, reverse=reverse)
    
    if print_results:
        for (dataset, methods) in results.items():
            for method, folder in methods.items():
                if isinstance(folder, list):
                    for sub_folder in folder:
                        print(f"{os.path.split(sub_folder.dir_path)[1]}: {sub_folder.dir_path}")
                else:
                    print(f"{method}[{dataset}]: {folder.dir_path}")
        print()
    
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


def calc_eval_metrics(ignore_datasets: List[Builtin] = None, ignore_methods: List[str] = None) -> EvalMetrics:
    data = load_result_folders(ignore_datasets, ignore_methods)
    results: Dict[str, Dict[str, dict]] = {dataset: {} for dataset in data.keys()}
    normalized_scores: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    mean_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    max_accs: Dict[str, Dict[str, float]] = {dataset: {} for dataset in data.keys()}
    datasets_max_acc: [str, float] = {dataset: 0 for dataset in data.keys()}

    printed_newline = False
    for dataset, methods in data.items():
        for (method, folder) in methods.items():
            if isinstance(folder, list):
                raise ValueError(f"Calculate eval metrics with multiple results folders ({folder})")

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

def print_folder_results(
    ignore_datasets: List[str] = None, 
    ignore_methods: List[str] = None, 
    ignore_with_info_filter: Callable[[dict], bool] = None, 
    skip_unfinished=True, 
    load_all_unique_info_folders=True, 
    load_all_folder_versions=True):

    folder_sorter = lambda folder: tuple(chain.from_iterable(
        [
            (folder.search_method, folder.dataset.name, folder.info is not None, folder.info.get("nparams", "")),
            folder.info.get('k', {}).values()
        ]
    ))

    data = load_result_folders(
        ignore_datasets=ignore_datasets,
        ignore_methods=ignore_methods,
        load_all_unique_info_folders=load_all_unique_info_folders, 
        load_all_folder_versions=load_all_folder_versions, 
        sort_fn=folder_sorter,
        ignore_with_info_filter=ignore_with_info_filter,
        print_results=False
    )

    def load_data(folder: ResultFolder):
        results_path = os.path.join(folder.dir_path, "result.json")
        history_path = os.path.join(folder.dir_path, "history.csv")
        file_data = load_json(results_path, default={}) 

        base_test = None
        ver_regex = r'V\d+$'
        base_method = removeprefix(folder.search_method, "KSpace")
        base_method = re.sub(r'V\d+$', '', base_method)
        base_dir = f"{base_method}[{folder.dataset.name.lower()};nparams={get_n_search_space(base_method)}]"
        base_path = data_dir(add=f"test_results/{base_dir}")
        if os.path.exists(base_path):
            base_results = load_json(os.path.join(base_path, "result.json"))
            if 'result' in base_results:
                base_test = base_results["result"]["mean_train_acc"]

        if ('result' not in file_data) and not skip_unfinished:
            df = load_csv(history_path)
            train_ = df["train_score"].mean()
            test_ = df["test_score"].mean()
            time_ = df["time"].mean()
        elif 'result' in file_data:
            train_ = file_data["result"]["mean_train_acc"]
            test_ = file_data["result"]["mean_test_acc"]
            time_ = file_data["result"]["time"]
        else:
            return None, None, None, base_test, file_data["info"]
        return train_, test_, time_, (test_ - base_test), file_data["info"] 
    
    def dict_str(dct: dict, include_bracets=True) -> str:
        dct_str = ",".join(f"{k}={v}" for k, v in dct.items())
        return f"{{dct_str}}" if include_bracets else dct_str

    def info_str(folder: ResultFolder, is_sub_folder=False, data=None, prev_n_k=None) -> str: 
        train_, test_, time_, _base_diff, info = load_data(folder) if data is None else data

        if 'k' in info["method_params"]:
            info_k = f", k=(" + dict_str(info["method_params"]["k"], False) + ")"
        else:
            info_k = ""

        if folder.info is not None:
            info_str = "[" + dict_str(folder.info, include_bracets=False) + info_k
            info_str += f"] ({folder.version}): " if folder.version > 0 else "]:"
        else:
            info_str = ""

        n_k = len(info["method_params"].get("k", {}))
        if (prev_n_k is not None) and (n_k > prev_n_k) and is_sub_folder:
            prefix = "\n\n         "
            info_str = "\n" + info_str
        elif is_sub_folder:
            prefix = "\n         "
        else:
            prefix = ""

        if any(v is None for v in (train_, test_, time_)):
            return info_str + prefix + f" unfinished"
        
        base_diff = f", base_diff={_base_diff}" if _base_diff is not None else ""
        result = info_str + prefix + f"train={train_}, test={test_}{base_diff}, time={round(time_)}s, time={BaseSearch.time_to_str(time_)}" 
        return result, (None if n_k < 0 else n_k) 

    for (dataset, methods) in data.items():
            strings = []
            for method, folder in methods.items():
                if isinstance(folder, list):
                    # sort by n_kspace_params, then test_score
                    file_datas = [load_data(d) for d in folder]
                    joined = list(zip(folder, file_datas))
                    joined.sort(key=lambda tup: (len(tup[1][-1]["method_params"].get("k", {})), tup[1][1]))
                    dirs_sorted, datas_sorted = list(zip(*joined))

                    sub_strings = []
                    n_k = 0
                    for i, f in enumerate(dirs_sorted):
                        sub_string, n_k = info_str(f, is_sub_folder=True, data=datas_sorted[i], prev_n_k=n_k)     
                        sub_strings.append(sub_string)

                    sub_strings = '\n      ' + f'\n      '.join(sub_strings)
                    strings.append(f"   {method}:{sub_strings}")
                else:
                    strings.append(f"   {method}:\n      {info_str(folder)}")

            print(f"{dataset}: \n" + "\n".join(strings) + '\n')
            strings.clear()

def print_untesed_kspace_combos(
    ignore_datasets: List[str] = None, 
    ignore_methods: List[str] = None, 
    ignore_with_info_filter: Callable[[dict], bool] = None, 
    skip_unfinished=True,
    print_folders_loaded=True):

    folder_sorter = lambda folder: tuple(chain.from_iterable(
        [
            (folder.search_method, folder.dataset.name, folder.info is not None, folder.info.get("nparams", "")),
            folder.info.get('k', {}).values()
        ]
    ))

    
    data = load_result_folders(
        ignore_datasets=ignore_datasets,
        ignore_methods=ignore_methods,
        load_all_unique_info_folders=True, 
        load_all_folder_versions=True, 
        sort_fn=folder_sorter,
        ignore_with_info_filter=ignore_with_info_filter,
        print_results=print_folders_loaded
    )

    for (dataset, methods) in data.items():
        strings = []
        for method, folder in methods.items():
            folders = (folder, ) if isinstance(folder, ResultFolder) else folder
            
            # Every result folder should have a results file with an info section created before training!
            file_infos = [load_json(os.path.join(d.dir_path, "result.json"))["info"] for d in folders]
            tested_params = [
                info["method_params"]["k"] 
                for info in file_infos
                if 'k' in info["method_params"]
            ]            
            
            file_path = data_dir(add=f"kspace_values_{dataset.lower()}.json")
            kspace_params = [params['k'] for params in load_json(file_path, default=[])]
            #print(file_path)
            if len(tested_params) > 0:
                untested = tuple(filter(lambda tup: not any(tup[1] == p for p in tested_params), enumerate(kspace_params)))
                sub_strings = '\n      ' + f'\n      '.join([f"{tup[0]} {tup[1]}" for tup in untested])
                strings.append(f"   {method}(n_tested={len(tested_params)}, n_kspace={len(kspace_params)}, n_untested={len(untested)}):{sub_strings}")

        print(f"{dataset}: \n" + "\n".join(strings) + '\n')
        strings.clear()



if __name__ == "__main__":
    ignore_datasets = ()
    ignore_methods = ("KSpaceOptunaSearch", )
    ignore_info_filter = lambda info: ( 
        # Ignore initial tuning results where non-kspace parameters where tuned with kspace parameters by mistake!
        info['nparams'] != info['kparams'] if 'kparams' in info else False
    )
    #metrics = calc_eval_metrics(ignore_datasets)
    print_folder_results(ignore_datasets, ignore_methods, ignore_with_info_filter=ignore_info_filter)
    print("--------------------------------untested kspace combos---------------------------------------------")
    print_untesed_kspace_combos(ignore_datasets, ignore_methods, ignore_info_filter, print_folders_loaded=False)

    



            
            

