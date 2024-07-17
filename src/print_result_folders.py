import os
from Util import Dataset, Builtin, data_dir, Task, get_n_search_space
from Util.io_util import load_json, json_to_str, load_csv
from Util.compat import removeprefix
from benchmark import BaseSearch
from typing import List, Dict, Tuple, Callable, Any, Union, Optional
import re
from dataclasses import dataclass
import pandas as pd
import hashlib
from itertools import chain
from numbers import Number
from calc_metrics import ResultFolder, EvalMetrics, load_result_folders, calc_eval_metrics
import logging

TY_FOLDERS = Dict[str, Dict[str, Union[List[ResultFolder], ResultFolder]]]

def get_kspace_base_method_results(folder: ResultFolder, folders: TY_FOLDERS, file_data: dict = None) -> Optional[dict]:
    base_test = None
    ver_regex = r'V\d+$'
    base_method = removeprefix(folder.search_method, "KSpace")
    base_method = re.sub(r'V\d+$', '', base_method)

    results = folders[folder.dataset.name.upper()].get(base_method)
    if isinstance(results, ResultFolder):
        base_folder = results
    elif isinstance(results, list):
        results.sort(
            key=lambda f: (
                int(f.info["nrepeat"] == folder.info["nrepeat"]), 
                int(f.info["nparams"] == folder.info["nparams"])
            ), 
            reverse=True
        )
        base_folder = results[0]
        if base_folder.info["nrepeat"] != folder.info["nrepeat"]:
            logging.warning(
                f"Non maching nrepeats ({folder.info["nrepeats"]}) for {folder.search_method}"
                f" (ver {folder.version}): {base_folder.search_method} (ver {base_folder.version})"
                f" which have {base_folder.info["nrepeats"]} nrepeats."
            )

    base_results = load_json(os.path.join(base_folder.dir_path, "result.json"))
    return base_results

def dict_str(dct: dict, include_bracets=True) -> str:
    dct_str = ",".join(f"{k}={v}" for k, v in dct.items())
    return f"{{dct_str}}" if include_bracets else dct_str

def load_data(folder: ResultFolder, data: TY_FOLDERS, skip_unfinished=True):
    results_path = os.path.join(folder.dir_path, "result.json")
    history_path = os.path.join(folder.dir_path, "history.csv")
    file_data = load_json(results_path, default={}) 

    base_results = get_kspace_base_method_results(folder, data, file_data)
    if base_results is not None and ('result' in base_results):
        base_test = base_results["result"]["mean_test_acc"]
    else:
        base_test = None

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
    
    base_diff = (test_ - base_test) if base_test is not None else None
    return train_, test_, time_, base_diff, file_data["info"] 

def info_str(folder: ResultFolder, data, is_sub_folder=False, prev_n_k=None) -> str: 
    train_, test_, time_, _base_diff, info = data

    if 'k' in info["method_params"]:
        info_k = f", k=(" + dict_str(info["method_params"]["k"], False) + ")"
    else:
        info_k = ""

    if folder.info is not None:
        info_str = "[" + dict_str(folder.info, include_bracets=False) + info_k
        info_str += f"] (ver {folder.version}): " if folder.version > 0 else "]:"
    else:
        info_str = ""

    n_k = len(info["method_params"].get("k", {}))
    if (prev_n_k is not None) and (n_k > prev_n_k) and is_sub_folder:
        prefix = "\n         "
        info_str = "\n      " + info_str
    elif is_sub_folder:
        prefix = "\n         "
    else:
        prefix = " "

    if any(v is None for v in (train_, test_, time_)):
        return info_str + prefix + f" unfinished"
    
    base_diff = f", base_diff={_base_diff}" if _base_diff is not None else ""
    result = info_str + prefix + f"train={train_}, test={test_}{base_diff}, time={BaseSearch.time_to_str(time_)}" 
    return result, (None if n_k < 0 else n_k) 

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

    for (dataset, methods) in data.items():
        strings = []
        for method, folder in methods.items():
            if isinstance(folder, list):
                # sort by n_kspace_params, then test_score
                file_datas = [load_data(d, data, skip_unfinished) for d in folder]
                joined = list(zip(folder, file_datas))
                joined.sort(key=lambda tup: (len(tup[1][-1]["method_params"].get("k", {})), tup[1][1]))
                dirs_sorted, datas_sorted = list(zip(*joined))

                sub_strings = []
                n_k = None
                for i, f in enumerate(dirs_sorted):
                    sub_string, n_k = info_str(f, is_sub_folder=True, data=datas_sorted[i], prev_n_k=n_k)     
                    sub_strings.append(sub_string)

                sub_strings = '\n      ' + f'\n      '.join(sub_strings)
                strings.append(f"   {method}:{sub_strings}")
            else:
                strings.append(f"   {method}:\n      {info_str(folder, data=load_data(folder, data, skip_unfinished))[0]}")

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