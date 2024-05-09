from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, load_json
from calc_metrics import load_result_folders
import logging
from benchmark import BaseSearch, RandomSearch
import os
import numpy as np
import shutil

def main():
    _dir = data_dir(add="test_results")
    data = load_result_folders(
        ignore_datasets=("airlines", "sgemm_gkp"),
        print_results=False
    )

    dist_dir = data_dir("zips/old_folders_name_fix", make_add_dirs=True)
    os.makedirs(dist_dir, exist_ok=True)

    dicts = [methods for methods in data.values()]
    folders = []
    for d in dicts:
        folders.extend(d.values())
    
    for folder in folders:
        result = load_json(os.path.join(folder.dir_path, "result.json"))

        n_repeats = result['info'].get("cv", {}).get("n_repeats", 0)
        n_params = len(result['info']["space"])

        str_repeats = f"nrepeat={n_repeats}" if n_repeats > 0 else ""
        str_version = f" ({folder.version})" if folder.version > 0 else ""

        new_name = f"{folder.search_method}[{folder.dataset.name.lower()};" + str_repeats + f",nparams={n_params}]" + str_version
        old_name = os.path.split(folder.dir_path)[1]
        print(f"{old_name}      {new_name}")

        shutil.copytree(folder.dir_path, os.path.join(dist_dir, new_name))

        




if __name__ == "__main__":
    main()