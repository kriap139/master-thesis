from Util import Dataset, find_dir_ver, load_csv, save_csv, Integer, Real, Categorical, find_file_ver, CVInfo, Builtin
from Util.io_util import load_json, save_json, data_dir, json_to_str
import os
from typing import Dict

OLD_PREFIX = "_folds.json"

def get_old_fn(dataset: Dataset):
    return f"{dataset.name}{OLD_PREFIX}"

def get_renamed_files() -> Dict[Dataset, str]:
    result = {}

    for bn in Builtin:
        try:
            dataset = Dataset(bn)
        except Exception:
            print(f"Dataset {bn.name} not found")
            continue

        old_path = os.path.join(dataset.get_dir(), get_old_fn(dataset))
        if os.path.exists(old_path):
            data = load_json(old_path)
            cv_info = CVInfo(data["info"])
            new_fn = cv_info.fn()
            result[dataset] = new_fn

            if 'is_old' not in data["info"]:
                data["info"]["is_old"] = True
                save_json(old_path, data, indent=None, overwrite=True)

    return result

def print_renamed_files():
    files = get_renamed_files()
    for dataset, fn in files.items():
        print(f"{dataset.name}: {get_old_fn(dataset)} -> {fn}")

def rename():
    data = get_renamed_files()
    for dataset, fn in data.items():
        old_path = os.path.join(dataset.get_dir(), get_olf_fn(dataset))
        new_path = os.path.join(dataset.get_dir(), fn)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    print_renamed_files()
