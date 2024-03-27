import pandas as pd
import scipy.io as scp
import os
import json
import dataclasses
import numpy as np
from typing import Union, Iterable, List, Any, Tuple, Optional
import csv
import re
from .space import Integer, Real, Categorical
from .sparse_arff import load_sparse_arff
import gc
import glob

def json_serialize_unknown(o):
    if isinstance(o, np.integer):
        return int(o)
    elif dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, Integer):
         return dict(
                cls=o.__class__.__name__, name=o.name, high=o.high, low=o.low, base=o.base, 
                prior=o.prior, transform=o.transform_
            )
    elif isinstance(o, Real):
        return dict(
            cls=o.__class__.__name__, name=o.name, high=o.high, low=o.low, base=o.base, 
            prior=o.prior, transform=o.transform_
        )
    elif isinstance(o, Categorical):
        return dict(cls=o.__class__.__name__, name=o.name, categories=o.categories, prior=o.prior, transform=o.transform_)
    else:
        raise TypeError(f"Object of class {type(o)} is not JSON serializable: {o}")

def find_files(exp: str, recursive=False) -> List[str]:
    return glob.glob(exp, recursive=recursive)

def find_dir_ver(folder: str) -> str:
    if not os.path.exists(folder):
        return folder
    
    head, tail = os.path.split(folder)
    new_dir = os.path.join(head, f"{tail} (1)")

    if not os.path.exists(new_dir):
        return new_dir

    dirs = [tup[0] for tup in os.walk(head) if tail in tup[0]]
    nums = []
    for d in dirs:
        nums.extend(re.findall(r'(\d+)', d))
    
    if len(nums):
        nums.sort(reverse=True)
        num = int(nums[0]) + 1
        new_dir = os.path.join(head, f"{tail} ({num})")  
    else:
        raise RuntimeError(f"Failed to find new directory version for directory: {tail}")
    
    if os.path.exists(new_dir):
        raise RuntimeError(f"Failed to find new directory version as it already exists: {new_dir}")
    
    return new_dir


def find_file_ver(folder: str, name: str) -> str:
    if not os.path.exists(os.path.join(folder, name)):
        return os.path.join(folder, name)

    fn, ext = os.path.splitext(name)

    ver = 1
    fn_new = f"{fn} ({ver}){ext}"

    while True:
        if not os.path.exists(os.path.join(folder, fn_new)):
            break
        ver += 1
        fn_new = f"{fn} ({ver}){ext}"

    return os.path.join(folder, fn_new)

def save_json(fp: str, data: Union[dict, list], indent: int = 3, overwrite=False):
    folder = os.path.dirname(fp)
    os.makedirs(folder, exist_ok=True)
    
    if os.path.isfile(fp) and (not overwrite):
        fp = find_file_ver(folder, os.path.basename(fp))

    with open(fp, mode='w') as f:
        json.dump(data, f, indent=indent, default=json_serialize_unknown)

def json_to_str(data: Union[dict, list], indent=3, sort_keys=False) -> str:
    return json.dumps(data, default=json_serialize_unknown, sort_keys=sort_keys, indent=indent)
    

def load_json(fp: str, default = None):
    if os.path.exists(fp):
        with open(fp, mode='r') as f:
            return json.load(f)
    return default

def save_csv(fp: str, field_names: Iterable[str], data: Union[dict, Iterable[dict]] = None):
    mode = 'a' if os.path.exists(fp) else 'w'

    with open(fp, mode=mode) as f:
        if mode == 'w':
            csv.DictWriter(f, field_names).writeheader()
        
        if data is not None and type(data) == dict:
            csv.DictWriter(f, field_names).writerow(data)
        elif data is not None and isinstance(data, Iterable):
            csv.DictWriter(f, field_names).writerows(data)

def load_csv(fp: str, default = None) -> pd.DataFrame:
    return pd.read_csv(fp)

def has_csv_header(fp: str, n_sample=20) -> bool:
    with open(fp, mode='r') as f:
        head = [next(f) for _ in range(n_sample)]
        head = "".join(head)

        sniffer = csv.Sniffer()
        return sniffer.has_header(head)

def get_n_csv_columns(fp: str) -> int:
    with open(fp, mode='r') as f:
        reader = csv.reader(f)
        ncol = len(next(reader))
        f.seek(0)    
        return ncol

def data_dir(add: str = None, make_add_dirs=True, force_add_is_fn=False, force_add_is_not_fn=False) -> str:
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        raise RuntimeError(f"data directory dosen't exist: {data_dir}")

    if add is not None:
        add.strip()
        add = add[1:] if add.startswith(os.path.sep) else add
        head, tail = os.path.split(add)

        if (tail.__contains__('.') and not force_add_is_not_fn) or force_add_is_fn:
            path = os.path.join(data_dir, head)
        else:
            path = os.path.join(data_dir, add)

        if make_add_dirs:
            os.makedirs(path, exist_ok=True)
        path = os.path.join(data_dir, add)
        
    return path

def load_arff(path: str) -> pd.DataFrame:
    try:
        data = scp.arff.loadarff(path)
        train = pd.DataFrame(data[0])
    except ValueError as e:
        print(f"ValueError: {e}")
        print(f"Trying to load arrf as a sparse_arff file!")
        train = load_sparse_arff(path)
        
    catCols = [col for col in train.columns if train[col].dtype=="O"]
    # print(f"Catcols: {catCols}")

    train[catCols] = train[catCols].apply(lambda x: x.str.decode('utf8'))
    return train

def arff_to_csv(path: str):
    data = load_arff(path)
    print(data.head())

    fn, ext = os.path.splitext(os.path.basename(path))
    fn = f"{fn}.csv"
    print(f"Saving converted arff file: {fn}")
    data.to_csv(os.path.join(os.path.dirname(path), fn),index=False)

if __name__ == "__main__":
    path = data_dir(add="datasets/rcv1/php7t4FlC.arff")
    print(path)