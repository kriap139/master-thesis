import pandas as pd
from scipy.io import arff
import os
import json
import dataclasses
import numpy as np
from typing import Union

def json_serialize_unknown(o):
    if isinstance(o, np.integer):
        return int(o)
    elif dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of class {type(o)} is not JSON serializable")

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

def json_to_str(data: Union[dict, list], indent=3) -> str:
    return json.dumps(data, indent=indent)
    

def load_json(fp: str, default = None):
    if os.path.exists(fp):
        with open(fp, mode='r') as f:
            return json.load(f)
    return default

def data_dir(add: str = None) -> str:
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        raise RuntimeError(f"data directory dosen't exist: {data_dir}")

    
    if add is not None and len(os.path.basename(add)):
        path = os.path.join(data_dir, os.path.dirname(add))
        os.makedirs(path, exist_ok=True)
        path = os.path.join(data_dir, add)
    else:
        path = os.path.join(data_dir, add)
        os.makedirs(path, exist_ok=True)
        
    return path

def load_arff(path: str) -> pd.DataFrame:
    data = arff.loadarff(path)
    train= pd.DataFrame(data[0])
    # print(train.info())

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