import pandas as pd
import scipy.io as scp
import os
import json
import dataclasses
import numpy as np
from typing import Union, Iterable, List, Any, Tuple
import csv
import re
from .space import Integer, Real, Categorical


def json_serialize_unknown(o):
    if isinstance(o, np.integer):
        return int(o)
    elif dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, Integer):
         return dict(
                high=o.high, low=o.low, base=o.base, prior=o.prior, 
                transform=o.transform_
            )
    elif isinstance(o, Real):
        return dict(
            high=o.high, low=o.low, base=o.base, prior=o.prior, 
            transform=o.transform_
        )
    elif isinstance(o, Categorical):
        return dict(categories=o.categories, prior=o.prior, transform=o.transform_)
    else:
        raise TypeError(f"Object of class {type(o)} is not JSON serializable: {o}")

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

def json_to_str(data: Union[dict, list], indent=3) -> str:
    return json.dumps(data, default=json_serialize_unknown, indent=indent)
    

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

# Code from: https://stackoverflow.com/questions/59271661/cannot-load-arff-dataset-with-scipy-arff-loadarff
def load_sparse_arff(path: str) -> pd.DataFrame:
    """
    Converts an ARFF file to a DataFrame.

    Args:
        path (Union[str, Path]): Path to the input ARFF file.

    Returns:
        pd.DataFrame: Converted DataFrame.
    """

    def parse_row(line: str, row_len: int) -> List[Any]:
        """
        Parses a row of data from an ARFF file.

        Args:
            line (str): A row from the ARFF file.
            row_len (int): Length of the row.

        Returns:
            List[Any]: Parsed row as a list of values.
        """
        line = line.strip()  # Strip the newline character
        if '{' in line and '}' in line:
            # Sparse data row
            line = line.replace('{', '').replace('}', '')
            row = np.zeros(row_len, dtype=object)
            for data in line.split(','):
                index, value = data.split()
                try:
                    row[int(index)] = float(value)
                except ValueError:
                    row[int(index)] = np.nan if value == '?' else value.strip("'")
        else:
            # Dense data row
            row = [
                float(value) if value.replace(".", "", 1).isdigit()
                else (np.nan if value == '?' else value.strip("'"))
                for value in line.split(',')
            ]

        return row

    def extract_columns_and_data_start_index(
            file_content: List[str]
    ) -> Tuple[List[str], int]:
        """
        Extracts column names and the index of the @data line from ARFF file content.

        Args:
            file_content (List[str]): List of lines from the ARFF file.

        Returns:
            Tuple[List[str], int]: List of column names and the index of the @data line.
        """
        columns = []
        len_attr = len('@attribute')

        for i, line in enumerate(file_content):
            if line.startswith('@attribute '):
                col_name = line[len_attr:].split()[0]
                columns.append(col_name)
            elif line.startswith('@data'):
                return columns, i

        return columns, 0

    with open(path, 'r') as fp:
        file_content = fp.readlines()

    columns, data_index = extract_columns_and_data_start_index(file_content)
    len_row = len(columns)
    rows = [parse_row(line, len_row) for line in file_content[data_index + 1:]]
    return pd.DataFrame(data=rows, columns=columns)

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