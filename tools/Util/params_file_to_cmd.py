import sys
import json
from numbers import Number
from typing import Union, List

def format_list(a: dict) -> str:
    components = []
    for v in a:
        if isinstance(v, (Number, bool)) or (v is None):
            components.append(str(v))
        elif isinstance(v, str):
            components.append(v)
        elif isinstance(v, dict):
            components.append(format_dict(v))
        elif isinstance(v, list):
            components.append(format_list(v))
    return "[" + ",".join(components) + "]"

def format_dict(d: dict, add_brackets=True) -> str:
    components = []
    for k, v in d.items():
        if isinstance(v, (Number, bool, str)) or (v is None):
            components.append(f"{k}={v}")
        elif isinstance(v, dict):
            components.append(f"{k}={format_dict(v)}")
        elif isinstance(v, list):
            components.append(f"{k}={format_list(v)}")
    
    if add_brackets:
        return "{" + ",".join(components) + "}"
    return ",".join(components)

def format_value(v) -> str:
    if isinstance(v, (Number, bool)) or (v is None):
        return str(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, dict):
        return format_dict(v)
    elif isinstance(v, list):
        return format_list(v)

def main(fp: str, indexes: Union[int, List[int]] = None) -> Union[str, dict, list]:
    with open(fp, mode='r') as f:
        array: list = json.load(f)
    if indexes is None:
        return fp
    elif isinstance(indexes, int):
        indexes = (indexes, )
    
    results = []
    add_bracket = len(indexes) > 1
    for index in indexes:
        params: dict = array[index]
        results.append(format_dict(params, add_brackets=add_bracket))
    
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        return format_list(results)
        
        
if __name__ == "__main__":
    n_idx = len(sys.argv) - 2

    if n_idx < 0:
        raise RuntimeError(f"Need arguments FILE INDEXES")
    elif n_idx == 0:
        idxes = None
    elif n_idx == 1:
        idxes = int(sys.argv[2])
    else:
        args = sys.argv[2:]
        idxes = [int(s) for s in args]

    print(main(sys.argv[1], idxes))