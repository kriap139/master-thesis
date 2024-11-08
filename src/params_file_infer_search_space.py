import sys
import json
from numbers import Number
from typing import Union, List

def get_search_space(params: dict) -> str:
    if 'k' in params:
        keys = params["k"].keys()
        return ",".join(keys)
    elif "search_space" in params:
        search_space =  params["search_space"]
        return ",".join(search_space)
    else:
        return "all"

def main(fp: str, indexes: Union[int, List[int]] = None) -> Union[str, dict, list]:
    with open(fp, mode='r') as f:
        array: list = json.load(f)
    if isinstance(indexes, int):
        indexes = (indexes, )
    
    results = []
    for index in indexes:
        params: dict = array[index]
        results.append(get_search_space(params))
    
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        return " ".join(results)
        
        
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