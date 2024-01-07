import os
from Util import Dataset, Builtin
from typing import List, Dict
import re
from dataclasses import dataclass

@dataclass
class ResultFolder:
    dir_path: str
    dataset: Builtin
    search_method: str
    version: int = 0

def load_result_files() -> Dict[str, Dict[str, ResultFolder]]:
    result_dir = data_dir(add="test_results")

    results: Dict[str, Dict[str, ResultFolder]] = {}

    for test in os.listdir(result_dir):
        path = os.path.join(result_dir, test)

        array = test.split("[")
        method, remainder = array[0].strip(), array[1].strip()
        array = remainder.split("]")
        dataset, remainder = array[0].strip().upper(), array[1].strip()
        version = re.findall(r'(\d+)', remainder)
        version = int(version[0]) if len(version) else 0

        method_results = results.get(method, None)

        if method_results is None: 
            result[method] = {dataset: ResultFolder(path, Builtin[dataset], method, version)}
        else:
            result = method_results.get(dataset, None)
            if result is None:
                method_results[dataset] = ResultFolder(path, Builtin[dataset], method, version)
            elif result.version < version:
                result.dir_path = path
                result.version = version
    
    return results


if __name__ == "__main__":
    pass

            
            

