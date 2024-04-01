import sys
import json
from numbers import Number

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

def format_dict(d: dict) -> str:
    components = []
    for k, v in d.items():
        if isinstance(v, (Number, bool, str)) or (v is None):
            components.append(f"{k}={v}")
        elif isinstance(v, dict):
            components.append(f"{k}={format_dict(v)}")
        elif isinstance(v, list):
            components.append(f"{k}={format_list(v)}")
    return "{" + ",".join(components) + "}"

def format_value(v) -> str:
    if isinstance(v, (Number, bool)) or (v is None):
        return str(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, dict):
        return format_dict(v)
    elif isinstance(v, list):
        return format_list(v)

def main(fp: str, index: int):
    with open(fp, mode='r') as f:
        array: list = json.load(f)
    params: dict = array[index]

    components = []
    for k, v in params.items():
        components.append("=".join((k, format_value(v))))
        
    print(",".join(components))
        
        
if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))