import sys
import json
from numbers import Number
from typing import Union, List, Optional
import os
import re
from .maths import try_number

comma_pattern = r',(?![^{]*})(?![^\[]*\])'
eq_pattern = r'=(?![^{]*})(?![^\[]*\])'

def parse_list(a: str) -> list:
    result = []
    string = a[1:len(a) - 1].strip()
    for param in re.split(comma_pattern, string):
        param = param.strip()
        if param.startswith("{"):
            result.append(parse_dict(param))
        elif param.startswith("["):
            result.append(parse_list(param))
        else:
            result.append(try_number(param))
    return result

def parse_dict(d: str) -> dict:
    result = {}
    string = d.strip()[1:len(d) - 1].strip()
    for param in re.split(comma_pattern, string):
        k, v = re.split(eq_pattern, param)
        v = v.strip()
        if v.startswith("{"):
            result[k.strip()] = parse_dict(v)
        elif v.startswith("["):
            result[k.strip()] = parse_list(v)
        else:
            result[k.strip()] = try_number(v)
    return result

def parse_cmd_params(s: str) -> Union[dict, list]:
    s = s.strip()
    if os.path.exists(s):
        return load_json(s, default={})
    else:
        if s.startswith("["):
            return parse_list(s)
        else:
            if not s.startswith("{"):
                s = '{' + s.strip() + '}'
            return parse_dict(s)