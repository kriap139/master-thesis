from typing import Union, Iterable

def removeprefix(s: str, prefixes: Union[str, Iterable]) -> str:
    if isinstance(prefixes, str):
        prefixes = (prefixes, )
    s = s.strip()

    for prefix in prefixes:
        end = len(prefix)
        if (len(prefix) > len(s)) or (len(prefix) == 0):
            continue
        sub_str = s[:end]
        if sub_str == prefix:
            return s[end:]
    return s
