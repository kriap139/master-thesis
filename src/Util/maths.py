
from numbers import Number
from typing import Dict, Union, Optional
import pandas as pd
import numpy as np
from . import Categorical, Integer, Real, TY_SPACE

TY_MAP_X = Union[pd.Series, np.ndarray, Number]
TY_MAP_LIM = Union[pd.Series, np.ndarray, Number]

def _map(x: TY_MAP_X, x_min: TY_MAP_LIM = 0, x_max: TY_MAP_LIM = 1, y_min: TY_MAP_LIM = 0, y_max: TY_MAP_LIM = 1) -> TY_MAP_X:
    slope = (y_max - y_min) / (x_max - x_min)
    return slope * (x - x_min) + y_min

def map_space(search_space: TY_SPACE, norm_frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(norm_frame, pd.DataFrame):
        result = pd.DataFrame(np.zeros(norm_frame.shape), columns=norm_frame.columns)
        for param in norm_frame.columns:
            y_u, y_l = search_space[param].low, search_space[param].high
            result[param] = _map(norm_frame[param], y_min=y_l, y_max=y_u)

            if isinstance(search_space[param], Integer):
                result[param].round().astype(int)

        return result
    elif isinstance(norm_frame, pd.Series):
        y_u, y_l = search_space[norm_frame.name].low, search_space[norm_frame.name].high
        if isinstance(search_space[norm_frame.name], Integer):
            return _map(norm_frame, y_min=y_l, y_max=y_u).round().astype(int)
        else: 
            return _map(norm_frame, y_min=y_l, y_max=y_u)

def try_number(param: str) -> Optional[Union[Number, str]]:
    if param == "None":
        return None
    try:
        return int(param)
    except ValueError:
        try:
            return float(param)
        except ValueError:
            return param
        

        