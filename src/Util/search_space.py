import argparse
from sklearn.model_selection import ParameterGrid
from . import Integer, Real, Categorical
from typing import List, Iterable

def get_search_space(method: str, limit_space: List[str] = None) -> dict:
    if method == "GridSearch":
        space = dict(
            n_estimators=[50, 100, 200, 350, 500],
            learning_rate=[0.001, 0.01, 0.05, 0.1, 0.02],
            max_depth=[0, 5, 10, 20, 25],
            num_leaves=[20, 60, 130, 200, 250],
        )
        print(f"GridSearch runs: {len(ParameterGrid(space))}")
    else:
        space = dict(
            n_estimators=Integer(1, 500, name="n_estimators", prior="log-uniform"),
            learning_rate=Real(0.0001, 1.0, name="learning_rate", prior="log-uniform"),
            max_depth=Integer(0, 300, name="max_depth"),
            num_leaves=Integer(2, 1000, name="num_leaves", prior="log-uniform"),
            min_data_in_leaf=Integer(0, 300, name="min_data_in_leaf"),
            feature_fraction=Real(0.1, 1.0, name="feature_fraction", prior="log-uniform")
        )
    
    if limit_space is not None:
        if isinstance(limit_space, str):
            space = {limit_space: space[limit_space]}
        elif isinstance(limit_space, Iterable):
            space = {param: space[param] for param in limit_space}
    return space

def get_n_search_space(method: str) -> int:
    return len(get_search_space(method).keys())