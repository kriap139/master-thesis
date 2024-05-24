import argparse
from sklearn.model_selection import ParameterGrid
from .scikit_space import Integer, Real, Categorical, TY_DIM, TY_SPACE
from typing import List, Iterable, Dict
from . import scikit_space

def get_search_space(method: str, limit_space: List[str] = None, add_priors: Dict[str, str]  = None, add_unset_default_priors=True) -> dict:
    default_priors = dict(learning_rate="log-uniform")
    if add_priors is None:
        prior_map = default_priors
    else:
        prior_map = add_priors.copy()
        if add_unset_default_priors:
            for k, v in default_priors.items():
                if prior_map.get(k, None) is None:
                    prior_map[k] = v

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
            n_estimators=Integer(1, 500, name="n_estimators", prior=prior_map.get("n_estimators", "uniform")),
            learning_rate=Real(0.0001, 1.0, name="learning_rate", prior=prior_map.get("learning_rate", "uniform")),
            max_depth=Integer(0, 30, name="max_depth", prior=prior_map.get("max_depth", "uniform")),
            num_leaves=Integer(10, 300, name="num_leaves", prior=prior_map.get("num_leaves", "uniform")),
            min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf", prior=prior_map.get("min_data_in_leaf", "uniform")),
            feature_fraction=Real(0.1, 1.0, name="feature_fraction", prior=prior_map.get("feature_fraction", "uniform"))
        )
    
    if limit_space is not None:
        if isinstance(limit_space, str):
            space = {limit_space: space[limit_space]}
        elif isinstance(limit_space, Iterable):
            space = {param: space[param] for param in limit_space}
    return space

def get_n_search_space(method: str) -> int:
    return len(get_search_space(method).keys())

def json_to_space(data: dict) -> TY_SPACE:
    return {k: getattr(scikit_space, d.pop('cls'))(**d) for k, d in data.items()}