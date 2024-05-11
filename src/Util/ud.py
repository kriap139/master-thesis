import numpy as np
import pandas as pd
import pyunidoe as pydoe
from . import Categorical

def generate_ud_design(search_space: dict, n_runs_per_stage=20, max_search_iter=100) -> pd.DataFrame:
    """
    This function generates the initial uniform design.

    Returns
    ----------
    para_set_ud: A pandas dataframe where each row represents a UD trial point,
            and columns are used to represent variables.
    """
    

    variable_number = [0]
    factor_number = len(search_space)
    para_names = list(search_space.keys())
    #para_ud_names = []

    for k, v in search_space.items():
        if isinstance(v, Categorical):
            variable_number.append(len(v.categories))
            #para_ud_names.extend(k + "_UD_" + str(i + 1) for i in range(len(v.categories)))
        else:
            variable_number.append(1)
            #para_ud_names.append(k + "_UD")
    extend_factor_number = sum(variable_number)

    ud_space = np.repeat(
        np.linspace(1 / (2 * n_runs_per_stage), 1 - 1 / (2 * n_runs_per_stage),
        n_runs_per_stage).reshape([-1, 1]),
        extend_factor_number, 
        axis=1
    )

    base_ud = pydoe.design_query(
        n=n_runs_per_stage, 
        s=extend_factor_number,
        q=n_runs_per_stage, 
        crit="CD2", 
        show_crit=False
    )

    if base_ud is None:
        base_ud = pydoe.gen_ud_ms(
            n=n_runs_per_stage, 
            s=extend_factor_number, 
            q=n_runs_per_stage, 
            crit="CD2",
            maxiter=max_search_iter, 
            random_state=0, 
            n_jobs=10, 
            nshoot=10
        )

    if (not isinstance(base_ud, np.ndarray)):
        raise ValueError('Uniform design is not correctly constructed!')

    para_set_ud = np.zeros((n_runs_per_stage, extend_factor_number))
    for i in range(factor_number):
        loc_min = np.sum(variable_number[:(i + 1)])
        loc_max = np.sum(variable_number[:(i + 2)])
        for k in range(int(loc_min), int(loc_max)):
            para_set_ud[:, k] = ud_space[base_ud[:, k] - 1, k]

    para_set_ud = pd.DataFrame(para_set_ud, columns=para_names)
    return para_set_ud