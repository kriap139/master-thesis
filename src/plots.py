import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct, KSearchOptuna
from Util import json_to_space, Task, SizeGroup
from kspace import KSpaceV3
import numbers
import matplotlib.pyplot as plt
import numpy as np

def plot_k_search_method_comp(metrics: EvalMetrics):
    method_names = [method for method in metrics.get_method_names() if method.startswith("KSearch")]
    method_params = {}
    dataset_names = list(metrics.folders.keys())
    param_names = None
    search_space = None

    for dataset, methods in metrics.folders.items():
        for method, folder in methods.items():
            if method.startswith("KSearch"):
                data = KSearchOptuna.recalc_results(folder.dir_path)
                result = data["result"]
                k_params = [param for param in result.keys() if param.startswith("k_")]
                params = [param[len("k_"):] for param in k_params]

                search_space = data["info"]["space"]
                param_names = params

                if method not in method_params:
                    method_params[method] = {dataset: {k_param: result["k_" + k_param] for k_param in params}}
                else:
                    method_params[method][dataset] = {k_param: result["k_" + k_param] for k_param in params}
    
    print(method_names)
    print(dataset_names)
    print(param_names)

    search_space = json_to_space(search_space)
    fig, ax = plt.subplots(nrows=2, ncols=2)

    parm_to_ax = {param_names[0]: (0, 0), param_names[1]: (0, 1), param_names[2]: (1, 0), param_names[3]: (1, 1)}
    dataset_to_ax = {dataset_names[0]: (0, 0), dataset_names[1]: (0, 1), dataset_names[2]: (1, 0), dataset_names[3]: (1, 1)}

    assert len(method_names) == 2
    method_colors = {method_names[0]: "red", method_names[1]: "green"}
    
    param = "n_estimators"
        
    for method, dataset_params in method_params.items():
        for dataset, k in dataset_params.items():
            kspace = KSpaceV3(search_space, k, x_in_search_space=True)
            y_u, y_l = search_space[param].high, search_space[param].low
            x = np.linspace(y_u, y_l, 10_000)
            y = kspace.kmap(param, x)
            x = KSpaceV3._rescale(y_u, y_l, x)
            
            axis = ax[dataset_to_ax[dataset][0], dataset_to_ax[dataset][1]]
            axis.plot(x, y, color=method_colors[method], alpha=0.8, label=method)
            axis.set_title(dataset.lower())
            axis.legend()
            
    plt.show()

if __name__ == "__main__":
    ignore_datasets = ("fps", "acsi", "wave_e", "rcv1", "delays_zurich", "comet_mc", "epsilon", "kdd1998", "kdd1998_allcat", "kdd1998_nonum")
    ignore_methods = ["GridSearch"]
    ignore_with_info_filter = lambda info: info["nparams"] != "4"

    folder_sorter = lambda folder: ( 
        folder.dataset.info().task in (Task.BINARY, Task.MULTICLASS), 
        folder.dataset.info().task == Task.REGRESSION,
        folder.dataset.info().size_group == SizeGroup.SMALL,
        folder.dataset.info().size_group == SizeGroup.MODERATE,
        folder.dataset.info().size_group == SizeGroup.LARGE,
        folder.dataset.name, 
        folder.search_method
    )

    metrics = calc_eval_metrics(0.5, ignore_datasets, ignore_methods, sort_fn=folder_sorter, ignore_with_info_filter=ignore_with_info_filter)
    plot_k_search_method_comp(metrics)
    

    