import numpy as np
import pandas as pd
from typing import Iterable, Callable, Tuple, Dict, Union, List
from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, has_csv_header, CVInfo, save_json, TY_CV, load_json, load_csv, get_search_space, json_to_str
from kspace import KSpaceV3
from benchmark import (
    BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold, SeqUDSearch, OptunaSearch, AdjustedSeqUDSearch, RandomSearch,
    KSpaceSeqUDSearch, KSpaceOptunaSearch
)
import matplotlib.pyplot as plt
import Util
from sklearn.model_selection import ParameterSampler
from dataclasses import dataclass
import json
from numbers import Number

def kspace_discrepancy(
    search: Union[BaseSearch, str, dict], 
    param: str, 
    outer_iters: Union[int, List[int]] = 0, 
    show=True, 
    save=False, 
    k_graph_label: str = None,
    iters_labels: List[str] = None,
    plot_k_graph=True,
    k_graph_alpha=0.8,
    alpha=0.5,
    colors: Union[list, str] = None,
    k_graph_color=None,
    fig=None):

    if isinstance(search, BaseSearch):
        assert search._inner_history_fp is not None and os.path.exists(search._inner_history_fp)
        inner_history = load_csv(search._inner_history_fp)
        info = load_json(search._result_fp)["info"]
        space = info["space"]
    elif isinstance(search, dict):
        inner_history = search["inner_history"]
        info = search["info"]
        space = info["space"]
    else:
        assert os.path.exists(search)
        print(f"results dir: {search}")
        inner_history = load_csv(os.path.join(search, "inner_history.csv"))
        info = load_json(os.path.join(search, "result.json"))["info"]
        space = info["space"]
    
    if info["method_params"].get("x_in_search_space", None) is not None:
        x_in_search_space = info["method_params"]["x_in_search_space"]
    else:
        x_in_search_space = False
    
    k = info["method_params"]["k"]
    if not isinstance(k, dict):
        try:
            k = int(k)
        except:
            k = float(k)
    
    if fig is None:
        fig = plt

    k_space = {k: getattr(Util, d.pop('cls'))(**d) for k, d in space.items()}
    kspace = KSpaceV3(k_space, k=k, x_in_search_space=x_in_search_space)
    y_u, y_l = k_space[param].high, k_space[param].low

    para_names = space.keys()
    if isinstance(outer_iters, int):
        outer_iters = (outer_iters, )

    if k_graph_label is None:
        k_graph_label = f"k={kspace._kmap[param]} plot"
    if iters_labels is None:
        iters_labels = [f"outer_iter {n_iter}" for n_iter in outer_iters]
    if colors is None:
        colors = ['red' for n_iter in outer_iters]
    elif isinstance(colors, str):
        colors = [colors for n_iter in outer_iters]

    if plot_k_graph:
        if k_graph_color is None:
            k_graph_color = 'green'
        if x_in_search_space:
            x = np.linspace(y_u, y_l, 60_000)
            y = kspace.kmap(param, x)
            x = KSpaceV3._rescale(y_u, y_l, x)
        else:
            x = np.linspace(0, 1, 60_000)
            y = kspace.kmap(param, x)

        #print(list(y))
        plt.plot(x, y, color=k_graph_color, alpha=k_graph_alpha, label=k_graph_label)

    for n_iter, outer_iter in enumerate(outer_iters):
        _iter = inner_history[inner_history["outer_iter"] == outer_iter]
        cols = list(_iter.columns)

        for i in range(len(cols)):
            checks = tuple(cols[i].startswith(prefix) for prefix in ("params_", "user_attrs_"))
            if any(checks):
                splits = 2 if checks[1] else 1
                cols[i] = cols[i].split("_", maxsplit=splits)[splits]

        rename = {old: _new for old, _new in zip(_iter.columns, cols)}
        _iter = _iter.rename(columns=rename)
        params = _iter[para_names]
        #print(_iter.head())

        param_vals = params[param]
        params_x = _iter[param + "_kx"].to_numpy()
        params_y = param_vals.to_numpy()
        #print(f"npx={params_x.shape}, npy={params_y.shape}")

        # soriting by x
        indexes = params_x.argsort()
        params_x = params_x[indexes]
        params_y = params_y[indexes]

        if x_in_search_space:
            params_x = KSpaceV3._rescale(y_u, y_l, params_x)

        plt.scatter(params_x, params_y, color=colors[n_iter], label=iters_labels[n_iter], alpha=alpha)
        
    plt.legend()
    if save is not None:
        name = f"kspace_plot_{param}.png"
        plt.savefig(name)
    if show:
        plt.show()

def plot_kspace_wrapper(
    search_space: dict, 
    params: List[Dict[str, Number]], 
    param: str, 
    k: Union[int, dict], 
    x_in_search_space=False, 
    show=True, 
    save=True,
    k_graph_label: str = None,
    iters_labels: List[str] = None,
    plot_k_graph=True,
    k_graph_alpha=0.8,
    alpha=0.5,
    colors: Union[list, str] = None,
    k_graph_color=None,
    fig=None):

    kspace = KSpaceV3(search_space, k, x_in_search_space=x_in_search_space)
    
    frame_data = []
    for _param in params:
        d = {f"{key}_kx": v for (key, v) in _param.items()}
        for (key, v) in _param.items():
            d[key] = kspace.kmap(key, v)
        frame_data.append(d)
        
    inner_history = pd.DataFrame(frame_data)
    inner_history["outer_iter"] = 0
    #params.append(dict(learning_rate=search_space["learning_rate"].low, n_estimators=search_space["n_estimators"].low))
    #params.append(dict(learning_rate=search_space["learning_rate"].high, n_estimators=search_space["n_estimators"].high))
    print(inner_history.head())

    data = dict(
        inner_history=inner_history,
        info=dict(
            space=json.loads(json_to_str(search_space)),
            method_params=dict(
                x_in_search_space=x_in_search_space,
                k=k
            )
        )
    )
    kspace_discrepancy(
        data, 
        param, 
        show=show, 
        save=save, 
        k_graph_label=k_graph_label,
        iters_labels=iters_labels,
        plot_k_graph=plot_k_graph, 
        alpha=alpha, 
        k_graph_alpha=k_graph_alpha,
        colors=colors,
        k_graph_color=k_graph_color,
        fig=fig
    )

def plot_kspace_random(
    param: str, 
    k: int = 3, 
    n_iter: int = 100, 
    show=True, 
    save=True, 
    limit_space: List[str] = None, 
    k_graph_label: str = None,
    iters_labels: List[str] = None,
    alpha=0.5, 
    k_graph_alpha=0.8, 
    plot_k_graph=True,
    colors: Union[list, str] = None,
    k_graph_color=None,
    fig=None):

    search_space = get_search_space("RandomSearch", limit_space=limit_space)
    sampler = ParameterSampler(search_space, n_iter=n_iter)
    params = list(sampler)
    plot_kspace_wrapper(
        search_space=search_space, 
        params=params, 
        param=param, 
        k=k, 
        x_in_search_space=True, 
        save=save, 
        show=show, 
        plot_k_graph=plot_k_graph, 
        k_graph_label=k_graph_label,
        iters_labels=iters_labels,
        k_graph_alpha=k_graph_alpha, 
        alpha=alpha,
        colors=colors,
        k_graph_color=k_graph_color,
        fig=fig
    )

def plot_kspace_ud(
    param: str, 
    k: int = 3, 
    n_iter: int = 100, 
    show=True, 
    save=True, 
    limit_space: List[str] = None, 
    plot_k_graph=True, 
    k_graph_label=None,
    iters_labels: List[str] = None,
    alpha=0.5,
    k_graph_alpha=0.8,
    colors: Union[list, str] = None,
    k_graph_color=None,
    fig=None):
    from Util.ud import generate_ud_design

    search_space = get_search_space("", limit_space=limit_space)
    frame = generate_ud_design(search_space, n_runs_per_stage=n_iter)
    params = frame.to_dict(orient='records')
    plot_kspace_wrapper(
        search_space=search_space, 
        params=params, 
        param=param, 
        k=k, 
        x_in_search_space=False, 
        save=save, 
        show=show, 
        plot_k_graph=plot_k_graph,
        k_graph_label=k_graph_label,
        iters_labels=iters_labels,
        k_graph_alpha=k_graph_alpha,
        alpha=alpha,
        colors=colors,
        k_graph_color=k_graph_color,
        fig=fig
    )

def plot_kspace_ud_random( 
    param: str, 
    k: int = 3, 
    n_iter: int = 100, 
    show=True, 
    save=True, 
    limit_space: List[str] = None, 
    plot_k_graph=True, 
    alpha=0.5,
    k_graph_alpha=0.8,
    colors: Union[list, str] = None,
    k_graph_color=None):

    if colors is None or (len(colors) < 2):
        colors = ['red', 'blue']    
    
    plt.subplot(2, 1, 1)
    plt.title(f"kspace random {param}")
    plot_kspace_random(
        param=param, 
        k=k, 
        n_iter=n_iter, 
        show=False, 
        save=False, 
        limit_space=limit_space, 
        k_graph_label=None,
        iters_labels=["Random"], 
        alpha=alpha, 
        k_graph_alpha=k_graph_alpha, 
        plot_k_graph=plot_k_graph, 
        colors=[colors[0]], 
        k_graph_color=k_graph_color
    )

    plt.subplot(2, 1, 2)
    plt.title(f"kspace UD {param}")
    plot_kspace_ud(
        param=param, 
        k=k, 
        n_iter=n_iter, 
        show=show, 
        save=save, 
        limit_space=limit_space, 
        k_graph_label=None,
        iters_labels=["UD"], 
        alpha=alpha, 
        k_graph_alpha=k_graph_alpha, 
        plot_k_graph=plot_k_graph, 
        colors=[colors[1]], 
        k_graph_color=k_graph_color
    )

if "__main__" == __name__:
    path = ""
    param = "n_estimators"  # "n_estimators" "learning_rate"
    limit_space=['learning_rate', 'n_estimators']
    k = dict(
        learning_rate=-3,
        n_estimators=3
    )

    #plot_kspace_random(param, k, n_iter=100, show=True, save=False, limit_space=limit_space, plot_k_graph=True, alpha=0.3, k_graph_alpha=0.9)
    #plot_kspace_ud(param, k, n_iter=100, show=True, save=False, limit_space=limit_space, plot_k_graph=True, alpha=0.3, k_graph_alpha=0.9)
    plot_kspace_ud_random(param, k, n_iter=100, show=True, save=False, limit_space=limit_space, plot_k_graph=True, alpha=0.3, k_graph_alpha=0.9)