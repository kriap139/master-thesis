import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct
import numbers

def plot_lr_iter(data: EvalMetrics):
    pass


if __name__ == "__main__":
    ignore_datasets = (Builtin.AIRLINES.name, )
    result_folders = load_result_folders(ignore_datasets)

    for method, result in result_folders.items():
        datasets = tuple(result.keys())
        print(f"{method}: {datasets}, len={len(datasets)}")

    print()
    metrics = calc_eval_metrics(result_folders)
    print()

    