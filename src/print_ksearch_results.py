from calc_metrics import ResultFolder, EvalMetrics, load_result_folders, calc_eval_metrics
import logging
import pandas as pd
import numpy as np
from scipy.stats import mode
from typing import Dict, Union, List
from Util import load_csv

def print_k_search(folders: Dict[str, Dict[str, Union[List[ResultFolder], ResultFolder]]]):
    for (dataset, methods) in data.items():
        strings = []
        for method, folder in methods.items():
            history_path = os.path.join(folder.dir_path, "history.csv")
            data = load_csv(folder.dir_path) 

            train_scores = data["train_score"]
            test_scores = data["test_score"]
            mean_test_acc=np.mean(test_scores)
            std_test_acc=np.std(test_scores)
            max_test_acc=np.max(test_scores)

            strings.append(f"{folder.dir_str}: mean={mean_test_acc}, std={std_test_acc}, max={max_test_acc}")

        print(f"{dataset}: \n" + "\n".join(strings) + '\n')
        strings.clear()

if __name__ == "__main__":
    ignore_datasets = ("kdd1998_allcat", "kdd1998_nonum")
    ignore_methods = (
        "RandomSearch", "SeqUDSearch", "GridSearch", #"OptunaSearch", 
        "KSpaceSeqUDSearch", "KSpaceSeqUDSearchV2", "KSpaceSeqUDSearchV3", 
        "KSpaceOptunaSearch", "KSpaceOptunaSearchV2", "KSpaceOptunaSearchV3",
        "NOSearch", "KSpaceRandomSearchV3"
    )

    folder_sorter = lambda folder: ( 
        folder.dataset.info().task in (Task.BINARY, Task.MULTICLASS), 
        folder.dataset.info().task == Task.REGRESSION,
        folder.dataset.info().size_group == SizeGroup.SMALL,
        folder.dataset.info().size_group == SizeGroup.MODERATE,
        folder.dataset.info().size_group == SizeGroup.LARGE,
        folder.dataset.name, 
        folder.search_method
    )

    folders = load_result_folders(ignore_datasets, ignore_methods, sort_fn=folder_sorter)
    print_k_search(folders)