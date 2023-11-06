from Util import get_train_dataset, get_test_dataset, Dataset, get_dataset_labels
import numpy as np
import pandas as pd
from benchmarks.benchmark import Benchmark
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from typing import Sequence
from ..Util.dataset import Dataset, Builtin
from skopt.space import Real, Integer
import gc


class LGBMBaseSearch:
    OBJECTIVES = dict(
        binary="binary",
        regression="l2",
        multiclass="softmax",
    )

    METRICS = dict(
        binary="binary_logloss",
        regression="l2",
        multiclass="multi_logloss",
    )

    def __init__(self, train_data: Dataset, fixed_params: dict):
        passba
    
    
class RandomSearch:
    def __init__(self, dataset: Dataset):


if __name__ == "__main__":
    pass

