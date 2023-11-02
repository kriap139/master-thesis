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
        
    
    
class RandomSearch:
    def __init__(self, dataset: Dataset):


if __name__ == "__main__":
    dataset = Dataset(Builtin.OKCUPID_STEM).load()

    fixed_params = dict(
        
    )

    search_space = dict(
        n_estimators=Integer(1, 1000, name="n_estimators"),
        learning_rate=Real(0.0001, 1.0, name="learning_rate"),
        max_depth=Integer(0, 30, name="max_depth"),
        num_leaves=Integer(10, 300, name="num_leaves"),
        min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
        feature_fraction=Real(0.1, 1.0, name="feature_fraction")
    )

    result = lgbm.cv(train_data=dataset, categorical_feature=dataset.cat_features)
    print(result)

