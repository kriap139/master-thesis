from Util import get_train_dataset, get_test_dataset, Dataset, get_dataset_labels
import numpy as np
import pandas as pd
from benchmarks.benchmark import Benchmark
import lightgbm as lgm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from typing import Sequence
from ..Util.dataset import Dataset, get_test_dataset, get_train_dataset
from skopt.space import Real, Integer
import gc


class BaseSearch:
    def __init__(self, dataset: Dataset, fixed_params: dict):
        pass



        

        
class RandomSearch:
    def __init__(self, dataset: Dataset):


if __name__ == "__main__":
    dataset = Dataset.OKCUPID_STEM

    fixed_params = {
        
    }

    search_space = dict(
        n_estimators=Integer(1, 1000, name="n_estimators"),
        learning_rate=Real(0.0001, 1.0, name="learning_rate"),
        max_depth=Integer(0, 30, name="max_depth"),
        num_leaves=Integer(10, 300, name="num_leaves"),
        min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
        feature_fraction=Real(0.1, 1.0, name="feature_fraction")
    )

    x_train, y_train = get_train_dataset(dataset)