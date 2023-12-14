import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score
from sequd import SeqUD

from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold, SeqUDSearch, OptunaSearch
from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, has_csv_header
import lightgbm as lgb
from search import get_sklearn_model
import logging

import csv

def search_test():
    logging.getLogger().setLevel(logging.DEBUG)
    search_space = dict(
        n_estimators=Integer(1, 500, name="n_estimators", prior="log-uniform"),
        learning_rate=Real(0.0001, 1.0, name="learning_rate", prior="log-uniform"),
        max_depth=Integer(0, 30, name="max_depth"),
        num_leaves=Integer(10, 300, name="num_leaves", prior="log-uniform"),
        min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
        feature_fraction=Real(0.1, 1.0, name="feature_fraction", prior="log-uniform")
    )

    dataset = Dataset(Builtin.OKCUPID_STEM).load()
    print(f"column names: {list(dataset.x.columns)}")
    print(f"cat_features: {dataset.cat_features}")


    fixed_params = dict(
        categorical_feature=dataset.cat_features,
    )

    tuner = OptunaSearch
    save_dir = data_dir(f"test_results/{tuner.__name__}[{dataset.name}]")
    model = get_sklearn_model(dataset, verbose=-1, n_jobs=None)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    tuner = tuner(model=model, train_data=dataset, test_data=None, n_iter=100, 
                    n_jobs=None, cv=cv, inner_cv=None, scoring=None, save_dir=save_dir)

    print(f"Results saved to: {tuner._save_dir}")
    tuner.search(search_space, fixed_params)

dataset = Dataset(Builtin.RCV1).load()
print(f"x={dataset.x.shape}, y={dataset.y.shape}")
#print(f"{dataset.name.upper()}: {has_csv_header(dataset.train_path)}")