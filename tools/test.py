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
from scipy.sparse import coo_matrix
from sklearn.metrics import get_scorer, get_scorer_names

from benchmark import BaseSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold, SeqUDSearch, OptunaSearch, AdjustedSeqUDSearch
from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical, has_csv_header, CVInfo
import lightgbm as lgb
from search import get_sklearn_model, get_cv, build_cli, search
import logging
import csv



def search_test(method: str, dataset: Builtin, max_lgb_jobs=None, n_jobs=None):
    if (max_lgb_jobs is not None) and n_jobs is not None:
        args = build_cli(method, dataset, max_lgb_jobs, n_jobs)
    elif max_lgb_jobs is not None:
        args = build_cli(method, dataset, test_max_lgb_jobs=max_lgb_jobs)
    elif n_jobs is not None:
        args = build_cli(method, dataset, n_jobs=n_jobs)
    else:
        args = build_cli(method, test_dataset)
    search(args)

if "__main__" == __name__:
    search_test(AdjustedSeqUDSearch.__name__, Builtin.ACCEL, max_lgb_jobs=6, n_jobs=1)