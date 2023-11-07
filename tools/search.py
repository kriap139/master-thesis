from Util import Dataset, Builtin, Task, data_dir
import lightgbm as lgb
from skopt.space import Real, Integer
import logging
from benchmark import BaseSearch, RandomSearch, RepeatedStratifiedKFold, RepeatedKFold, KFold, StratifiedKFold
from typing import Union

OBJECTIVES = {
        Task.BINARY: "binary",
        Task.MULTICLASS: "softmax",
        Task.REGRESSION: "l2"
}

METRICS = {
    Task.BINARY: "binary_logloss",
    Task.MULTICLASS: "multi_logloss",
    Task.REGRESSION: "l2",
}

SCORING = {
    Task.BINARY: "acc"
}

def get_sklearn_model() -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        if self.train_data.task in (Task.MULTICLASS, Task.BINARY):
            return lgbm.LGBMClassifier
        elif self.train_data.task == Task.REGRESSION:
            return lgbm.LGBMRegressor

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    search_space = dict(
        n_estimators=Integer(1, 1000, name="n_estimators"),
        learning_rate=Real(0.0001, 1.0, name="learning_rate"),
        max_depth=Integer(0, 30, name="max_depth"),
        num_leaves=Integer(10, 300, name="num_leaves"),
        min_data_in_leaf=Integer(0, 30, name="min_data_in_leaf"),
        feature_fraction=Real(0.1, 1.0, name="feature_fraction")
    )

    dataset = Dataset(Builtin.OKCUPID_STEM).load()
    print(f"column names: {list(dataset.x.columns)}")
    print(f"cat_features: {dataset.cat_features}")
    #print(dataset.x.info())

    fixed_params = dict(
        #objective=OBJECTIVES[dataset.get_builtin()],
        #metric=METRICS[dataset.get_builtin()]
    )

    tuner = RandomSearch
    save_file = f"[{dataset.name}].json"
    save_path = data_dir(f"test_results/{save_file}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    tuner = tuner(model=None, train_data=dataset, test_data=None, n_iter=100, 
                  n_jobs=None, cv=cv, inner_cv=None, scoring=None, save_path=save_path)

    tuner.search(search_space, fixed_params)