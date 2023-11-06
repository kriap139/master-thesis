from Util import Dataset, Builtin
import lightgbm as lgbm
from skopt.space import Real, Integer


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

    dt = lgbm.Dataset(dataset.x, dataset.y, categorical_feature=dataset.cat_features)
    result = lgbm.train({}, train_set=dt, categorical_feature=dataset.cat_features)
    print(result)