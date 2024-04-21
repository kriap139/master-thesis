from Util import Dataset, Builtin, Task, data_dir, Integer, Real, Categorical
import logging
from benchmark import BaseSearch, RandomSearch
import os

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    dataset = Dataset(Builtin.OKCUPID_STEM)

    tuner = RandomSearch
    save_dir = data_dir(f"test_results/{tuner.__name__}[{dataset.name}]")

    tuner = tuner(model=None, train_data=dataset, test_data=None, n_iter=100, 
                  n_jobs=None, cv=None, inner_cv=None, scoring=None, save_dir=save_dir)
    
    print(f"Recalculating results for {save_dir}")
    tuner._history_fp = os.path.join(save_dir, os.path.basename(tuner._history_fp))
    tuner._result_fp = os.path.join(save_dir, os.path.basename(tuner._result_fp))
    tuner._calc_result()
