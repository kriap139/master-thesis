from enum import Enum
from typing import Tuple, Union, Optional
import pandas as pd
import os
import subprocess
from Util.io_util import arff_to_csv, data_dir, load_arff
import lightgbm as lgm
import gc

class SizeGroup(Enum):
    SMALL = 0,
    MODERATE = 1,
    LARGE = 2

class DatasetInfo:
    def __init__(self, label_column: any, task: str, size_group: SizeGroup, name = None):
        self.name = name
        self.label_column = label_column
        self.task = task
        self.size_group = size_group


class Builtin(Enum):
    HIGGS = DatasetInfo(0, "binary", SizeGroup.LARGE)
    HEPMASS = DatasetInfo(0, "binary", SizeGroup.LARGE)
    AIRLINES = DatasetInfo("DepDelay", "regression", SizeGroup.LARGE)
    FPS = DatasetInfo("FPS", "regression", SizeGroup.MODERATE)
    ACSI = DatasetInfo("PINCP", "binary", SizeGroup.MODERATE)
    SGEMM_GKP = DatasetInfo("Run1", "regression", SizeGroup.SMALL)
    PUF_128 = DatasetInfo(128, "binary", "large", SizeGroup.LARGE)
    WAVE_E = DatasetInfo("energy_total", "regression", SizeGroup.SMALL)
    OKCUPID_STEM = DatasetInfo("job", "multiclass", SizeGroup.SMALL)
    ACCEL = DatasetInfo("wconfid", "multiclass", SizeGroup.SMALL)


def extract_labels( 
                data: pd.DataFrame, 
                label_column: Union[str, int], 
                remove=True, 
                inplace=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exstracts labels from dataset"""

    if type(label_column) == str:
        indexes = data.columns.get_indexer([label_column])
        assert len(indexes) == 1
        print(f"column index {int(indexes[0])} found for column label '{label_column}'")
        label_column = int(indexes[0])


    if label_column == 0:
        y = data.iloc[:, :1].copy()
    elif label_column == data.shape[1]:
        y = data.iloc[:, data.shape[1]:].copy()
    else:
        last = label_column + 1
        y = data.iloc[:, label_column:last].copy()

    if remove:
        x = data
        data.drop(data.columns[0], axis=1, inplace=inplace)

    return x, y

class Dataset:
    def __init__(self, builtin: Builtin, is_test=False):
        self.info = builtin.value
        self.name = builtin.name.lower()
        self.label_column:  Union[str, int] = TARGET_COLUMNS[dname]
        self.x = None
        self.y = None
        self.cat_features: list = []
        self.is_test = is_test

        path = data_dir(f"datasets/{nm}")
        fns, exts = zip(*[os.path.splitext(f) for f in os.listdir(path)])

        try:
            idx = fns.index(f"{self.name}_train")
            self.test_path = os.path.join(path, f"{fns[idx]}{exts[idx]}")
        except:
            try: 
                idx = fns.index(self.name)
                self.test_path = None
            except:
                raise RuntimeError(f"Dataset files not present({path}): {fns}")
        
        self.train_path = os.path.join(path, f"{fns[idx]}{exts[idx]}")

        if not os.path.exists(self.train_path):
            raise RuntimeError(f"Dataset {self.name} not found in data folder: {self.train_path}")

        if (self.test_path is not None) and (not os.path.exists(self.test_path)):
            raise RuntimeError(f"Test data for dataset '{self.name}' not found in data folder: {self.test_path}")
    
    def __load(self, load_labels_only=False, force_load_test=False) -> pd.DataFrame:
        if (self.is_test or force_load_test) and self.test_path is None:
            raise RuntimeError(f"Test path for {self.name} dataset not found")
        else:
            path = self.test_path if load_test else self.train_path
        
        fn, ext = os.path.splitext(os.path.basename(path))

        if ext.strip() == ".csv":
            return pd.read_csv(path) if not load_labels_only else pd.read_csv(path, usecols=[self.label_column])
        elif ext.strip() == ".arff":
            return load_arff(path) if not load_labels_only else extract_labels(load_arff(path), label_column=self.label_column)[1]
    
    def load(self) -> 'Dataset':
        data = self.__load()
        self.x, self.y = extract_labels(data, label_column=self.label_column)
        self.cat_features = self.x.select_dtypes('object').columns.tolist()
        return self

    def load_labels(self, include_test=True) -> 'Dataset':
        y_train = self.__load(load_labels_only=True)
        # print(y_train.head())

        if self.test_path is not None and include_test:
            y_test = self.__load(load_labels_only=True, force_load_test=True)
            #print(y_test.head())

            y = pd.concat([y_train, y_test])
            assert y.shape[0] == (y_train.shape[0] + y_test.shape[0])
            assert y.shape[1] == 1

            del y_test
            del y_train
            gc.collect()
        else:
            y = y_train
        
        self.y = y
        return self