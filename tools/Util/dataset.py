from enum import Enum
from typing import Tuple, Union, Optional
import pandas as pd
import os
from dataclasses import dataclass, field
import subprocess
from Util.io_util import arff_to_csv, data_dir, load_arff
import lightgbm as lgm
import gc

class Name(Enum):
    HIGGS = 0
    HEPMASS = 1
    KASANDR = 2
    AIRLINES = 3
    FPS = 4
    ACSI = 5
    SGEMM_GKP = 6
    PUF_128 = 7
    WAVE_E = 8 
    OKCUPID_STEM = 9
    ACCEL = 10

TARGET_COLUMNS = {
    Dataset.HIGGS: 0,
    #Dataset.HEPMASS: 0,
    #Dataset.KASANDR: "",
    Dataset.AIRLINES: "DepDelay",
    Dataset.FPS: "FPS",
    Dataset.ACSI: "PINCP",
    Dataset.SGEMM_GKP: "Run1",
    Dataset.PUF_128: 128,
    Dataset.WAVE_E: "energy_total",
    Dataset.OKCUPID_STEM: "job",
    Dataset.ACCEL: "wconfid"
}

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
    def __init__(self, dname: Name, is_test=False):
        self.name = dname.name.lower()
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