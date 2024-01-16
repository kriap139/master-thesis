from enum import Enum
from typing import Tuple, Union, Optional
import pandas as pd
import os
import subprocess
from Util.io_util import arff_to_csv, data_dir, load_arff, load_json, save_json, has_csv_header, get_n_csv_columns
import lightgbm as lgb
import gc
import logging
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold, train_test_split
from numbers import Integral
import numpy as np

TY_CV = Union[KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold]

class SizeGroup(Enum):
    SMALL = 0,
    MODERATE = 1,
    LARGE = 2

class Task(Enum):
    BINARY = 1
    MULTICLASS = 2
    REGRESSION = 3

class DatasetInfo:
    def __init__(self, name: str, label_column: Union[str, int], task: Task, size_group: SizeGroup):
        self.name = name
        self.label_column = label_column
        self.task = task
        self.size_group = size_group

class Builtin(Enum):
    HIGGS = DatasetInfo("HIGGS", 0, Task.BINARY, SizeGroup.LARGE)
    HEPMASS = DatasetInfo("HEPMASS", 0, Task.BINARY, SizeGroup.LARGE)
    AIRLINES = DatasetInfo("AIRLINES", "DepDelay", Task.REGRESSION, SizeGroup.LARGE)
    FPS = DatasetInfo("FPS", "FPS", Task.REGRESSION, SizeGroup.MODERATE)
    ACSI = DatasetInfo("ACSI", "PINCP", Task.BINARY, SizeGroup.MODERATE)
    SGEMM_GKP = DatasetInfo("SGEMM_GKP", "Run1", Task.REGRESSION, SizeGroup.SMALL)
    PUF_128 = DatasetInfo("PUF_128", 128, Task.BINARY, SizeGroup.LARGE)
    WAVE_E = DatasetInfo("WAVE_E", "energy_total", Task.REGRESSION, SizeGroup.SMALL)
    OKCUPID_STEM = DatasetInfo("OKCUPID_STEM", "job", Task.MULTICLASS, SizeGroup.SMALL)
    ACCEL = DatasetInfo("ACCEL", "wconfid", Task.MULTICLASS, SizeGroup.SMALL)
    RCV1 = DatasetInfo("RCV1", "class", Task.BINARY, SizeGroup.MODERATE)
    DELAYS_ZURICH = DatasetInfo("DELAYS_ZURICH", "delay", Task.REGRESSION, SizeGroup.LARGE)
    COMET_MC = DatasetInfo("COMET_MC", "label", Task.MULTICLASS, SizeGroup.LARGE)

    def info(self) -> DatasetInfo:
        return self.value


def extract_labels( 
                data: pd.DataFrame, 
                label_column: Union[str, int], 
                remove=True, 
                inplace=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exstracts labels from dataset"""

    if type(label_column) == str:
        y = data[label_column].copy()
        col = label_column
    else:
        if label_column == 0:
            y = data.iloc[:, :1].copy()
        elif label_column == data.shape[1]:
            y = data.iloc[:, data.shape[1]:].copy()
        else:
            last = label_column + 1
            y = data.iloc[:, label_column:last].copy()
        
        col = data.columns[label_column]

    if remove:
        x = data
        data.drop(col, axis=1, inplace=inplace)

    return x, y

class Dataset(DatasetInfo):
    FOLDS_FILE_PREFIX = "_folds"

    def __init__(self, bn: Builtin, is_test=False):
        super().__init__(bn.info().label_column, bn.info().task, bn.info().size_group, bn.name.lower())
        self.x = None
        self.y = None
        self.cat_features: list = []
        self.is_test = is_test

        self.test_path = None
        self.train_path = None
        self.saved_folds_path = None
        self.__saved_folds_fn = f"{self.name}{self.FOLDS_FILE_PREFIX}.json"
        self.__set_dataset_paths()
    
    def get_builtin(self):
        return Builtin[self.name]

    def get_dir(self) -> str:
        path = data_dir(f"datasets/{self.name}")
        if not os.path.exists(path):
            raise RuntimeError(f"Dataset path dosen't exist: {path}")
        return path
    
    def __set_dataset_paths(self):
        path = self.get_dir()
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
        
        fn, ext = os.path.splitext(self.__saved_folds_fn)
        if (fn in fns) and (exts[fns.index(fn)] == ".json"):
            self.saved_folds_path = os.path.join(path, self.__saved_folds_fn)

    def has_saved_folds(self) -> bool:
        return self.saved_folds_path is not None
    
    def has_test_set(self) -> bool:
        return self.test_path is not None
    
    def load_saved_folds_file(self) -> dict:
        if self.saved_folds_path is None:
            raise RuntimeError(f"No saved folds data found for dataset {self.name}")

        data = load_json(self.saved_folds_path)
        folds = data["folds"]
        data["folds"] = [
            (
                np.array(fold["train"]).reshape(fold["shape_train"]), 
                np.array(fold["test"]).reshape(fold["shape_test"])
            ) for fold in folds]

        return data
    
    def load_saved_folds_info(self):
        return self.load_saved_folds_file()["info"]
    
    def load_saved_folds(self) -> dict:
        return self.load_saved_folds_file()["folds"]
        
    def save_folds(self, cv: TY_CV):
        info = self.get_cv_info(cv)
        folds = []
        for train_idx, test_idx in cv.split(self.x, self.y):
            folds.append(
                dict(
                    train=train_idx.tolist(),
                    test=test_idx.tolist(),
                    shape_train=train_idx.shape,
                    shape_test=test_idx.shape
                )
            )

        path = os.path.join(self.get_dir(), self.__saved_folds_fn)
        save_json(path, data=dict(info=info, folds=folds), indent=None)
        self.__set_dataset_paths()
    
    def __load(self, load_labels_only=False, force_load_test=False) -> pd.DataFrame:
        if (self.is_test or force_load_test) and (self.test_path is None):
            raise RuntimeError(f"Test path for {self.name} dataset not found")
        else:
            path = self.test_path if self.is_test else self.train_path
        
        fn, ext = os.path.splitext(os.path.basename(path))
        print(f"Loading dataset from path: {path}")

        if ext.strip() == ".csv":
            if has_csv_header(self.train_path):
                return pd.read_csv(path) if not load_labels_only else pd.read_csv(path, usecols=[self.label_column])
            else:
                n_cols = get_n_csv_columns(self.train_path)
                head = tuple(range(n_cols))
                data = pd.read_csv(path, names=head) if not load_labels_only else pd.read_csv(path, usecols=[self.label_column], names=head)
                return data
        elif ext.strip() == ".arff":
            return load_arff(path) if not load_labels_only else extract_labels(load_arff(path), label_column=self.label_column)[1]
    
    def load_test_dataset(self) -> 'Dataset':
        return Dataset(self.get_builtin(), is_test=True).load()
    
    def load(self) -> 'Dataset':
        data = self.__load()
        shape = data.shape

        self.x, self.y = extract_labels(data, label_column=self.label_column)
        self.cat_features = self.x.select_dtypes('object').columns.tolist()
        
        self.y = self.y.to_numpy()
        self.y.shape = (-1,)

        # make acsi dataset into a binary classification problem!
        if self.name == "acsi":
            self.y[self.y <= 50_000] = 0
            self.y[self.y > 50_000] = 1
            values, counts = np.unique(self.y, return_counts=True)
            print(f"acsi value counts: values={values}, counts={counts}")

        print(f"data={shape}, x={self.x.shape}, y={self.y.shape}")
        assert self.y.shape == (shape[0],)
        assert self.x.shape == (shape[0], shape[1] - 1)

        if len(self.cat_features):
            for col in self.cat_features:
                self.x[col] = self.x[col].astype('category')

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
    
    def create_train_test_splits(self, force=False, tests_size=0.3, shuffle=True):
        if (self.test_path is not None) and not force:
            print("train test split already exists!")
            return
        
        if self.x is None:
            self.load()
        
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=tests_size, shuffle=shuffle)
        
        print(f"trainX={x_train.shape}, trainY{y_train.shape}, testX={x_test.shape}, testY={y_test.shape}")

        if type(self.label_column) == str:
            x_train[self.label_column] = y_train
            x_test[self.label_column] = y_test
        else:
            x_train.insert(self.label_column, self.label_column, y_train)
            x_test.insert(self.label_column, self.label_column, y_test)

        path = data_dir(f"datasets/{self.name}")
        fn, ext = os.path.splitext(os.path.basename(path))
        train_path = os.path.join(path, f"{self.name}_train{ext}")
        test_path = os.path.join(path, f"{self.name}_test{ext}")

        x_train.to_csv(train_path, index=False, index_label=False, header=False)
        x_test.to_csv(test_path, index=False, index_label=False, header=False)

        self.__set_dataset_paths()
        self.load()
        return self
    
    @classmethod
    def get_cv_info(cls, cv: TY_CV) -> dict:
        name = cv.__class__.__name__
        info = dict(
            name=name, 
            random_state=cv.random_state,
        )

        if hasattr(cv, "n_repeats"):
            info["n_repeats"] = cv.n_repeats
        if "Stratified" in name:
            info["stratified"] = True

        for attr in ("n_splits", "shuffle"):
            if hasattr(cv, attr):
                info[attr] = getattr(cv, attr)
            elif hasattr(cv, "cv"):
                if hasattr(cv.cv, attr):
                    info[attr] = getattr(cv.cv, attr)
                elif hasattr(cv, "cvargs"):
                    if attr in cv.cvargs.keys():
                        info[attr] = cv.cvargs[attr]
        
        if "n_splits" in info.keys():
            n_folds = info.pop("n_splits")
            info["n_folds"] = n_folds
        
        return info
    
    @classmethod
    def merge_train_test(cls, d: Builtin):
        dataset = Dataset(d)
        train = dataset.__load()
        test = dataset.__load(force_load_test=True)

        joined = pd.concat([train, test], ignore_index=True)
        path = os.path.join(os.path.dirname(dataset.train_path), f"{d.name.lower()}.csv")
        joined.to_csv(path, index=False)
