from enum import Enum, Flag, auto, EnumMeta
from typing import Tuple, Union, Optional
import sklearn.datasets as sk_datasets
import pandas as pd
import os
import subprocess
from Util.io_util import arff_to_csv, data_dir, load_arff, load_json, save_json, has_csv_header, get_n_csv_columns, load_libsvm
from Util.sparse_arff import save_sparse_arff
import lightgbm as lgb
import gc
import logging
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold, train_test_split
from numbers import Integral
import numpy as np
from Util.compat import removeprefix

TY_CV = Union[KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold]

SK_DATASETS = [
    "sk." + removeprefix(func, ("load_", "fetch_")) 
    for func in sk_datasets.__all__
    if (func.startswith("fetch") or func.startswith("load")) and 
        all(not func.__contains__(s) for s in ('file', 'image', 'openml', 'olivetti', 'lfw', '20newsgroups'))
]

class SizeGroup(Enum):
    SMALL = 0,
    MODERATE = 1,
    LARGE = 2

class Task(Flag):
    BINARY = auto()
    MULTICLASS = auto()
    REGRESSION = auto()

class DatasetInfo:
    def __init__(self, name: str, label_column: Union[str, int], task: Task, size_group: SizeGroup, mod=None):
        self.name = name
        self.label_column = label_column
        self.task = task
        self.size_group = size_group
        self.mod = mod

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True  

class Builtin(Enum, metaclass=MetaEnum):
    HIGGS = DatasetInfo("HIGGS".lower(), 0, Task.BINARY, SizeGroup.LARGE)                                  #
    FPS = DatasetInfo("FPS".lower(), "FPS", Task.REGRESSION, SizeGroup.MODERATE)                           #
    ACSI = DatasetInfo("ACSI".lower(), "PINCP", Task.BINARY, SizeGroup.MODERATE)                           #
    WAVE_E = DatasetInfo("WAVE_E".lower(), "energy_total", Task.REGRESSION, SizeGroup.SMALL)               # 
    OKCUPID_STEM = DatasetInfo("OKCUPID_STEM".lower(), "job", Task.MULTICLASS, SizeGroup.SMALL)            #
    ACCEL = DatasetInfo("ACCEL".lower(), "wconfid", Task.MULTICLASS, SizeGroup.SMALL)                      #
    RCV1 = DatasetInfo("RCV1".lower(), "class", Task.BINARY, SizeGroup.MODERATE)                           #
    DELAYS_ZURICH = DatasetInfo("DELAYS_ZURICH".lower(), "delay", Task.REGRESSION, SizeGroup.LARGE)        #
    COMET_MC = DatasetInfo("COMET_MC".lower(), "label", Task.MULTICLASS, SizeGroup.LARGE)                  #

    EPSILON = DatasetInfo("epsilon", 'target', Task.BINARY, SizeGroup.LARGE)
    ELECTRICITY = DatasetInfo("electricity", "class", Task.BINARY, SizeGroup.SMALL) 
    EL_NINO = DatasetInfo("el_nino", "ss_temp", Task.REGRESSION, SizeGroup.SMALL)                       
    #PUF_128 = DatasetInfo("PUF_128".lower(), 128, Task.BINARY, SizeGroup.LARGE)

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

class CVInfo(dict):
    def __init__(self, cv: Union[TY_CV, dict] = None):    
        if cv is None:
            print(f"cv is None, assuming default standard KFold!")
            cv = KFold()

        mappings = cv if isinstance(cv, dict) else self._create_info(cv)
        dict.__init__(self, mappings)
    
    def fn(self) -> str:
        return f"{str(self)}_folds.json"
    
    def path(self, dir: str) -> str:
        return os.path.join(dir, self.fn())
    
    def get_n_repeats(self) -> int:
        self.get("n_repeats", 0)
    
    @staticmethod
    def _str_filter_cond(pair) -> bool:
        key, value = pair 
        ignore_keys = ("name", "stratified")
        if (key not in ignore_keys) and value is not None:
            return value if type(value) == bool else True
        return False

    def __str__(self) -> str:
        info = dict(filter(self._str_filter_cond, self.items()))
        strings = [f"{k}={v}" for k, v in info.items()]
        return f"{self['name']}[{','.join(strings)}]"

    @classmethod
    def _create_info(cls, cv: TY_CV) -> dict:
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
    
    def to_dict(self) -> dict:
        return {k:v for k, v in self.items()}

class Dataset(DatasetInfo):
    def __init__(self, d: Union[Builtin, DatasetInfo, str], is_test=False):
        if isinstance(d, Builtin):
            super().__init__(d.info().name, d.info().label_column, d.info().task, d.info().size_group, d.info().mod)
        elif isinstance(d, DatasetInfo):
            super().__init__(d.name, d.label_column, d.task, d.size_group, d.mod)
        elif isinstance(d, str) and d.__contains__("."):
            super().__init__(d, None, None, None, mod=True)
        else:
            raise RuntimeError(f"Dataset info is of invalid type ({type(d)}): {d}")
        
        self.x = None
        self.y = None
        self.cat_features: list = []
        self.is_test = is_test

        self.test_path = None
        self.train_path = None
        self.saved_folds_path = None

        # Need to load module datasets to find task.
        if self.mod:
            self.load()

        self.__set_dataset_paths()
    
    @classmethod
    def splitext(cls, fn: str) -> Tuple[str, str]:
        parts = fn.split('.')
        name = parts[0]
        ext = "." + ".".join(parts[1:]) if (len(parts) > 1) else ''
        return name, ext

    @classmethod
    def try_from(cls, bn: Builtin, log=True, load=True) -> Optional['Dataset']:
        try:
            return cls(bn).load() if load else cls(bn)
        except RuntimeError as e:
            if log:
                print(e)
            return None

    def get_builtin(self):
        return Builtin[self.name.upper()]

    def get_dir(self) -> str:
        path = data_dir(f"datasets/{self.name}", make_add_dirs=self.mod)
        if not os.path.exists(path):
            raise RuntimeError(f"Dataset path dosen't exist: {path}")
        return path
    
    def __set_dataset_paths(self):
        path = self.get_dir()
        if self.mod is not None:
            return

        fns, exts = zip(*[self.splitext(f) for f in os.listdir(path)])
        
        try:
            train_idx = fns.index(f"{self.name}_train")
            test_idx = fns.index(f"{self.name}_test")
            self.test_path = os.path.join(path, f"{fns[test_idx]}{exts[test_idx]}")
        except:
            try: 
                train_idx = fns.index(self.name)
                test_idx = None
            except:
                raise RuntimeError(f"Dataset files not present({path}): {fns}")
        
        self.train_path = os.path.join(path, f"{fns[train_idx]}{exts[train_idx]}")

        if not os.path.exists(self.train_path):
            raise RuntimeError(f"Dataset {self.name} not found in data folder: {self.train_path}")

        if (self.test_path is not None) and (not os.path.exists(self.test_path)):
            raise RuntimeError(f"Test data for dataset '{self.name}' not found in data folder: {self.test_path}")
    
    def has_test_set(self) -> bool:
        return self.test_path is not None
    
    def has_saved_folds(self, cv: TY_CV) -> bool:
        info = CVInfo(cv)
        return os.path.exists(info.path(self.get_dir()))

    def load_saved_folds_file(self, cv: TY_CV) -> dict:
        info = CVInfo(cv)
        path = info.path(self.get_dir())
        if not os.path.exists(path):
            raise RuntimeError(f"No {str(info)} saved folds data found for dataset {self.name}")

        data = load_json(path)
        folds = data["folds"]
        data["folds"] = [
            (
                np.array(fold["train"]).reshape(fold["shape_train"]), 
                np.array(fold["test"]).reshape(fold["shape_test"])
            ) for fold in folds]

        return data
    
    def load_saved_cv_folds_info(self, cv: TY_CV):
        return self.load_saved_folds_file(cv)["info"]
    
    def load_saved_cv_folds(self, cv: TY_CV) -> dict:
        return self.load_saved_folds_file(cv)["folds"]
        
    def save_folds(self, cv: TY_CV):
        info = CVInfo(cv)
        path = info.path(self.get_dir())

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
        save_json(path, data=dict(info=info.to_dict(), folds=folds), indent=None)
    
    def __load_from_module(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mod, name = self.name.split('.', maxsplit=1)
        func = None
        for prefix in ("load_", "fetch_"):
            func = getattr(sk_datasets, f"{prefix}{name}", None)
            if func is not None:
                break
        if func is None:
            raise ValueError(f"Sklearn dataset {name} not found!")

        x, y =  func(as_frame=True, return_X_y=True)
        size = x.shape[0]

        self.name = name
        self.mod = mod

        if size < 100_000:
            self.size = SizeGroup.SMALL
        elif size < 500_000:
            self.size = SizeGroup.MODERATE
        else:
            self.size = SizeGroup.LARGE
        
        if isinstance(y.dtype, (object, int)):
            if y.nunique() < 2:
                self.task = Task.BINARY
            else:
                self.task = Task.MULTICLASS
        else:
            self.task = Task.REGRESSION
        
        return (x, y)
    
    def __load(self, load_labels_only=False, force_load_test=False) -> pd.DataFrame:
        if self.mod is not None:
            return self.__load_from_module() if (self.x is None) else None
        elif (self.is_test or force_load_test) and (self.test_path is None):
            raise RuntimeError(f"Test path for {self.name} dataset not found")
        else:
            path = self.test_path if (self.is_test or force_load_test) else self.train_path 
            print(f"Loading dataset from path: {path}")

        fn, ext = self.splitext(os.path.basename(path))
        ext = ext.strip()
        if ext.count('.') > 1:
            exts = tuple(filter(None, ext.split('.'))) 
            ext = f".{exts[0]}"
            
        if ext == ".csv":
            if has_csv_header(self.train_path):
                return pd.read_csv(path) if not load_labels_only else pd.read_csv(path, usecols=[self.label_column])
            else:
                n_cols = get_n_csv_columns(self.train_path)
                head = tuple(range(n_cols))
                data = pd.read_csv(path, names=head) if not load_labels_only else pd.read_csv(path, usecols=[self.label_column], names=head)
                return data
        elif ext == ".arff":
            return load_arff(path) if not load_labels_only else extract_labels(load_arff(path), label_column=self.label_column)[1]
        elif ext == '.libsvm':
            return load_libsvm(path) if not load_labels_only else extract_labels(load_arff(path), label_column=self.label_column)[1]
    
    def load_test_dataset(self) -> 'Dataset':
        return Dataset(self.get_builtin(), is_test=True).load()
    
    def load(self) -> 'Dataset':
        data = self.__load()
        if data is None:
            return self
        elif isinstance(data, tuple):
            self.x, self.y = data
            shape = (self.x.shape[0], self.x.shape[1] + 1)
        else:
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
    
    def load_frame(self) -> pd.DataFrame:
        return self.__load()
    
    @classmethod
    def merge_train_test(cls, d: Builtin):
        dataset = Dataset(d)
        train = dataset.__load()
        test = dataset.__load(force_load_test=True)
        
        train_shape = train.shape
        test_shape = test.shape 
        
        is_sparse = train.dtypes.apply(pd.api.types.is_sparse).all()
        print(f"train={train.shape}, test={test.shape}")
        if is_sparse:
            from scipy import sparse
            joined = sparse.vstack([train, test])
        else:
            joined = pd.concat([train, test], ignore_index=True)

        correct_shape = (train_shape[0] + test_shape[0], train_shape[1])
        print(f"{d.name} joind: joined_shape={joined.shape}, expected={correct_shape}")
        assert joined.shape == correct_shape

        if is_sparse:
            path = os.path.join(os.path.dirname(dataset.train_path), f"{d.name.lower()}.arff")
            save_sparse_arff(path, joined)
        else:
            path = os.path.join(os.path.dirname(dataset.train_path), f"{d.name.lower()}.csv")
            joined.to_csv(path, index=False)
