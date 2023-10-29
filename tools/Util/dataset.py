from enum import Enum
from typing import Tuple, Union, Optional
import pandas as pd
import os
from dataclasses import dataclass, field
import subprocess
from Util.io_util import arff_to_csv, data_dir, load_arff

class Dataset(Enum):
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

@dataclass(eq=True, frozen=True)
class DatasetInfo:
    train_path: str
    test_path: str
    label_column: Union[str, int]

def extract_labels(data: pd.DataFrame, label_column: Union[str, int], remove=True, inplace=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def get_dataset_info(dataset: Dataset) -> DatasetInfo:
    name = dataset.name.lower()
    path = data_dir(f"datasets/{name}")
    label_column = TARGET_COLUMNS[dataset]

    fns, exts = zip(*[os.path.splitext(f) for f in os.listdir(path)])

    try:
        idx = fns.index(f"{name}_train")
        test_path = os.path.join(path, f"{fns[idx]}{exts[idx]}")
    except:
        try: 
            idx = fns.index(name)
            test_path = None
        except:
            raise RuntimeError(f"Dataset files not present({path}): {fns}")
    
    train_path = os.path.join(path, f"{fns[idx]}{exts[idx]}")

    if not os.path.exists(train_path):
        raise RuntimeError(f"Dataset {name} not found in data folder: {train_path}")

    if (test_path is not None) and (not os.path.exists(test_path)):
        raise RuntimeError(f"Test data for dataset '{name}' not found in data folder: {test_path}")
    
    return DatasetInfo(train_path, test_path, label_column)

def load_dataset(info: DatasetInfo, load_test=False, usecols=None) ->  Optional[pd.DataFrame]:
    if load_test and info.test_path is None:
        return None
    else:
        path = info.test_path if load_test else info.train_path
    
    fn, ext = os.path.splitext(os.path.basename(path))

    if ext.strip() == ".csv":
        return pd.read_csv(path) if usecols is None else pd.read_csv(path, usecols=usecols)
    elif ext.strip() == ".arff":
        return load_arff(path)

def get_train_dataset(dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info = get_dataset_info(dataset)
    df = load_dataset(info)
    # print(df.head())
    return extract_labels(df, label_column=info.label_column)

def get_test_dataset(dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info = get_dataset_info(dataset)
    if info.test_path is None:
        return None, None
    
    df = load_dataset(info)
    # print(df.head())
    return extract_labels(df, label_column=info.label_column)

def get_dataset_labels(dataset: Dataset, include_test=True) -> pd.DataFrame:
    info = get_dataset_info(dataset)
    y_train = load_dataset(info, usecols=[info.label_column])

    if ".arff" in info.train_path:
        _, y_train = extract_labels(y_train, label_column=info.label_column)

    # print(y_train.head())

    if info.test_path is not None and include_test:
        y_test = load_dataset(info, usecols=[info.label_column], load_test=True)
        #print(y_test.head())

        y = pd.concat([y_train, y_test])
        assert y.shape[0] == (y_train.shape[0] + y_test.shape[0])
        assert y.shape[1] == 1
    else:
        y = y_train
    
    return y

