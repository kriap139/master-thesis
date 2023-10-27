import pandas as pd
from scipy.io import arff
import os

def data_dir(add: str) -> str:
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        raise RuntimeError(f"data directory dosen't exist: {data_dir}")
    return os.path.join(data_dir, add)

def load_arff(path: str) -> pd.DataFrame:
    data = arff.loadarff(path)
    train= pd.DataFrame(data[0])
    # print(train.info())

    catCols = [col for col in train.columns if train[col].dtype=="O"]
    # print(f"Catcols: {catCols}")

    train[catCols] = train[catCols].apply(lambda x: x.str.decode('utf8'))
    return train

def arff_to_csv(path: str):
    data = load_arff(path)
    print(data.head())

    fn, ext = os.path.splitext(os.path.basename(path))
    fn = f"{fn}.csv"
    print(f"Saving converted arff file: {fn}")
    data.to_csv(os.path.join(os.path.dirname(path), fn),index=False)

if __name__ == "__main__":
    path = data_dir(add="datasets/rcv1/php7t4FlC.arff")
    print(path)