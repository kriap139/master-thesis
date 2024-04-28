import arff
import pandas as pd
from typing import List, Callable, Union
from scipy import sparse
import gc

def _load_arff(path: str) -> dict:
    with open(path, mode='r') as f:
        decoder = arff.ArffDecoder()
        return decoder.decode(f, return_type=arff.COO)

def load_sparse_arff(path: str) -> pd.DataFrame:
    d = _load_arff(path)
    data = d['data'][0]
    row = d['data'][1]
    col = d['data'][2]

    columns = tuple(tup[0] for tup in d["attributes"])  
    matrix = sparse.coo_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
    df = pd.DataFrame.sparse.from_spmatrix(matrix, columns=columns)
    print(df.info())
    return df

def save_sparse_arff(path: str, name: str, data: Union[pd.DataFrame, Callable[[], pd.DataFrame]]):
    is_callable = callable(data)
    if is_callable:
        data = data()

    attributes = [(j, 'NUMERIC') if data[j].dtypes in ['int64', 'float64'] else (j, data[j].unique().astype(str).tolist()) for j in data]
    arff_dic = {
        'attributes': attributes,
        'data': data.sparse.to_coo(),
        'relation': name,
        'description': f"{name} dataset"
    }

    if is_callable:
        del data
        gc.collect()
    
    with open(path, mode='w', encoding='utf8') as f:
        arff.dump(arff_dic, f)