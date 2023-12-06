import arff
import pandas as pd
from typing import List
from scipy import sparse

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
    matrix = sparse.coo_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1))
    return pd.DataFrame.sparse.from_spmatrix(matrix, columns=columns)