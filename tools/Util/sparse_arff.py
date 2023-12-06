import pandas as pd
from typing import Union
import gc
import arff
import re

# Code based on the  liac-arff library: https://github.com/renatopp/liac-arff

# CONSTANTS ===================================================================
_SIMPLE_TYPES = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']

_TK_DESCRIPTION = '%'
_TK_COMMENT     = '%'
_TK_RELATION    = '@RELATION'
_TK_ATTRIBUTE   = '@ATTRIBUTE'
_TK_DATA        = '@DATA'

_RE_RELATION     = re.compile(r'^([^\{\}%,\s]*|\".*\"|\'.*\')$', re.UNICODE)
_RE_ATTRIBUTE    = re.compile(r'^(\".*\"|\'.*\'|[^\{\}%,\s]*)\s+(.+)$', re.UNICODE)
_RE_TYPE_NOMINAL = re.compile(r'^\{\s*((\".*\"|\'.*\'|\S*)\s*,\s*)*(\".*\"|\'.*\'|\S*)\s*\}$', re.UNICODE)
_RE_QUOTE_CHARS = re.compile(r'["\'\\\s%,\000-\031]', re.UNICODE)
_RE_ESCAPE_CHARS = re.compile(r'(?=["\'\\%])|[\n\r\t\000-\031]')
_RE_SPARSE_LINE = re.compile(r'^\s*\{.*\}\s*$', re.UNICODE)
_RE_NONTRIVIAL_DATA = re.compile('["\'{}\\s]', re.UNICODE)






def parse_row(line: str, columns: list, column_types: list, data: dict):
    """
    Parses a row of data from an ARFF file.

    Args:
        line (str): A row from the ARFF file.
        row_len (int): Length of the row.

    Returns:
        List[Any]: Parsed row as a list of values.
    """

    if '{' in line and '}' in line:
        # Sparse data row
        line = line.replace('{', '').replace('}', '')

        # init columns to zero
        for name in columns:
            data[name] = 0
        
        for data in line.split(','):
            index, value = data.split()
            index = int(index)
            data[columns[index]] = column_types[index](value)
            indexes.append(indexes)
    else:
        # Dense data row
        for i, value in enumerate(line.split(',')):
            data[columns[i]] = column_types[i](value)

    return row
    
def _load_sparse_arff(path: str) -> pd.DataFrame:
    """
    Converts an ARFF file to a DataFrame.

    Args:
        path (Union[str, Path]): Path to the input ARFF file.

    Returns:
        pd.DataFrame: Converted DataFrame.
    """

    columns = []
    column_types = []

    len_attr = len('@attribute')
    line = '\n'

    with open(path, 'r') as fp:
        line = fp.readline()

        # Parsing metadata
        while True:
            if line.startswith(('@attribute ', '@ATTRIBUTE ')):
                name, ty = line[len_attr:].split()
                columns.append(name.strip())
                column_types.append(get_attr_type(ty))
            elif line.startswith(("@data", "@DATA")):
                break
            elif line.startswith(("@relation", "@RELATION", '%', '\n')):
                line = fp.readline()
                continue
            else:
                raise RuntimeError(f"Invalid line in sparse arff file: {line}")
            
            gc.collect()
            print(f"Columns: {len(columns)}")

        data = {col: [] for col in columns}
        print(data)
        exit(0)
        line = fp.readline()

        while line != '':
            parse_row(line, columns, column_types, data)
            line = fp.readline()
            gc.collect()

    return pd.DataFrame(data)

def load_sparse_arff(path: str) -> pd.DataFrame:

    with open(path, mode='r') as f:
        decoder = arff.ArffDecoder()
        d = decoder.decode(f, return_type=arff.DENSE)
        
        print(d)
        exit(0)