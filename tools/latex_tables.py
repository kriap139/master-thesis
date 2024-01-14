import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct
import numbers

SPACES_L1 = "   "
SPACES_L2 = SPACES_L1 * 2
SPACES_L3 = SPACES_L1 * 3

"""
\\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c}
    \hline 
    \multirow{2}{*}{Dataset}  & \multicolumn{2}{c|}{Random Search} & \multicolumn{2}{c|}{Grid Search} & \multicolumn{2}{c|}{SeqUD} & \multicolumn{2}{c|}{SigOpt} & \multicolumn{2}{c}{Optuna} \\
    \cline{2-11}
    & Train & Test & Train & Test & Train & Test & Train & Test & Train & Test \\
    \hline
    Wave\_E & ? & ? & ? & ? & ? & ? & ? & ? & ? & ? \\
    sgemm\_GKP & ? & ? & ? & ? & ? & ? & ? & ? & ? & ? \\
    FPS & ? & ? & ? & ? & ? & ? & ? & ? & ? & ? \\
    Airlines & ? & ? & ? & ? & ? & ? & ? & ? & ? & ? \\
    \hline
\end{tabular}  
"""

def latex_table_skel(tabular: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[H]\n"
        f"{SPACES_L1}\\centering\n"
        f"{SPACES_L1}\\resizebox{{\\textwidth}}{{!}}{{\n"
        f"{tabular}\n"
        f"{SPACES_L1}}}\n"
        f"{SPACES_L1}\\caption{{{caption}}}\n"
        f"{SPACES_L1}\\label{{tab:{label}}}\n"
        f"\\end{{table}}"
    ) 

def col_format(n_cols: int, col_lines=True, outer_col_lines=True) -> str:
    if col_lines and outer_col_lines:
        col_format = "|" + ("c|" * n_cols)
    elif col_lines and not outer_col_lines:
        col_format =  "c|" * (n_cols - 1) + 'c'
    else:
        col_format = "c" * n_cols
    return col_format

def latex_tabular_skel(header: str, rows: List[str], n_cols: int, col_lines=True, outer_col_lines=True) -> str:
    newline = '\n'
    return (
        f"{SPACES_L2}\\begin{{tabular}}{{{col_format(n_cols, col_lines,  outer_col_lines)}}}\n"
        f"{header}"
        f"{newline.join(rows)}\n"
        f"{SPACES_L2}\\end{{tabular}}"
    )

def latex_tabular_header(column_labels: List[str], bold=True, row_lines=True, outer_row_lines=True, add_row_label: str = None):
    if bold:
        labels = [f"\\textbf{{{label}}}" for label in column_labels]
    else:
        labels = column_labels
    
    if add_row_label is not None:
        row_label = f"{add_row_label} & "
    else:
        row_label = ""
    
    header = row_label + " & ".join(labels) + " \\\\"
    return (SPACES_L3 + "\\hline\n" + SPACES_L3 + header + " \\hline\n") if row_lines else header

def latex_tabular_multicol_header(labels: List[str], sub_labels: List[str], rest_space_idx: int = None, bold_labels=True, bold_sub_labels=False, row_lines=True, outer_row_lines=True, col_lines=True, outer_col_lines=True) -> str:
    if bold_labels:
        labels = [f"\\textbf{{{label}}}" for label in labels]
    if bold_sub_labels:
        sub_labels = [f"\\textbf{{{label}}}" for label in sub_labels]

    row_line = " \\hline\n" if row_lines else "\n"
    parts = []

    for i, label in enumerate(labels):
        if rest_space_idx is not None and (i == rest_space_idx):
            parts.append(f"\\multirow{{{len(sub_labels)}}}{{*}}{{{label}}}")
        else:
            parts.append(f"\\multirow{{{len(sub_labels)}}}{{{col_format(1, col_lines, outer_col_lines)}}}{{{label}}}")
    
    label_header = f"{SPACES_L3}\\hline\n" + SPACES_L3 + " & ".join(parts) + " \\\\\n"
    parts.clear()

    for i, _ in enumerate(labels):
        if rest_space_idx is not None and (i == rest_space_idx):
            parts.append(f"& ")
        else:
            parts.extend(" & ".join(sub_labels))
            if i == (len(labels) - 1):
                parts.append(" \\\\")
            else:
                parts.append(f" & ")
    
    label_header += SPACES_L3 + "".join(parts) + row_line
    return label_header

def latex_tabular_rows(data: pd.DataFrame, n_round: int = None, row_lines=True, outer_row_lines=True, add_row_labels=False, col_cells_postfix: Union[str, List[str]] = None) -> List[str]:
    table_rows = []
    row_line = "\\hline" if row_lines else ""

    def rounder(v: Any):
        if n_round is None or (not isinstance(v, numbers.Number)):
            return v
        elif n_round == 0:
            return int(round(v, n_round))
        else:
            return round(v, n_round)

    def create_cell(col: int, v: Any):
        v = str(rounder(v))       
        if type(col_cells_postfix) == str:
            return v + col_cells_postfix
        elif type(col_cells_postfix) == list:
            return v + col_cells_postfix[col]
        else:
            return v
    
    if add_row_labels:
        row_labels = [f"{label} & " for label in data.index]
    else:
        row_labels = ["" for _ in data.index]

    n_rows = len(data.index)
    for i, (label, row) in enumerate(data.iterrows()):
        row_data = [create_cell(col, cell) for col, cell in enumerate(row)]
        if (i == n_rows - 1) and not outer_row_lines:
            table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\")
        else:
            table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {row_line}")
    return table_rows

def create_latex_table(
    data: pd.DataFrame,
    label: str,
    caption: str = None,
    add_column_labels=True, 
    add_row_label: str = None,
    round: int = None, 
    bold_header=True, 
    row_lines=True, 
    outer_row_lines=True,
    col_lines=True,
    outer_col_lines=True, 
    col_cells_postfix: Union[str, List[str]] = None) -> str:

    if caption is None:
        caption = str(caption)

    if add_row_label is not None:
        n_cols = len(data.columns) + 1
    else:
        n_cols = len(data.columns)        
    
    header = latex_tabular_header(data.columns, bold_header, row_lines, outer_col_lines, add_row_label)
    rows = latex_tabular_rows(data, round, row_lines, outer_row_lines, add_row_label is not None, col_cells_postfix)
    tabular = latex_tabular_skel(header, rows, n_cols, col_lines, outer_col_lines)
    table = latex_table_skel(tabular, caption, label)
    return table

def encode_multicol_labels(labels: List[str], sub_labels: List[str], rest_space_idx: int = None) -> List[str]:
    columns = []
    for i, label in enumerate(labels):
        if rest_space_idx is not None and (i == rest_space_idx):
            columns.append(f"rest_space_{label}")
        else:
            for sub_label in sub_labels:
                columns.append(f"sublabel_{label}_{sub_label}")
    return columns

def decode_multicol_labels(data: pd.DataFrame) -> Tuple[List[str], List[str], int]:
    labels = []
    sub_labels = []
    rest_space_idx = None

    for i, col in enumerate(data.columns):
        if col.startswith("rest_space_"):
            col_label = col.removeprefix("rest_space_")
            labels.append(col_label)
            rest_space_idx = i
        elif col.startswith("sublabel_"):
            arr = col.split("_")
            if not (arr[1] in labels):
                labels.append(arr[1])
            if not (arr[2] in sub_labels):
                sub_labels.append(arr[2])
        else:
            raise ValueError(f"Invalid multicol label encoding: {col}")
    
    return list(labels), list(sub_labels), rest_space_idx
    

def create_multicol_latex_table(
    data: pd.DataFrame, 
    label: str,
    caption: str = None,
    add_column_labels=True,
    round: int = None, 
    bold_header=True, 
    bold_sub_header=False,
    row_lines=True, 
    outer_row_lines=True,
    col_lines=True,
    outer_col_lines=True,
    col_cells_postfix: Union[str, List[str]] = None) -> str:

    if caption is None:
        caption = str(caption)
    
    rest_space_idx = None
    col_labels, col_sub_labels, rest_space_idx = decode_multicol_labels(data) 

    header = latex_tabular_multicol_header(
        col_labels, col_sub_labels, rest_space_idx, bold_header, 
        bold_sub_header, row_lines, outer_row_lines, col_lines, outer_col_lines
    )
    rows = latex_tabular_rows(data, round, row_lines, outer_row_lines, False, col_cells_postfix)
    tabular = latex_tabular_skel(header, rows, len(data.columns), col_lines, outer_col_lines)
    table = latex_table_skel(tabular, caption, label)
    return table

def create_test_results_stats_table(data: EvalMetrics) -> str:
    table_data = dict()
    labels = ["Dataset"]
    labels.extend(data.get_method_names())
    sub_labels = ["mean", "std", "min", "max", "median"]
    
    header = encode_multicol_labels(labels, sub_labels, rest_space_idx=0)
    for i, (dataset, results) in enumerate(data.results.items()):
        row = [None for _ in header]
        row[0] = dataset.lower()
        for (method, result) in results.items():
            row[header.index(f"sublabel_{method}_mean")] = result["result"]["mean_test_acc"]
            row[header.index(f"sublabel_{method}_std")] = result["result"]["std_test_acc"]
            row[header.index(f"sublabel_{method}_min")] = result["result"]["min_test_acc"]
            row[header.index(f"sublabel_{method}_max")] = result["result"]["max_test_acc"]
            row[header.index(f"sublabel_{method}_median")] = result["result"]["meadian_test_acc"]
        table_data[i] = row
    
    frame = pd.DataFrame.from_dict(table_data, orient='index', columns=header)
    return create_multicol_latex_table(frame, "test_result_stats", None, round=3)

def create_time_pct_table(data: EvalMetrics) -> str:
    pct = time_frame_pct(data)
    latex_dict = dict(Dataset=list(map(lambda s: s.lower(), data.results.keys())))
    latex_dict.update(pct.to_dict(orient='list'))
    latex_frame = pd.DataFrame.from_dict(latex_dict)
    return create_latex_table(latex_frame, "test_result_times", None, round=0, col_cells_postfix='%')

def create_ns_rank_table(data: EvalMetrics) -> str:
    table_data = dict()
    labels = ["Dataset"]
    labels.extend(data.get_method_names())
    sub_labels = ["ns", "rank"]
    
    header = encode_multicol_labels(labels, sub_labels, rest_space_idx=0)
    ranks = data.mean_accs.rank(axis=1, ascending=False)
    for i, (dataset, results) in enumerate(data.results.items()):
        row = [None for _ in header]
        row[0] = dataset.lower()
        for (method, result) in results.items():
            row[header.index(f"sublabel_{method}_ns")] = data.normalized_scores.at[method, dataset]
            row[header.index(f"sublabel_{method}_rank")] = ranks.at[dataset, method]
        table_data[i] = row
    
    frame = pd.DataFrame.from_dict(table_data, orient='index', columns=header)
    return create_multicol_latex_table(frame, "test_result_stats", None, round=3)

def create_method_metrics_table(data: EvalMetrics) -> str:
    latex_dict = dict(
        AS=data.agg_scores.to_dict(),
        RS=data.rank_scores.to_dict(),
        NAS=data.nas.to_dict(),
        NRS=data.nrs.to_dict(),
        JS=data.js.to_dict()
    )
    frame = pd.DataFrame.from_dict(latex_dict)
    return create_latex_table(frame, "method_metrics", None, round=3, add_row_label="Dataset")

def save_table(table: str):
    with open("table.txt", mode='w') as f:
        f.write(table)

if __name__ == "__main__":
    ignore_datasets = (Builtin.AIRLINES.name, )
    result_folders = load_result_folders(ignore_datasets)

    for method, result in result_folders.items():
        datasets = tuple(result.keys())
        print(f"{method}: {datasets}, len={len(datasets)}")

    print()
    metrics = calc_eval_metrics(result_folders)
    print()

    table = create_method_metrics_table(metrics)
    print(table)
    save_table(table)








