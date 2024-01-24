import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Iterable
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct, time_frame_stamps, sort_folders
from Util import Task, SizeGroup
import numbers
from dataclasses import dataclass

@dataclass(frozen=True, eq=True)
class RowLabel:
   start_idx: int
   label: str
   bold: bool 
   end_idx: int = None

SPACES_L1 = "   "
SPACES_L2 = SPACES_L1 * 2
SPACES_L3 = SPACES_L1 * 3

ROW_LINE = "\\hline"
ROW_LINE_NL = f"\n{SPACES_L3}{ROW_LINE}"



class Latex:
    def __init__(self):
        self.table = ""
    



class LatexTable(Latex):
    pass

class LatexMulticolTable(Latex):
    pass



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

def col_format(n_cols: int, col_lines=True, outer_col_lines=True, multicol_header=False) -> str:
    if col_lines and outer_col_lines:
        col_format = "|" + ("c|" * n_cols)
    elif col_lines and not outer_col_lines:
        if multicol_header:
            col_format = "c|"
        else:
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
        if add_row_label is not None:
            add_row_label = f"\\textbf{{{add_row_label}}}"
    else:
        labels = column_labels
    
    if add_row_label is not None:
        row_label = f"{SPACES_L3}{add_row_label} & "
    else:
        row_label = f"{SPACES_L3}"
    
    header = row_label + " & ".join(labels) + " \\\\"
    
    if outer_row_lines:
        header = ROW_LINE_NL + header
    if row_lines:
        header += ROW_LINE_NL

    return header + '\n'

def latex_tabular_multicol_header(labels: List[str], sub_labels: List[str], rest_space_idx: int = None, bold_labels=True, bold_sub_labels=False, row_lines=True, outer_row_lines=True, col_lines=True, outer_col_lines=True) -> str:
    if bold_labels:
        labels = [f"\\textbf{{{label}}}" for label in labels]
    if bold_sub_labels:
        sub_labels = [f"\\textbf{{{label}}}" for label in sub_labels]

    if rest_space_idx is not None:
        cline = f"\n{SPACES_L3}\\cline{{{len(sub_labels)}-{len(labels) * len(sub_labels) - len(sub_labels) + 1}}}\n"
    else:
        cline = ROW_LINE_NL

    row_line = "\\hline\n" if row_lines else "\n"
    parts = []

    for i, label in enumerate(labels):
        if rest_space_idx is not None and (i == rest_space_idx):
            parts.append(f"\\multirow{{{len(sub_labels)}}}{{*}}{{{label}}}")
        elif (i == len(labels) - 1):
            parts.append(f"\\multicolumn{{{len(sub_labels)}}}{{{col_format(1, col_lines, False)}}}{{{label}}}")
        else:
            parts.append(f"\\multicolumn{{{len(sub_labels)}}}{{{col_format(1, col_lines, False, True)}}}{{{label}}}")
    
    label_header = f"{SPACES_L3}\\hline\n" + SPACES_L3 + " & ".join(parts) + f" \\\\{cline}"
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

def latex_span_row_label(label: str, n_cols: int, bold=False, row_line_top=True, row_line_bottom=True, outer_col_lines=True) -> str:
    col_fmt = col_format(1, col_lines=False, outer_col_lines=outer_col_lines)
    row_line_top = f"{SPACES_L3}{ROW_LINE}\n" if row_line_top else ""
    row_line_bottom = ROW_LINE_NL if row_line_bottom else ""

    if bold:
        label = f"\\textbf{{{label}}}"

    return f"{row_line_top}{SPACES_L3}\\multicolumn{{{n_cols}}}{{{col_fmt}}}{{{label}}} \\\\ {row_line_bottom}"

def latex_string_escapes(s: str) -> str:
    escapes = ["_", "%"]
    for esc in escapes:
        s = s.replace(esc, f"\\{esc}")
    return s

def latex_tabular_rows(data: pd.DataFrame, n_round: int = None, row_lines=True, outer_row_lines=True, add_row_labels=False, col_cells_postfix: Union[str, List[str]] = None) -> List[str]:
    table_rows = []
    row_line = ROW_LINE_NL if row_lines else ""

    if col_cells_postfix is not None:
        if type(col_cells_postfix) == str:
            col_cells_postfix = latex_string_escapes(col_cells_postfix)
        elif type(col_cells_postfix) == list:
            col_cells_postfix = [latex_string_escapes(s) for s in col_cells_postfix]
        else:
            raise RuntimeError(f"col_cells_postfix is of invalid type({type(col_cells_postfix)}): {col_cells_postfix}")

    def rounder(v: Any):
        if n_round is None:
            return v
        elif n_round == 0:
            return int(round(v, n_round))
        else:
            return round(v, n_round)

    def create_cell(col: int, v: Any):
        if isinstance(v, numbers.Number):
            v = str(rounder(v))
        else:
            v = latex_string_escapes(v) 

        if type(col_cells_postfix) == str:
            return v + col_cells_postfix
        elif type(col_cells_postfix) == list:
            return v + col_cells_postfix[col]
        else:
            return v
    
    if add_row_labels:
        row_labels = [f"{latex_string_escapes(label)} & " for label in data.index]
    else:
        row_labels = ["" for _ in data.index]

    last_row = len(data.index) - 1
    for i, (label, row) in enumerate(data.iterrows()):
        row_data = [create_cell(col, cell) for col, cell in enumerate(row)]
        if (i == last_row) and not outer_row_lines:
            table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\")
        elif (i == last_row) and outer_row_lines and not row_lines:
            table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {ROW_LINE_NL}")
        else:
            table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {row_line}")
    return table_rows

def latex_tabular_rows_with_label_rows(data: pd.DataFrame, n_cols: int, label_rows: Iterable[RowLabel], n_round: int = None, row_lines=True, outer_row_lines=True, outer_col_lines=False, add_row_labels=False, col_cells_postfix: Union[str, List[str]] = None) -> List[str]:
    rows = []

    for lr in label_rows:
        if lr.end_idx is not None:
            rows.append(latex_span_row_label(lr.label, n_cols, lr.bold, row_line_top=outer_row_lines, row_line_bottom=False, outer_col_lines=outer_col_lines))
            subset = data.iloc[lr.start_idx:lr.end_idx, :]
            rows.extend(
                latex_tabular_rows(subset, n_round, row_lines, False, add_row_labels, col_cells_postfix)
            )
        else:
            rows.append(latex_span_row_label(lr.label, n_cols, lr.bold, row_line_top=outer_row_lines, row_line_bottom=False, outer_col_lines=outer_col_lines))
            subset = data.iloc[lr.start_idx:, :]
            rows.extend(
                latex_tabular_rows(subset, n_round, row_lines, outer_row_lines, add_row_labels, col_cells_postfix)
            )
    return rows

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
    col_cells_postfix: Union[str, List[str]] = None,
    row_labels: Iterable[RowLabel] = None) -> str:

    if caption is None:
        caption = str(caption)
    
    rest_space_idx = None
    col_labels, col_sub_labels, rest_space_idx = decode_multicol_labels(data) 

    header = latex_tabular_multicol_header(
        col_labels, col_sub_labels, rest_space_idx, bold_header, 
        bold_sub_header, row_lines, outer_row_lines, col_lines, outer_col_lines
    )

    if row_labels is None:
        rows = latex_tabular_rows(data, round, row_lines, outer_row_lines, False, col_cells_postfix)
    else:
        rows = latex_tabular_rows_with_label_rows(data, len(data.columns), row_labels, round, row_lines, outer_row_lines, outer_col_lines, False, col_cells_postfix)

    tabular = latex_tabular_skel(header, rows, len(data.columns), col_lines, outer_col_lines)
    table = latex_table_skel(tabular, caption, label)
    return table

def create_task_filter_fn(task: Task):
    return lambda folder: folder.dataset.info().task in task



def create_test_results_stats_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label='test_result_stats') -> str:
    folders = load_result_folders(ignore_datasets, filter_fn=filter_fn, sort_fn=sort_fn, reverse=True)
    data = calc_eval_metrics(folders)

    table_data = dict()
    labels = ["Dataset"]
    labels.extend(data.get_method_names())
    sub_labels = ["mean", "std", "min", "max"]
    
    header = encode_multicol_labels(labels, sub_labels, rest_space_idx=0)
    for i, (dataset, results) in enumerate(data.results.items()):
        row = [None for _ in header]
        row[0] = dataset.lower()
        for (method, result) in results.items():
            row[header.index(f"sublabel_{method}_mean")] = result["result"]["mean_test_acc"]
            row[header.index(f"sublabel_{method}_std")] = result["result"]["std_test_acc"]
            row[header.index(f"sublabel_{method}_min")] = result["result"]["min_test_acc"]
            row[header.index(f"sublabel_{method}_max")] = result["result"]["max_test_acc"]
        table_data[i] = row
    
    frame = pd.DataFrame.from_dict(table_data, orient='index', columns=header)
    return create_multicol_latex_table(frame, label, None, round=3, row_lines=False, outer_col_lines=False)

def create_test_results_stats_tables(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True) -> str:
    reg = create_test_results_stats_table(ignore_datasets, create_task_filter_fn(Task.REGRESSION), sort_fn, sort_reverse)
    classif = create_test_results_stats_table(ignore_datasets, create_task_filter_fn(Task.BINARY | Task.MULTICLASS), sort_fn, sort_reverse)
    return f"{reg}\n\n{classif}"



def create_train_test_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label="test_result_stats") -> str:
    folders = load_result_folders(ignore_datasets, filter_fn=filter_fn, sort_fn=sort_fn, reverse=True)

    data = calc_eval_metrics(folders)
    table_data = dict()
    labels = ["Dataset"]
    labels.extend(data.get_method_names())
    sub_labels = ["Train", "Test"]
    
    header = encode_multicol_labels(labels, sub_labels, rest_space_idx=0)
    for i, (dataset, results) in enumerate(data.results.items()):
        row = [None for _ in header]
        row[0] = dataset.lower()
        for (method, result) in results.items():
            row[header.index(f"sublabel_{method}_Train")] = result["result"]["mean_train_acc"]
            row[header.index(f"sublabel_{method}_Test")] = result["result"]["mean_test_acc"]
        table_data[i] = row
    
    frame = pd.DataFrame.from_dict(table_data, orient='index', columns=header)
    return create_multicol_latex_table(frame, label, None, round=3, row_lines=False, outer_col_lines=False)

def create_train_test_tables(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True) -> str:
    reg = create_train_test_table(ignore_datasets, create_task_filter_fn(Task.REGRESSION), sort_fn, sort_reverse, label='baseline_results_reg_train_test')
    classif = create_train_test_table(ignore_datasets, create_task_filter_fn(Task.BINARY | Task.MULTICLASS), sort_fn, sort_reverse, label='baseline_results_cls_train_test')
    return f"{reg}\n\n{classif}"



def create_time_pct_table(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True, label='baseline_times_pct') -> str:
    folders = load_result_folders(ignore_datasets, sort_fn=sort_fn, reverse=True)
    data = calc_eval_metrics(folders)
    pct = time_frame_pct(data)

    mins = pct.min().sort_values()
    pct = pct[mins.index]

    latex_dict = dict(Dataset=list(map(lambda s: s.lower(), data.results.keys())))
    latex_dict.update(pct.to_dict(orient='list'))
    latex_frame = pd.DataFrame.from_dict(latex_dict)

    cells_postfix = [""]
    cells_postfix.extend(['%' for _ in range(len(latex_frame.columns) - 1)])

    return create_latex_table(latex_frame, label, None, round=0, row_lines=False, outer_col_lines=False, col_cells_postfix=cells_postfix)

def create_time_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label="test_result_stats") -> str:
    folders = load_result_folders(ignore_datasets, filter_fn=filter_fn, sort_fn=sort_fn, reverse=True)
    data = calc_eval_metrics(folders)
    pct = time_frame_pct(data)
    stamps = time_frame_stamps(data)

    mins = pct.min().sort_values()
    pct = pct[mins.index]
    stamps = stamps[mins.index]

    table_data = dict()
    labels = ["Dataset"]
    labels.extend(pct.columns)
    sub_labels = ["pct", "stamp"]
    
    header = encode_multicol_labels(labels, sub_labels, rest_space_idx=0)
    cells_postfix = ["" for _ in header]

    for i, (dataset, results) in enumerate(pct.iterrows()):
        row = [None for _ in header]
        row[0] = dataset.lower()
        for (method, pct) in results.items():
            row[header.index(f"sublabel_{method}_pct")] = pct
            row[header.index(f"sublabel_{method}_stamp")] = stamps.at[dataset, method]
            cells_postfix[header.index(f"sublabel_{method}_pct")] = '%'
        table_data[i] = row
        
    frame = pd.DataFrame.from_dict(table_data, orient='index', columns=header)
    return create_multicol_latex_table(frame, label, None, round=0, row_lines=False, outer_col_lines=False, col_cells_postfix=cells_postfix)

def create_ns_rank_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label='baseline_ns_ranks') -> str:
    folders = load_result_folders(ignore_datasets, filter_fn=filter_fn, sort_fn=sort_fn, reverse=True)
    data = calc_eval_metrics(folders)

    table_data = dict()
    labels = ["Dataset"]
    labels.extend(data.get_method_names())
    sub_labels = ["ns", "rank"]
    header = encode_multicol_labels(labels, sub_labels, rest_space_idx=0)
    
    for i, (dataset, results) in enumerate(data.results.items()):
        row = [None for _ in header]
        row[0] = dataset.lower()
        for (method, result) in results.items():
            row[header.index(f"sublabel_{method}_ns")] = data.normalized_scores.at[method, dataset]
            row[header.index(f"sublabel_{method}_rank")] = data.mean_ranks.at[dataset, method]
        table_data[i] = row
    
    frame = pd.DataFrame.from_dict(table_data, orient='index', columns=header)

    methods = data.get_method_names()
    labels = [f"sublabel_{method}_ns" for method in methods]
    summed = frame[labels].sum().sort_values(ascending=False)
    
    labels = ["rest_space_Dataset"]
    for label in summed.index:
        method = label.split("_")[1].strip()
        labels.append(f"sublabel_{method}_ns")
        labels.append(f"sublabel_{method}_rank")
    
    frame = frame[labels]

    return create_multicol_latex_table(frame, label, None, round=3, row_lines=False, outer_col_lines=False)

def create_ns_rank_tables(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True) -> str:
    reg = create_ns_rank_table(ignore_datasets, None, sort_fn, sort_reverse, label='baseline_ns_ranks_reg')
    classif = create_ns_rank_table(ignore_datasets, None, sort_fn, sort_reverse, label='baseline_ns_ranks_cls')
    return f"{reg}\n\n{classif}"



def create_method_metrics_table(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True, label='basline_metric') -> str:
    folders = load_result_folders(ignore_datasets, sort_fn=sort_fn, reverse=True)
    data = calc_eval_metrics(folders)

    latex_dict = dict(
        AS=data.agg_scores.to_dict(),
        RS=data.rank_scores.to_dict(),
        NAS=data.nas.to_dict(),
        NRS=data.nrs.to_dict(),
        JS=data.js.to_dict()
    )
    frame = pd.DataFrame.from_dict(latex_dict)
    return create_latex_table(frame, label, None, round=3, add_row_label="Dataset", row_lines=False, outer_col_lines=False)

def save_table(table: str):
    with open("table.txt", mode='w') as f:
        f.write(table)

if __name__ == "__main__":
    ignore_datasets = (Builtin.AIRLINES.name, )

    folder_sorter = lambda folder: ( 
        folder.dataset.info().task in (Task.BINARY, Task.MULTICLASS), 
        folder.dataset.info().task == Task.REGRESSION,
        folder.dataset.info().size_group == SizeGroup.SMALL,
        folder.dataset.info().size_group == SizeGroup.MODERATE,
        folder.dataset.info().size_group == SizeGroup.LARGE,
        folder.dataset.name, 
        folder.search_method
    )

    table = create_time_table(ignore_datasets, folder_sorter, sort_reverse=True)
    print(table)
    save_table(table)








