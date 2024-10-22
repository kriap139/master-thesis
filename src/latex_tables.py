import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Iterable, Optional
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct, time_frame_stamps, sort_folders, friedman_check
from calc_metrics import time_frame_deltas
from Util import Task, SizeGroup
import numbers
from dataclasses import dataclass
from itertools import chain
import itertools

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
    ESCAPES = ["_", "%", "&"]

    def __init__(self):
        self.post_process_table = None
    
    def apply_cell_post_processing(self, row: int, col: int, data: str) -> str:
        if self.post_process_table is None:
            return data

        post = self.post_process_table[row][col]
        if isinstance(post, str):
            return data + post
        elif callable(post):
            return post(data)
        return data
    
    def create_post_process_table(self, 
        data: pd.DataFrame, 
        col_cells_postfix: dict = None, 
        col_ordering: dict[str, str] = None, 
        row_ordering: str = None, 
        mark_best_in_row=None, 
        mark_best_in_column=None,
        exclude_marking_columns: list = None):

        self.post_process_table = {row: dict() for row in data.index}

        if col_cells_postfix is None:
            col_cells_postfix = {col: "" for col in data.columns}
        if exclude_marking_columns is None:
            exclude_marking_columns = []

        if mark_best_in_row is not None:
            mark_best_in_row = getattr(self, mark_best_in_row, None)
        if mark_best_in_column is not None:
            mark_best_in_column = getattr(self, mark_best_in_row, None)

        # ascending -> small to big, escending -> big to small
        best_by_row = data.idxmin(axis=0) if row_ordering == 'acending' else data.idxmax(axis=0)
        best_by_col = data.idxmin(axis=1) if row_ordering == 'acending' else data.idxmax(axis=1)

        for row in data.index:
            for col, postfix in col_cells_postfix.items():
                
                if col in exclude_marking_columns:
                    self.post_process_table[row][col] = postfix
                    continue

                if callable(mark_best_in_row) and (best_by_row[row] == col) and callable(mark_best_in_column) and (best_by_col[col] == row):
                    self.post_process_table[row][col] = lambda best: mark_best_in_column(mark_best_in_row(best + postfix))
                elif callable(mark_best_in_row) and (best_by_row[row] == col):
                    self.post_process_table[row][col] = lambda best: mark_best_in_row(best + postfix)
                elif callable(mark_best_in_column) and best_by_col[col] == row:
                    self.post_process_table[row][col] = lambda best: mark_best_in_column(best + postfix)
                else:
                    self.post_process_table[row][col] = postfix

    @classmethod
    def table_skel(cls, tabular: str, caption: str, label: str, vertical=False) -> str:
        resize_to = "\\linewidth" if vertical else "\\textwidth"

        table = (
            f"\\begin{{table}}[H]\n"
            f"{SPACES_L1}\\centering\n"
            f"{SPACES_L1}\\resizebox{{{resize_to}}}{{!}}{{\n"
            f"{tabular}\n"
            f"{SPACES_L1}}}\n"
            f"{SPACES_L1}\\caption{{{caption}}}\n"
            f"{SPACES_L1}\\label{{tab:{label}}}\n"
            f"\\end{{table}}"
        ) 

        if vertical:
            table = (
                f"\\begin{{landscape}}\n" # requires \usepackage{lscape}
                f"\\vspace*{{\\fill}}"
                f"{table}"
                f"\\vspace*{{\\fill}}"
                f"\\end{{landscape}}\n"
            )

        return table

    @classmethod
    def col_format(cls, n_cols: int, col_lines=True, outer_col_lines=True, multicol_header=False) -> str:
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
    
    @classmethod
    def tabular_skel(cls, header: str, rows: List[str], n_cols: int, col_lines=True, outer_col_lines=True) -> str:
        newline = '\n'
        return (
            f"{SPACES_L2}\\begin{{tabular}}{{{cls.col_format(n_cols, col_lines,  outer_col_lines)}}}\n"
            f"{header}"
            f"{newline.join(rows)}\n"
            f"{SPACES_L2}\\end{{tabular}}"
        )
    
    @classmethod
    def bold(cls, s: Union[str, Iterable]) -> Union[str, Iterable[str]]:
        if type(s) == str:
            return f"\\textbf{{{s}}}"
        elif isinstance(s, Iterable):
            return [f"\\textbf{{{label}}}" for label in s]
        return s
    
    @classmethod
    def underline(cls, s: Union[str, Iterable]) -> Union[str, Iterable[str]]:
        if type(s) == str:
            return f"\\textbf{{{s}}}"
        elif isinstance(s, Iterable):
            return [f"\\textbf{{{label}}}" for label in s]
        return s
    
    @classmethod
    def escape(cls, s: str) -> str:
        for esc in cls.ESCAPES:
            s = s.replace(esc, f"\\{esc}")
        return s
    
    @classmethod
    def escapes(cls, s: Union[str, Iterable[str]]) -> Union[str, list]:
        if type(s) == str:
            return cls.escape(s)
        elif isinstance(s, Iterable):
            return [cls.escape(ss) for ss in s]
        raise RuntimeError(f"Argument s is of invalid type({type(s)}), supported types are ('str', 'Iterable')")
    
    @classmethod
    def span_row_label(cls, label: str, n_cols: int, bold=False, row_line_top=True, row_line_bottom=True, outer_col_lines=True) -> str:
        col_fmt = cls.col_format(1, col_lines=False, outer_col_lines=outer_col_lines)
        row_line_top = f"{SPACES_L3}{ROW_LINE}\n" if row_line_top else ""
        row_line_bottom = ROW_LINE_NL if row_line_bottom else ""

        if bold:
            label = self.bold(label)
        return f"{row_line_top}{SPACES_L3}\\multicolumn{{{n_cols}}}{{{col_fmt}}}{{{label}}} \\\\ {row_line_bottom}"
    
    def tabular_rows_with_label_rows(self, data: pd.DataFrame, n_cols: int, label_rows: Iterable[RowLabel], n_round: int = None, row_lines=True, outer_row_lines=True, outer_col_lines=False, add_row_labels=False) -> List[str]:
        rows = []

        if len(label_rows) and (label_rows[0].start_idx != 0):
            subset = data.iloc[0:label_rows[0].start_idx, :]
            rows.extend(
                self.tabular_rows(subset, n_round, row_lines, False, add_row_labels)
            )

        for lr in label_rows:
            if lr.end_idx is not None:
                rows.append(self.span_row_label(lr.label, n_cols, lr.bold, row_line_top=outer_row_lines, row_line_bottom=False, outer_col_lines=outer_col_lines))
                subset = data.iloc[lr.start_idx:lr.end_idx, :]
                rows.extend(
                    self.tabular_rows(subset, n_round, row_lines, False, add_row_labels)
                )
            else:
                rows.append(self.span_row_label(lr.label, n_cols, lr.bold, row_line_top=outer_row_lines, row_line_bottom=False, outer_col_lines=outer_col_lines))
                subset = data.iloc[lr.start_idx:, :]
                rows.extend(
                    self.tabular_rows(subset, n_round, row_lines, outer_row_lines, add_row_labels)
                )
        return rows
    
    @classmethod
    def rounder(cls, v: Any, n_round: int = None) -> Union[int, float, str]:
        if n_round is None:
            return v
        elif n_round == 0:
            return int(round(v, n_round))
        else:
            return round(v, n_round)
    
    def cell(self, row, col, v: Any, n_round = None) -> str:
        if isinstance(v, numbers.Number):
            v = str(self.rounder(v, n_round))
        else:
            v = self.escapes(v)
        return self.apply_cell_post_processing(row, col, v)
    
    def tabular_rows(self, data: pd.DataFrame, n_round: int = None, row_lines=True, outer_row_lines=True, add_row_labels=False) -> List[str]:
        table_rows = []
        row_line = ROW_LINE_NL if row_lines else ""

        if add_row_labels:
            row_labels = [f"{self.escapes(label)} & " for label in data.index]
        else:
            row_labels = ["" for _ in data.index]

        last_row = len(data.index) - 1
        for i, (index, row) in enumerate(data.iterrows()):
            row_data = [self.cell(index, col, cell, n_round) for col, cell in enumerate(row)]

            if (i == last_row) and not outer_row_lines:
                table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\")
            elif (i == last_row) and outer_row_lines and not row_lines:
                table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {ROW_LINE_NL}")
            else:
                table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {row_line}")
        return table_rows



class LatexTable(Latex):
    def __init__(self):
        super().__init__()

    def header(self, column_labels: List[str], bold=True, row_lines=True, outer_row_lines=True, add_row_label: str = None):
        if bold:
            labels = self.bold(column_labels)
            add_row_label = self.bold(add_row_label)
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
    
    def create(
        self, 
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
        outer_col_lines=True) -> str: 

        if caption is None:
            caption = str(caption)

        n_cols = len(data.columns) if add_row_label is None else len(data.columns) + 1    
        
        header = self.header(data.columns, bold_header, row_lines, outer_col_lines, add_row_label)
        rows = self.tabular_rows(data, round, row_lines, outer_row_lines, add_row_label is not None)
        tabular = self.tabular_skel(header, rows, n_cols, col_lines, outer_col_lines)
        table = self.table_skel(tabular, caption, label)
        return table

class LatexMulticolTable(Latex):

    def __init__(self, rest_space_idx: int = None):
        super().__init__()
        self.rest_space_idx = rest_space_idx
        self.data: Dict[str, dict] = {}

    def header(self, labels: List[str], sub_labels: List[str], bold_labels=True, bold_sub_labels=False, row_lines=True, outer_row_lines=True, col_lines=True, outer_col_lines=True) -> str:
        labels = self.escapes(labels)
        sub_labels = self.escapes(sub_labels)
        
        if bold_labels:
            labels = self.bold(labels)
        if bold_sub_labels:
            sub_labels = self.bold(sub_labels)

        if self.rest_space_idx is not None:
            cline = f"\n{SPACES_L3}\\cline{{{len(sub_labels)}-{len(labels) * len(sub_labels) - len(sub_labels) + 1}}}\n"
        else:
            cline = ROW_LINE_NL

        row_line = "\\hline\n" if row_lines else "\n"
        parts = []

        for i, label in enumerate(labels):
            if self.rest_space_idx is not None and (i == self.rest_space_idx):
                parts.append(f"\\multirow{{{len(sub_labels)}}}{{*}}{{{label}}}")
            elif (i == len(labels) - 1):
                parts.append(f"\\multicolumn{{{len(sub_labels)}}}{{{self.col_format(1, col_lines, False)}}}{{{label}}}")
            else:
                parts.append(f"\\multicolumn{{{len(sub_labels)}}}{{{self.col_format(1, col_lines, False, True)}}}{{{label}}}")
        
        label_header = f"{SPACES_L3}\\hline\n" + SPACES_L3 + " & ".join(parts) + f" \\\\{cline}"
        parts.clear()

        for i, _ in enumerate(labels):
            if self.rest_space_idx is not None and (i == self.rest_space_idx):
                parts.append(f"& ")
            else:
                parts.extend(" & ".join(sub_labels))
                if i == (len(labels) - 1):
                    parts.append(" \\\\")
                else:
                    parts.append(f" & ")
        
        label_header += SPACES_L3 + "".join(parts) + row_line
        return label_header
    
    def _encode_label(self, label: str, sub_label: str = None) -> str:
        if sub_label is None:
            return f"rest_space_{label}"
        else:
            return f"sublabel_{label}_{sub_label}"
    
    def _decode_label(self, label: str) -> Tuple[str, Optional[str]]:
        if label.startswith("rest_space_"):
            return label[len("rest_space_"):], None
        elif label.startswith("sublabel_"):
            label = label[len("sublabel_"):]
            
            label_end = label.find("_")
            assert label_end != -1

            decoded_label = label[:label_end]
            decoded_sub_label = label[label_end + 1:]
            return decoded_label, decoded_sub_label
    
    def _decode_labels(self, labels: List[str]) -> Tuple[List[str], List[str]]:
        _labels = {}
        _sub_labels = {}
        for i, label in enumerate(labels):
            decoded, decoded_sub = self._decode_label(label)
            if _labels.get(decoded) is None:
                _labels[decoded] = i
            if decoded_sub is not None and (_sub_labels.get(decoded_sub) is None):
                _sub_labels[decoded_sub] = i

        return list(_labels.keys()), list(_sub_labels.keys())
        
    
    def add_cell(self, index, label: str, sub_label: str = None, data = None):
        mapped_label = self._encode_label(label, sub_label)
        if mapped_label not in self.data:
            self.data[mapped_label] = {index: data}
        else:
            self.data[mapped_label][index] = data
    
    def create(
        self,
        label: str,
        caption: str = None,
        add_column_labels=True,
        n_round: int = None, 
        bold_header=True, 
        bold_sub_header=False,
        row_lines=True, 
        outer_row_lines=True,
        col_lines=True,
        outer_col_lines=True,
        row_labels: Iterable[RowLabel] = None,
        vertical=False) -> str:

        if caption is None:
            caption = str(caption)

        data = pd.DataFrame.from_dict(self.data, orient='columns')

        labels, sub_labels = self._decode_labels(data.columns)
        print(labels)
        print(sub_labels)

        header = self.header(
            labels, sub_labels, bold_header, 
            bold_sub_header, row_lines, outer_row_lines, col_lines, outer_col_lines
        )

        if row_labels is None:
            rows = self.tabular_rows(data, n_round, row_lines, outer_row_lines, False)
        else:
            rows = self.tabular_rows_with_label_rows(data, len(data.columns), row_labels, n_round, row_lines, outer_row_lines, outer_col_lines, False)

        tabular = self.tabular_skel(header, rows, len(data.columns), col_lines, outer_col_lines)
        table = self.table_skel(tabular, caption, label, vertical)
        return table

def create_task_filter_fn(task: Task):
    return lambda folder: folder.dataset.info().task in task



def create_test_results_stats_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label='test_result_stats') -> str:
    reg_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=["NOSearch"],
        filter_fn=create_task_filter_fn(Task.REGRESSION), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    cls_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=["NOSearch"],
        filter_fn=create_task_filter_fn(Task.BINARY | Task.MULTICLASS), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    
    ltx = LatexMulticolTable(rest_space_idx=0)
    sub_labels = ["mean", "std", "min", "max"]
    row_labels = []
    
    row_start = 0
    n_rows = 0
    for domain, data in dict(Classification=cls_data, Regression=reg_data).items():
        for i, (dataset, results) in enumerate(data.results.items()):
            index = dataset.lower()
            # FIXME: Use index instead of own column as in normal table!
            ltx.add_cell(index, "Dataset", data=dataset.lower())

            for (method, result) in results.items():
                for sub_label in sub_labels:
                    ltx.add_cell(index, method, sub_label, data=result["result"][f"{sub_label}_test_acc"])
    
            n_rows += 1

        row_labels.append(RowLabel(row_start, domain, False, n_rows))
        row_start = n_rows

        print(row_labels)
    
    return ltx.create(label, None, n_round=4, row_lines=False, outer_col_lines=False, row_labels=row_labels, vertical=True)



def create_train_test_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label="test_result_stats") -> str:
    reg_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets,
        filter_fn=create_task_filter_fn(Task.REGRESSION), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    cls_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets,
        filter_fn=create_task_filter_fn(Task.BINARY | Task.MULTICLASS), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    
    ltx = LatexMulticolTable(rest_space_idx=0)
    sub_labels = ["train", "test"]
    row_labels = []

    row_start = 0
    n_rows = 0
    for domain, data in dict(Classification=cls_data, Regression=reg_data).items():
        for i, (dataset, results) in enumerate(data.results.items()):
            index = dataset.lower()
            # FIXME: Use index instead of own column as in normal table!
            ltx.add_cell(index, "Dataset", data=dataset.lower())

            for (method, result) in results.items():
                #print(f"{method}: train={result["result"]["mean_train_acc"]}, test={result["result"]["mean_test_acc"]}")
                for sub_label in sub_labels:
                    ltx.add_cell(index, method, sub_label, data=result["result"][f"mean_{sub_label}_acc"])
    
            n_rows += 1

        row_labels.append(RowLabel(row_start, domain, False, n_rows))
        row_start = n_rows
        print(row_labels)

    return ltx.create(label, None, n_round=4, row_lines=False, outer_col_lines=True, row_labels=row_labels)



def create_time_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, label="test_result_stats") -> str:
    data = calc_eval_metrics(w_nas=0.5, filter_fn=filter_fn, sort_fn=sort_fn, reverse=True, print_results=False)
    deltas = time_frame_deltas(data)

    # use the delta values to order to columns
    sort_deltas = [col for col in deltas.columns if col.endswith("_delta")]
    sort_deltas = deltas[sort_deltas].sum(axis=0)
    sort_deltas = sort_deltas.sort_values()
    sort_deltas = list(sort_deltas.index)
    sort_deltas.reverse()
    sort_cols = [col.rstrip("_delta") for col in sort_deltas]

    sorted_columns = list(itertools.chain.from_iterable(zip(sort_cols, sort_deltas)))
    sorted_columns.append("NOSearch")

    deltas = deltas[sorted_columns]
    deltas = deltas.astype(int)

    def stamp(secs: int) -> str:
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return "{:02}:{:02}".format(h, m)


    delta_t = "$\\Delta t$"
    ltx = LatexMulticolTable(rest_space_idx=0)
    
    for (dataset, results) in deltas.iterrows():
        index = dataset.lower()
        # FIXME: Use index instead of own column as in normal table!
        ltx.add_cell(index, "Dataset", data=dataset.lower())

        for (method, secs) in results.items():
            if method.endswith("_delta"):
                _label, sub_label = method.rstrip("_delta"), delta_t
            else:
                _label, sub_label = method, "time"

            ltx.add_cell(index, _label, sub_label, data=stamp(secs))
    
    return ltx.create(label, None, n_round=0, row_lines=False, outer_col_lines=False)



def create_ns_rank_table(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True, label='ns_ranks') -> str:
    data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=["NOSearch"], sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    ltx = LatexMulticolTable(rest_space_idx=0)

    for i, (dataset, results) in enumerate(data.results.items()):
        index = dataset.lower()
        # FIXME: Use index instead of own column as in normal table!
        ltx.add_cell(index, "Dataset", data=dataset.lower())

        for (method, result) in results.items():
            ltx.add_cell(index, method, "ns", data=data.normalized_scores.at[method, dataset])
            ltx.add_cell(index, method, "rank", data=data.mean_ranks.at[dataset, method])

    return  ltx.create(label, None, n_round=4, row_lines=False, outer_col_lines=True)



def create_method_metrics_table(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True, label='basline_metric') -> str:
    data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=["NOSearch"], sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    latex_dict = {
        f"\\(a_s\\)": data.agg_scores.to_dict(),
        f"\\(a_r\\)":data.rank_scores.to_dict(),
        f"\\(\\mu_s\\)": data.nas.to_dict(),
        f"\\(\\mu_r\\)": data.nrs.to_dict(),
        f"\\(\\lambda\\)": data.js.to_dict()
    }

    ltx = LatexTable()
    frame = pd.DataFrame.from_dict(latex_dict)
    return ltx.create(frame, label, None, round=4, add_row_label="Dataset", row_lines=False, outer_col_lines=True)


def save_table(table: str):
    with open("table.txt", mode='w') as f:
        f.write(table)

if __name__ == "__main__":
    ignore_datasets = tuple()

    folder_sorter = lambda folder: ( 
        folder.dataset.info().task in (Task.BINARY, Task.MULTICLASS), 
        folder.dataset.info().task == Task.REGRESSION,
        folder.dataset.info().size_group == SizeGroup.SMALL,
        folder.dataset.info().size_group == SizeGroup.MODERATE,
        folder.dataset.info().size_group == SizeGroup.LARGE,
        folder.dataset.name, 
        folder.search_method
    )

    #friedman_check(ignore_datasets, folder_sorter, sort_reverse=True)
    table = create_time_table(ignore_datasets, sort_fn=folder_sorter)
    #print(table)
    save_table(table)








