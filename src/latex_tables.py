import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Iterable, Optional
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct, time_frame_stamps, sort_folders, friedman_check
from calc_metrics import time_frame_deltas
from Util import Task, SizeGroup
import numbers
from dataclasses import dataclass, field
from itertools import chain
import itertools
import math

@dataclass(frozen=True, eq=True)
class RowLabel:
   label: str
   start_idx: int
   end_idx: int = None
   processing: List[str] = field(default_factory=[])
   
   def bold(self) -> RowLabel:
      self.processing.append("bold")
      return self
    
   def underline(self) -> RowLabel:
      self.processing.append("underline")
      return self


SPACES_L1 = "   "
SPACES_L2 = SPACES_L1 * 2
SPACES_L3 = SPACES_L1 * 3

ROW_LINE = "\\hline"
ROW_LINE_NL = f"\n{SPACES_L3}{ROW_LINE}"



class Latex:
    ESCAPES = ["_", "%", "&"]

    def __init__(
        self, 
        label: str = "tab:", 
        caption: str = "",
        add_column_labels=True, 
        add_row_label: str = None, 
        n_round: int = None, 
        bold_header=True, 
        row_lines=True, 
        outer_row_lines=True, 
        col_lines=True, 
        outer_col_lines=True,
        vertical=False):

        self.add_column_labels = add_column_labels
        self.add_row_label = self.process_header_labels(add_row_label)
        self.n_round = n_round
        self.bold_header = bold_header
        self.row_lines = row_lines
        self.col_lines = col_lines
        self.outer_row_lines = outer_row_lines
        self.outer_col_lines = outer_col_lines
        self.caption = caption
        self.label = label
        self.vertical = vertical
    
    def process_header_labels(self, labels: Union[None, str, List[str]], override_bold=False) -> Union[str, List[str]]:
        if labels is None:
            return
        elif type(labels) == str:
            labels = (labels, )
        elif not len(labels):
            return labels

        labels = self.escapes(s)
        if self.bold_header or override_bold:
            labels = self.bold(labels)
        
        return labels if len(labels) > 1 else labels[0]

    def table_skel(self, tabular: str) -> str:
        resize_to = "\\linewidth" if self.vertical else "\\textwidth"

        table = (
            f"\\begin{{table}}[H]\n"
            f"{SPACES_L1}\\centering\n"
            f"{SPACES_L1}\\resizebox{{{resize_to}}}{{!}}{{\n"
            f"{tabular}\n"
            f"{SPACES_L1}}}\n"
            f"{SPACES_L1}\\caption{{{self.caption}}}\n"
            f"{SPACES_L1}\\label{{tab:{self.table_label}}}\n"
            f"\\end{{table}}"
        ) 

        if self.vertical:
            table = (
                f"\\begin{{landscape}}\n" # requires \usepackage{lscape}
                f"\\vspace*{{\\fill}}"
                f"{table}"
                f"\\vspace*{{\\fill}}"
                f"\\end{{landscape}}\n"
            )

        return table

    def col_format(self, n_cols: int) -> str:
        if self.col_lines and self.outer_col_lines:
            col_format = "|" + ("c|" * n_cols)
        elif (not self.col_lines) and (not self.outer_col_lines):
            col_format = "c" * n_cols
        elif self.col_lines and (not self.outer_col_lines):
            col_format = "|".join('c' * n_cols) 
        elif (not self.col_lines) and self.outer_col_lines:
            col_format = f"|{'c' * n_cols}|" 
        return col_format
    
    def tabular_skel(self, header: str, rows: List[str], n_cols: int) -> str:
        newline = '\n'
        return (
            f"{SPACES_L2}\\begin{{tabular}}{{{self.col_format(n_cols)}}}\n"
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

    def multirow(self, n_rows: int, label: str) -> str:
        return f"\\multirow{{{n_rows}}}{{*}}{{{label}}}"

    def multicolumn(self, n_cols: int, label: str) -> str:
        col_fmt = self.col_format(1)
        return f"\\multicolumn{{{n_cols}}}{{{col_fmt}}}{{{label}}}"
    
    @classmethod
    def cline(cls, start_col: int, end_col: int) -> str:
        return f"\\cline{{{start_col}-{end_col}}}"
    
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
    
    def span_row_label(self, rl: RowLabel, n_cols: int) -> str:
        row_line_top = f"{SPACES_L3}{ROW_LINE}\n"
        row_line_bottom = ROW_LINE_NL

        label = rl.label
        for process in rl.processing: 
            label = getattr(self, process, lambda l: l)(label)

        return f"{row_line_top}{SPACES_L3}{self.multicolumn(n_cols, label)} \\\\ {row_line_bottom}"
    
    def tabular_rows_with_label_rows(self, data: pd.DataFrame, n_cols: int, label_rows: Iterable[RowLabel]) -> List[str]:
        rows = []
        if len(label_rows) and (label_rows[0].start_idx != 0):
            subset = data.iloc[0:label_rows[0].start_idx, :]
            rows.extend(self.tabular_rows(subset))

        for lr in label_rows:
            if lr.end_idx is not None:
                subset = data.iloc[lr.start_idx:lr.end_idx, :]
            else:
                subset = data.iloc[lr.start_idx:, :]
            rows.append(self.span_row_label(lr, n_cols))
            rows.extend(self.tabular_rows(subset))
        return rows
    
    def rounder(self, v: Any) -> Union[int, float, str]:
        if self.n_round is None:
            return v
        elif self.n_round == 0:
            return int(round(v, self.n_round))
        else:
            return round(v, self.n_round)
    
    def cell(self, row, col, v: Any) -> str:
        if isinstance(v, numbers.Number):
            v = str(self.rounder(v, self.n_round))
        else:
            v = self.escapes(v)
        return self.apply_cell_post_processing(row, col, v)
    
    def tabular_rows(self, data: pd.DataFrame) -> List[str]:
        table_rows = []
        row_line = ROW_LINE_NL if row_lines else ""

        if self.add_row_label is not None:
            row_labels = [f"{self.escapes(label)} & " for label in data.index]
        else:
            row_labels = ["" for _ in data.index]

        last_row = len(data.index) - 1
        for i, (index, row) in enumerate(data.iterrows()):
            row_data = [self.cell(index, col, cell) for col, cell in enumerate(row)]

            if (i == last_row) and not self.outer_row_lines:
                table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\")
            elif (i == last_row) and self.outer_row_lines and (not self.row_lines):
                table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {ROW_LINE_NL}")
            else:
                table_rows.append(SPACES_L3 + row_labels[i] + " & ".join(row_data) + f" \\\\ {row_line}")
        return table_rows



class LatexTable(Latex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def header(self, column_labels: List[str]):
        column_labels = self.process_header_labels(column_labels)
        self.add_row_label = self.bold(add_row_label)

        outer_line = ROW_LINE_NL if self.outer_row_lines else ""
        inner_line = ROW_LINE_NL if self.row_lines else "\n"

        row_label = f"{SPACES_L3}" if add_row_label is None else f"{outer_line}{SPACES_L3}{add_row_label} & "
        header = row_label + " & ".join(labels) + f" \\\\ {inner_line}"
        
        return header
    
    def create(self, data: pd.DataFrame, row_labels: Iterable[RowLabel] = None) -> str: 
        n_cols = len(data.columns) if self.add_row_label is None else len(data.columns) + 1    
        header = self.header(data.columns)
        
        rows = self.tabular_rows(data)
        if row_labels is None:
            rows = self.tabular_rows(data)
        else:
            rows = self.tabular_rows_with_label_rows(data, len(data.columns), row_labels)

        tabular = self.tabular_skel(header, rows, n_cols)
        table = self.table_skel(tabular)
        return table


class LatexMulticolTable(Latex):

    def __init__(self, bold_sub_labels=False, sub_header_top_line=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bold_sub_labels = bold_sub_labels
        self.sub_header_top_line = sub_header_top_line
        self.data: Dict[str, dict] = {}
    
    def calc_n_cols(self, decoded_labels: Dict[str, List[str]]) -> int:
        n_cols = 1 if self.add_row_label is not None else 0
        for (label, sub_labels) in decoded_labels.items():
            if len(sub_labels):
                n_cols += len(sub_labels)
            else:
                n_cols += 1
        return n_cols

    def header(self, decoded_labels: Dict[str, List[str]], n_cols: int) -> str:
        outer_row_line = ROW_LINE_NL if self.outer_row_lines else ""
        inner_row_line = f" \\\\{ROW_LINE_NL}" if self.row_lines else "\\\\\n"
        
        column_holes = []
        header = []
        sub_header = []

        if self.add_row_label:
            header.append(self.multirow(2, self.add_row_label))
            column_holes.append(0 + 1)

        for i, (label, sub_labels) in enumerate(decoded_labels.items()):
            if not len(sub_labels):
                header.append(self.multirow(2, self.process_header_labels(label)))
                column_holes.append(i + 1)
            else:
                header.append(self.multicolumn(len(sub_labels), label))
                sub_header.extend(
                    self.process_header_labels(sub_labels, override_bold=self.bold_sub_labels)
                )

        if len(column_holes) and self.sub_header_top_line:
            if len(column_holes) == 1 and column_holes[0] == 1:
                cline = self.cline(1, n_cols)
            elif len(column_holes) == 1 and column_holes[0] != 1:
                cline = self.cline(1, column_holes[0]) + self.cline(column_holes[0]+1, n_cols)
            else:
                cline = self.cline(1, n_cols)
            cline = f" \\\\\n{SPACES_L3}{cline}"
        else:
            cline = inner_row_line
        
        label_header = f"{outer_row_line}\n{' &\n'.join(header)}\n{SPACES_L3}{cline}{' & '.join(sub_header)}{inner_row_line}"
        return label_header
    
    def _encode_label(self, label: str, sub_label: str = None) -> str:
        if sub_label is None:
            return label
        else:
            return f"sublabel_{label}_{sub_label}"
    
    def _decode_label(self, label: str) -> Tuple[str, Optional[str]]:
        if label.startswith("sublabel_"):
            label = label[len("sublabel_"):]
            label_end = label.find("_")

            decoded_label = label[:label_end]
            decoded_sub_label = label[label_end + 1:]
            return decoded_label, decoded_sub_label
        return label, None
    
    def _decode_labels(self, encoded: List[str]) -> Dict[str, List[str]]:
        labels = {}
        for (label, sublabel) in (self._decode_label(l) for l in encoded):
            sub_labels = labels.get(label)
            if sub_labels is None:
                labels[label] = []
            elif sub_label is not None:
                sub_labels.append(sublabel)
            
        return labels
        
    
    def add_cell(self, index, label: str, sub_label: str = None, data = None):
        mapped_label = self._encode_label(label, sub_label)
        row = self.data.get(mapped_label, None)

        if row is None:
            self.data[mapped_label] = {index: data}        
        else:
            row[index] = data
    
    def create(self, row_labels: Iterable[RowLabel] = None) -> str:
        data = pd.DataFrame.from_dict(self.data, orient='columns')
        decoded_labels = self._decode_labels(data.columns)
        n_cols = self.calc_n_cols(decoded_labels)  

        print(decoded_labels)

        header = self.header(decoded_labels, n_cols)
        if row_labels is None:
            rows = self.tabular_rows(data)
        else:
            rows = self.tabular_rows_with_label_rows(data, n_cols, row_labels)

        tabular = self.tabular_skel(header, rows, n_cols)
        table = self.table_skel(tabular)
        return table









def create_task_filter_fn(task: Task):
    return lambda folder: folder.dataset.info().task in task

def create_test_results_stats_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True) -> str:
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
    
    ltx = LatexMulticolTable(n_round=4, row_lines=False, outer_col_lines=False, vertical=True)

    sub_labels = ["mean", "std", "min", "max"]
    row_labels = []
    
    row_start = 0
    n_rows = 0
    for domain, data in dict(Classification=cls_data, Regression=reg_data).items():
        for i, (dataset, results) in enumerate(data.results.items()):
            for (method, result) in results.items():
                for sub_label in sub_labels:
                    ltx.add_cell(index=dataset.lower(), method, sub_label, data=result["result"][f"{sub_label}_test_acc"])
            n_rows += 1

        row_labels.append(RowLabel(row_start, domain, False, n_rows))
        row_start = n_rows
        print(row_labels)
    return ltx.create(row_labels=row_labels)



def create_train_test_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True) -> str:
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
    
    ltx = LatexMulticolTable(n_round=4, row_lines=False, outer_col_lines=True, )
    sub_labels = ["train", "test"]
    row_labels = []

    row_start = 0
    n_rows = 0
    for domain, data in dict(Classification=cls_data, Regression=reg_data).items():
        for i, (dataset, results) in enumerate(data.results.items()):
            for (method, result) in results.items():
                #print(f"{method}: train={result["result"]["mean_train_acc"]}, test={result["result"]["mean_test_acc"]}")
                for sub_label in sub_labels:
                    ltx.add_cell(index=dataset.lower(), method, sub_label, data=result["result"][f"mean_{sub_label}_acc"])
            n_rows += 1

        row_labels.append(RowLabel(row_start, domain, False, n_rows))
        row_start = n_rows
        print(row_labels)
    return ltx.create(row_labels=row_labels)



def create_time_table(ignore_datasets: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True) -> str:
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
    ltx = LatexMulticolTable(n_round=0, row_lines=False, outer_col_lines=False)
    
    for (dataset, results) in deltas.iterrows():
        for (method, secs) in results.items():
            if method.endswith("_delta"):
                _label, sub_label = method.rstrip("_delta"), delta_t
            else:
                _label, sub_label = method, "time"
            ltx.add_cell(index=dataset.lower(), _label, sub_label, data=stamp(secs))
    
    return ltx.create()



def create_ns_rank_table(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True) -> str:
    data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=["NOSearch"], sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=False
    )
    ltx = LatexMulticolTable(n_round=4, row_lines=False, outer_col_lines=True)

    for i, (dataset, results) in enumerate(data.results.items()):
        for (method, result) in results.items():
            ltx.add_cell(index=dataset.lower(), method, "ns", data=data.normalized_scores.at[method, dataset])
            ltx.add_cell(index=dataset.lower(), method, "rank", data=data.mean_ranks.at[dataset, method])
    return  ltx.create()



def create_method_metrics_table(ignore_datasets: List[str] = None, sort_fn=None, sort_reverse=True) -> str:
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

    ltx = LatexTable(n_round=4, add_row_label="Dataset", row_lines=False, outer_col_lines=True)
    frame = pd.DataFrame.from_dict(latex_dict)
    return ltx.create(frame)


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








