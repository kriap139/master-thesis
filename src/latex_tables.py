import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Iterable, Optional
from calc_metrics import calc_eval_metrics, load_result_folders, Builtin, EvalMetrics, BaseSearch, time_frame_pct, time_frame_stamps, sort_folders, friedman_check, KSearchOptuna
from calc_metrics import time_frame_deltas
from Util import Task, SizeGroup, load_csv, load_json
import numbers
from dataclasses import dataclass, field
from itertools import chain
import itertools
import math
import os

@dataclass(frozen=True, eq=True)
class RowLabel:
   label: str
   start_idx: int
   end_idx: int = None
   color: str = None
   processing: List[str] = field(default_factory=list)
   
   def bold(self) -> 'RowLabel':
      self.processing.append("bold")
      return self
    
   def underline(self) -> 'RowLabel':
      self.processing.append("underline")
      return self


class Latex:
    ESCAPES = ["_", "%", "&"]
    SPACES_L1 = "   "
    ROW_LINE = "\\hline"
    NR = " \\\\ "
    SEP = ' & '

    def __init__(
        self, 
        label: str = None, 
        caption: str = "", 
        add_row_label: str = None, 
        n_round: int = None, 
        header_style=None, 
        row_lines=True, 
        outer_row_lines=True, 
        col_lines=True, 
        outer_col_lines=True,
        vertical=False):

        self.n_round = n_round
        self.row_lines = row_lines
        self.col_lines = col_lines
        self.outer_row_lines = outer_row_lines
        self.outer_col_lines = outer_col_lines
        self.caption = caption
        self.label = label
        self.vertical = vertical
        self.header_style = header_style
        self.add_row_label = self.process_header_labels(add_row_label)
    
    def table_space(self) -> str:
        return self.SPACES_L1

    def tabular_space(self) -> str:
        return self.table_space() + self.SPACES_L1

    def row_space(self) -> str:
        return self.tabular_space() + self.SPACES_L1
    
    def row_line(self) -> str:
        return f"\n{self.row_space()}{self.ROW_LINE}"
    
    def parse_styles(self, style: str) -> List[str]:
        if ',' in style:
            return style.split(',')
        else:
            return [style]
    
    def apply_styles(self, s: Union[str, pd.Series, Iterable[str]], styles: List[str]) -> Union[str, pd.Series, Iterable[str]]:
        styler = lambda s: s
        
        for style in styles:
            styler_x = getattr(self, style, None)
            if styler_tmp is not None:
                styler = lambda s: styler_x(styler(s))
        
        if type(s) == str:
            return styler(s)
        elif type(s) == pd.Series:
            print(f"Series type: {s.dtype}")
        else: 
            return [styler(_s) for _s in s]
    
    def process_header_labels(self, labels: Union[None, str, List[str]], override_style=None) -> Union[str, List[str]]:
        return_str = False

        if (labels is None) or (not len(labels)):
            return labels
        elif type(labels) == str:
            return_str = True
            labels = (labels, )

        labels = self.escapes(labels)

        if override_style is not None:
            styles = self.parse_styles(override_style)
        elif self.header_style is not None:
            styles = self.parse_styles(self.header_style)
        else:
            return labels if len(labels) > 1 else labels[0]

        labels = self.apply_styles(labels, styles)
        
        return labels[0] if return_str else labels

    def table_skel(self, tabular: str) -> str:
        resize_to = "\\linewidth" if self.vertical else "\\textwidth"

        table = (
            f"\\begin{{table}}[H]\n"
            f"{self.table_space()}\\centering\n"
            f"{self.table_space()}\\resizebox{{{resize_to}}}{{!}}{{\n"
            f"{tabular}\n"
            f"{self.table_space()}}}\n"
            f"{self.table_space()}\\caption{{{self.caption}}}\n"
            f"{self.table_space()}\\label{{tab:{self.label}}}\n"
            f"\\end{{table}}"
        ) 

        if self.vertical:
            table = (
                f"\\begin{{landscape}}\n" # requires \usepackage{lscape}
                f"\\vspace*{{\\fill}}\n"
                f"{table}"
                f"\n\\vspace*{{\\fill}}\n"
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
        return (
            f"{self.tabular_space()}\\begin{{tabular}}{{{self.col_format(n_cols)}}}\n"
            f"{header}"
            f"{'\n'.join(rows)}\n"
            f"{self.tabular_space()}\\end{{tabular}}"
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

    def rowcolor(self, color: str) -> str:
        return f"\\rowcolor{{{color}}}"
    
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
        row_line_top = f"{self.row_space()}{self.ROW_LINE}"
        row_line_bottom = self.row_line()

        label = rl.label
        for process in rl.processing: 
            label = getattr(self, process, lambda l: l)(label)

        if rl.color is not None:
            color = f"\n{self.row_space()}{self.rowcolor(rl.color)}\n{self.row_space()}"
        else:
            color = self.row_line()

        return f"{row_line_top}{color}{self.multicolumn(n_cols, label)}{self.NR}{row_line_bottom}"
    
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
            v = str(self.rounder(v))
        else:
            v = self.escapes(v)
        return v
    
    def tabular_rows(self, data: pd.DataFrame) -> List[str]:
        table_rows = []
        row_line = self.row_line() if self.row_lines else ""

        if self.add_row_label is not None:
            row_labels = [f"{self.escapes(label)}{self.SEP}" for label in data.index]
        else:
            row_labels = ["" for _ in data.index]

        for i, (index, row) in enumerate(data.iterrows()):
            row_data = [self.cell(index, col, cell) for col, cell in enumerate(row)]
            table_rows.append(f"{self.row_space()}{row_labels[i]}{self.SEP.join(row_data)}{self.NR}{row_line}")
        
        if self.outer_row_lines:
            row_labels.append(self.row_line())
            
        return table_rows

class LatexTable(Latex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def header(self, column_labels: List[str]):
        column_labels = self.process_header_labels(column_labels)
        self.add_row_label = self.bold(self.add_row_label)

        outer_line = self.row_line() if self.outer_row_lines else ""
        inner_line = self.row_line() if self.row_lines else "\n"

        row_label = f"{self.row_space()}" if self.add_row_label is None else f"{outer_line}{self.row_space()}{self.add_row_label}{self.SEP}"
        header = row_label + self.SEP.join(column_labels) + f"{self.NR}{inner_line}"
        
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

    def __init__(self, sub_label_style: str = None, sub_header_top_line=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_label_style = sub_label_style
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

    def header(self, labels: List[str], sub_labels: Dict[str, List[str]], n_cols: int) -> str:
        spaces = self.row_space()
        outer_row_line = f"{spaces}{self.ROW_LINE}" if self.outer_row_lines else ""
        inner_row_line = f"{self.NR}{self.row_line()}" if self.row_lines else f"{self.NR}\n"
        qoute = '& ' if self.add_row_label else ''

        column_holes = []
        header = []
        sub_header = []

        if self.add_row_label:
            header.append(self.multirow(2, self.add_row_label))
            column_holes.append(0 + 1)

        for i, label in enumerate(labels):
            _sub_labels = sub_labels[label]
            if not len(_sub_labels):
                header.append(self.multirow(2, self.process_header_labels(label)))
                column_holes.append(i + 1)
            else:
                header.append(self.multicolumn(len(_sub_labels), label))
                _sub_labels = self.process_header_labels(_sub_labels, override_style=self.sub_label_style)
                sub_header.extend(_sub_labels)
        
        if len(column_holes) and self.sub_header_top_line:
            if len(column_holes) == 1 and column_holes[0] == 1:
                cline = self.cline(2, n_cols)
            elif len(column_holes) == 1 and column_holes[0] != 1:
                cline = self.cline(1, column_holes[0]) + self.cline(column_holes[0]+1, n_cols)
            else:
                cline = self.cline(1, n_cols)

            cline = f"{self.NR}\n{spaces}{cline}\n"
        else:
            cline = inner_row_line
        
        label_header = f"{outer_row_line}\n{spaces}{f'{self.SEP}\n{spaces}'.join(header)} {cline}{spaces}{qoute}{self.SEP.join(sub_header)}{inner_row_line}"
        return label_header
    
    def _encode_label(self, label: str, sub_label: str = None) -> str:
        if sub_label is None:
            return label
        else:
            return f"sublabel_{label}_{sub_label}"
    
    def _encode_sub_labels(self, label: str, sub_labels: List[str]) -> List[str]:
        return [self._encode_label(label, sub_label) for sub_label in sub_labels]
    
    def _decode_label(self, label: str, remove_none=False) -> Tuple[str, Optional[str]]:
        if label.startswith("sublabel_"):
            label = label[len("sublabel_"):]
            label_end = label.find("_")

            decoded_label = label[:label_end]
            decoded_sub_label = label[label_end + 1:]
            return decoded_label, decoded_sub_label
        return label if remove_none else (label, None) 
    
    def _decode_labels(self, encoded: List[str]) -> Tuple[List[str], Dict[str, str]]:
        decoded = tuple(self._decode_label(l) for l in encoded)
        _labels = map(lambda tup: tup[0], decoded)

        labels = []
        for label in _labels:
            if label not in labels:
                labels.append(label) 

        sub_labels = {label: [] for label in labels}
        for (label, sub_label) in decoded:
            if sub_label is not None:
                sub_labels[label].append(sub_label)
            
        return labels, sub_labels
        
    def add_cell(self, index, label: str, sub_label: str = None, data = None):
        mapped_label = self._encode_label(label, sub_label)
        row = self.data.get(mapped_label, None) 
        if row is None:
            self.data[mapped_label] = {index: data}        
        else:
            row[index] = data
    
    def create(self, row_labels: Iterable[RowLabel] = None) -> str:
        data = pd.DataFrame.from_dict(self.data, orient='columns')
        labels, sub_labels = self._decode_labels(data.columns)
        n_cols = self.calc_n_cols(sub_labels)  

        print(f"labels: {labels}")
        print(f"sub_labels: {sub_labels.values()}")

        header = self.header(labels, sub_labels, n_cols)
        if row_labels is None:
            rows = self.tabular_rows(data)
        else:
            rows = self.tabular_rows_with_label_rows(data, n_cols, row_labels)

        tabular = self.tabular_skel(header, rows, n_cols)
        table = self.table_skel(tabular)
        return table
    
    def sort_columns(self, by_sub_labels: List[str], ascending=True):
        data = pd.DataFrame.from_dict(self.data, orient='columns')
        _, sub_labels = self._decode_labels(data.columns)
        data.columns = pd.MultiIndex.from_tuples(tuple(self._decode_label(label, remove_none=True) for label in data.columns), names=["label", "sub_label"])

        sort_columns = [col for col in data.columns if col[1] in by_sub_labels]
        sort_frame = data[sort_columns]
        sort_frame = sort_frame.T.groupby(level='label', sort=False).sum().sum(axis=1).sort_values(ascending=ascending)
        sort_columns = list(
            chain.from_iterable(self._encode_sub_labels(label, sub_labels[label]) for label in sort_frame.index)
        )

        data.columns = list(self._encode_label(*label) for label in data.columns)
        data = data[sort_columns]

        self.data = data.to_dict(orient='dict')



def create_task_filter_fn(task: Task):
    return lambda folder: folder.dataset.info().task in task

def create_test_results_stats_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, no_search_table=False, print_folders=False) -> str:

    if no_search_table:
        ignore_methods = ["RandomSearch", "SeqUDSearch", "GridSearch", "OptunaSearch", "KSearchOptuna"]
    else:
        if ignore_methods is None:
            ignore_methods = ["NOSearch"]
        else:
            ignore_methods = ignore_methods.copy()
            ignore_methods.append("NOSearch")

    reg_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods,
        filter_fn=create_task_filter_fn(Task.REGRESSION), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )
    cls_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods,
        filter_fn=create_task_filter_fn(Task.BINARY | Task.MULTICLASS), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )

    add_row_label = "Dataset" if not no_search_table else "NOSearch metrics"
    
    ltx = LatexMulticolTable(n_round=4, row_lines=False, outer_col_lines=False, vertical=True, add_row_label=add_row_label)
    sub_labels = ["mean", "std", "min", "max"]
    row_labels = [] if not no_search_table else None
    row_start = 0
    n_rows = 0

    for domain, data in dict(Classification=cls_data, Regression=reg_data).items():
        for i, (dataset, results) in enumerate(data.results.items()):
            for (method, result) in results.items():
                for sub_label in sub_labels:
                    if no_search_table:
                        ltx.add_cell(sub_label, domain, dataset.lower(), data=result["result"][f"{sub_label}_test_acc"])
                    else:
                        ltx.add_cell(dataset.lower(), method, sub_label, data=result["result"][f"{sub_label}_test_acc"])
            n_rows += 1

        if not no_search_table:
            row_labels.append(RowLabel(domain, row_start, n_rows, color='lightgray'))
            row_start = n_rows

    if not no_search_table:
        print(row_labels)
        ltx.sort_columns(by_sub_labels=['max', 'mean'], ascending=False)
    else:
        print(pd.DataFrame(ltx.data))

    return ltx.create(row_labels=row_labels)



def create_train_test_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, print_folders=False) -> str:
    reg_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods,
        filter_fn=create_task_filter_fn(Task.REGRESSION), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )
    cls_data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods,
        filter_fn=create_task_filter_fn(Task.BINARY | Task.MULTICLASS), sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )
    
    ltx = LatexMulticolTable(n_round=4, row_lines=False, outer_col_lines=True, add_row_label='Dataset')
    sub_labels = ["train", "test"]
    row_labels = []

    row_start = 0
    n_rows = 0
    for domain, data in dict(Classification=cls_data, Regression=reg_data).items():
        for i, (dataset, results) in enumerate(data.results.items()):
            for (method, result) in results.items():
                #print(f"{method}: train={result["result"]["mean_train_acc"]}, test={result["result"]["mean_test_acc"]}")
                for sub_label in sub_labels:
                    ltx.add_cell(dataset.lower(), method, sub_label, data=result["result"][f"mean_{sub_label}_acc"])
            n_rows += 1
        
        row_labels.append(RowLabel(domain, row_start, n_rows, 'lightgray'))
        row_start = n_rows
    
    ltx.sort_columns(by_sub_labels=["test"], ascending=False)
    print(row_labels)
    return ltx.create(row_labels=row_labels)



def create_time_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, filter_fn=None, sort_fn=None, sort_reverse=True, print_folders=False) -> str:
    data = calc_eval_metrics(w_nas=0.5, filter_fn=filter_fn, sort_fn=sort_fn, reverse=True, print_results=print_folders, 
    ignore_with_info_filter=ignore_with_info_filter, ignore_methods=ignore_methods, ignore_datasets=ignore_datasets)
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
            ltx.add_cell(dataset.lower(), _label, sub_label, data=stamp(secs))
    
    return ltx.create()



def create_ns_rank_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, sort_fn=None, sort_reverse=True, print_folders=False) -> str:
    if ignore_methods is None:
        ignore_methods = ["NOSearch"]
    else:
        ignore_methods = ignore_methods.copy()
        ignore_methods.append("NOSearch")

    data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods, sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )
    ltx = LatexMulticolTable(n_round=4, row_lines=False, outer_col_lines=True, add_row_label='Dataset')
    p_hat = f"\\(\\hat{{p}}\\)"

    for i, (dataset, results) in enumerate(data.results.items()):
        for (method, result) in results.items():
            ltx.add_cell(dataset.lower(), method, "p", data=data.normalized_scores.at[dataset, method])
            ltx.add_cell(dataset.lower(), method, p_hat, data=data.mean_normalized_scores.at[dataset, method])
    
    ltx.sort_columns(by_sub_labels=['p', p_hat], ascending=True)
    return  ltx.create()



def create_method_metrics_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, sort_fn=None, sort_reverse=True, print_folders=False, normalize_k_search_histories=False) -> str:
    if ignore_methods is None:
        ignore_methods = ["NOSearch"]
    else:
        ignore_methods = ignore_methods.copy()
        ignore_methods.append("NOSearch")

    data = calc_eval_metrics(
        w_nas=0.5, normalize_k_search_histories=normalize_k_search_histories, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods, sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders, 
        ignore_with_info_filter=ignore_with_info_filter
    )

    latex_dict = {
        f"\\(a_{{s}}\\)": data.agg_ns_scores.to_dict(),
        f"\\(a_{{ms}}\\)":data.agg_mns_scors.to_dict(),
        f"\\(\\mu_{{s}}\\)": data.nas.to_dict(),
        f"\\(\\mu_{{ms}}\\)": data.nams.to_dict(),
        f"\\(\\lambda\\)": data.js.to_dict()
    }

    ltx = LatexTable(n_round=4, add_row_label="Dataset", row_lines=False, outer_col_lines=True)
    frame = pd.DataFrame.from_dict(latex_dict)
    frame.sort_values(by="\\(\\lambda\\)", axis=0, ascending=False, inplace=True)

    return ltx.create(frame)

def create_lambda_metric_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, sort_fn=None, sort_reverse=True, print_folders=False) -> str:
    if ignore_methods is None:
        ignore_methods = ["NOSearch"]
    else:
        ignore_methods = ignore_methods.copy()
        ignore_methods.append("NOSearch")

    data = calc_eval_metrics(
        w_nas=0.5, ignore_datasets=ignore_datasets, ignore_methods=ignore_methods, sort_fn=sort_fn, reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )
    latex_dict = {
        f"\\(\\lambda\\)": data.js.to_dict()
    }
    ltx = LatexTable(n_round=4, add_row_label="Metric", row_lines=False, outer_col_lines=True)
    frame = pd.DataFrame.from_dict(latex_dict, orient='index')
    frame.sort_values(by="\\(\\lambda\\)", axis=1, ascending=False, inplace=True)
    return ltx.create(frame)

def create_ksearch_iter_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, sort_fn=None, sort_reverse=True, print_folders=False) -> str:
    metrics = calc_eval_metrics(
        w_nas=0.5, 
        ignore_datasets=ignore_datasets, 
        ignore_methods=ignore_methods, 
        sort_fn=sort_fn, 
        reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )
    names = [method for method in metrics.get_method_names() if method.startswith("KSearch")]
    data = {name: {} for name in names}

    for dataset, methods in metrics.folders.items():
        for method, folder in methods.items():
            if method.startswith("KSearch"):
                history = load_csv(os.path.join(folder.dir_path, "history.csv"))
                data[method][dataset] = history.shape[0]
    
    ltx = LatexTable(n_round=4, add_row_label="Dataset", row_lines=True, outer_row_lines=True, outer_col_lines=True)
    frame = pd.DataFrame.from_dict(data, orient='columns')
    frame.index = [index.lower() for index in frame.index]
    print(frame)
    return ltx.create(frame)

def create_k_space_table(ignore_datasets: List[str] = None, ignore_methods: List[str] = None, sort_fn=None, sort_reverse=True, print_folders=False, n_round=4) -> str:
    metrics = calc_eval_metrics(
        w_nas=0.5, 
        ignore_datasets=ignore_datasets, 
        ignore_methods=ignore_methods, 
        sort_fn=sort_fn, 
        reverse=sort_reverse, 
        print_results=print_folders,
        ignore_with_info_filter=ignore_with_info_filter
    )

    ltx = LatexMulticolTable(n_round=n_round, row_lines=True, outer_col_lines=True, add_row_label='param')

    for dataset, methods in metrics.folders.items():
        for method, folder in methods.items():
            if method.startswith("KSearch"):
                data = KSearchOptuna.recalc_results(folder.dir_path)
                result = data["result"]

                k_params = [param for param in result.keys() if param.startswith("k_")]
                params = [param[len("k_"):] for param in k_params]
                for param in params:
                    ltx.add_cell(param.replace('_', '-'), method, dataset.lower(), data=result['k_' + param])
    
    print(pd.DataFrame.from_dict(ltx.data, orient='columns'))
    return ltx.create()

def save_table(table: str):
    with open("table.txt", mode='w') as f:
        f.write(table)

if __name__ == "__main__":
    #ignore_datasets = ("kdd1998_allcat", "kdd1998_nonum")
    ignore_datasets = ("fps", "acsi", "wave_e", "rcv1", "delays_zurich", "comet_mc", "epsilon", "kdd1998", "kdd1998_allcat", "kdd1998_nonum")
    ignore_methods = ["GridSearch"]
    ignore_with_info_filter = lambda info: info["nparams"] != "4"

    folder_sorter = lambda folder: ( 
        folder.dataset.info().task in (Task.BINARY, Task.MULTICLASS), 
        folder.dataset.info().task == Task.REGRESSION,
        folder.dataset.info().size_group == SizeGroup.SMALL,
        folder.dataset.info().size_group == SizeGroup.MODERATE,
        folder.dataset.info().size_group == SizeGroup.LARGE,
        folder.dataset.name, 
        folder.search_method
    )

    #friedman_check(ignore_datasets=ignore_datasets, ignore_methods=ignore_methods, ignore_with_info_filter=ignore_with_info_filter, print_results=True)
    table = create_ksearch_iter_table(ignore_datasets, ignore_methods, sort_fn=folder_sorter, sort_reverse=True, print_folders=True)
    save_table(table)








