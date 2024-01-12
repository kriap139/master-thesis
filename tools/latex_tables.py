import pandas as pd
from typing import List

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

def latex_table_skel(tabular: str, caption: str, label: str):
    f"""
        \\begin{{table}}[H]
            \\centering
            \\resizebox{{\\textwidth}}{{!}}{{
                {tabular}
            }}
            \\caption{{{caption}}}
            \\label{{tab:{label}}}
        \\end{table} 
    """

def latex_tabular_skel(header: str, rows: List[str], n_cols: int, col_lines=True, outer_col_lines=True):
    if col_lines and outer_col_lines:
        col_format = "|" + ("c|" * n_cols)
    elif col_lines and not outer_col_lines:
        col_format =  "c|" * (n_cols - 1) + 'c'
    else:
        col_format = "c" * n_cols

    f"""
        \\begin{{tabular}}{{{col_format}}}
            {header}
            {''.join(rows)}
        \\end{{tabular}}  
    """

def latex_tabular_header(column_labels: List[str], bold=True, row_lines=True, outer_row_lines=True):
    if bold:
        labels = [f"\\textbf{{{label}}}" for label in column_labels]
    else:
        labels = column_labels
    
    header = " &".join(labels) + " \\"
    return " \\hline\n" + header + " \\hline" if row_lines else header

def latex_tabular_multicol_header(labels: List[str], sub_labels: List[str], bold_labels=True, bold_sub_labels=False, row_lines=True, outer_row_lines=True, rest_space_idx: int = None):
    if bold_labels:
        labels = [f"\\textbf{{{label}}}" for label in labels]
    if bold_sub_labels:
        sub_labels = [f"\\textbf{{{label}}}" for label in sub_labels]
    row_line = "\\hline" if row_lines else ""
    
    parts = []
    for i, label in enumerate(labels):
        if rest_space_idx is not None and (i == rest_space_idx):
            parts.append(f"\\multirow{{{len(sub_labels)}}}{{*}}{{{label}}}")
    
    label_header = "\\hline\n" " &".join(parts) + " \\\\\n"

    parts.clear()
    for i, label in enumerate(labels):
        if rest_space_idx is not None and (i == rest_space_idx):
            parts.append(f"& ")
        else:
            for j, sub_label in enumerate(sub_labels):
                if j == (len(sub_labels) - 1):
                    parts.append(f"{sub_label} \\\\")
                else:
                    parts.append(f"{sub_label} &")
    
    label_header += "".join(parts) + row_line
    
    
def latex_tabular_rows(data: pd.DataFrame, add_row_labels=True, round: int =None, row_lines=True, outer_row_lines=True) -> List[str]:
    table_rows = []
    rounder = lambda v: v if round is None else lambda v: round(v, round)
    row_line = "\\hline" if row_lines else ""

    if add_row_labels:
        row_labels = [f"{label} & " for label in data.index]
    else:
        row_labels = ["" for _ in range(len(data.index))]

    n_rows = len(data.index)
    for i, row in enumerate(data.index):
        row_data = [str(rounder(cell)) for cell in data[row]]

        if (i == n_rows - 1) and not outer_row_lines:
            table_rows.append(row_labels[i] + " &".join(row_data) + f"\\\\")
        else:
            table_rows.append(row_labels[i] + " &".join(row_data) + f"\\\\{row_line}")
    return table_rows

def create_latex_table(data: pd.DataFrame, add_row_labels=True, add_column_labels=True, round: int =None, row_lines=True, outer_row_lines=True):
    pass
    

    








