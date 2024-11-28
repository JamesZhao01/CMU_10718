import json
import re
import os

from collections.abc import Iterable

from typing import Callable

from utils.data.data_module import DataModule
from utils.data.testbench import TestBench


def rank_results_by_predicate(
    paths: str,
    j: int,
    criterion: Callable[[], int],
    asc: bool = True,
    read_metric_file_instead: bool = False,
):
    if type(paths) == str:
        paths = [paths]
    assert isinstance(paths, Iterable), "Paths must be an iterable"
    if read_metric_file_instead:
        data = [(path, read_all_metrics(path)) for path in paths]
    else:
        data = [(path, read_jth_line_from_log_file(path, j)) for path in paths]
    data = [
        (path, idx, hyperparameters, lines, line, criterion(line))
        for (path, folder_data) in data
        for (idx, hyperparameters, lines, line) in folder_data
    ]
    data.sort(key=lambda x: x[-1], reverse=not asc)
    return data


def read_jth_line_from_log_file(folder_path: str, j: int):
    for idx, hyperparameters, lines in read_all_logs_in_folder(folder_path):
        if j < len(lines):
            yield idx, hyperparameters, lines, lines[j]
        else:
            raise ValueError(f"{j=} exceeds line count {len(lines)}")


def read_all_logs_in_folder(folder_path: str):

    pattern = r"model_(\d)+\.pt\.log"
    for filename in os.listdir(folder_path):
        if match := re.match(pattern, filename):
            idx = match.group(1)
            with open(os.path.join(folder_path, filename), "r") as f:
                log_file_text = f.read()
                hyperparameters, line = process_log_file(log_file_text)
                yield idx, hyperparameters, line


def read_all_metrics(folder_path: str):
    pattern = r"model_(\d)+\.pt\.metrics\.log"
    for filename in os.listdir(folder_path):
        if match := re.match(pattern, filename):
            idx = match.group(1)
            with open(os.path.join(folder_path, filename), "r") as f:
                metrics_text = f.read()
                yield idx, {}, [], json.loads(metrics_text)


def process_log_file(log_file_text: str):
    log_file_text = log_file_text.strip().split(">>")

    # Hyperparameters
    header = "(hyperparameters)"
    hyperparameter_str = log_file_text[0][len(header) :]
    hyperparameters = json.loads(hyperparameter_str)

    out = []
    for text in log_file_text[1:]:
        text = text.rstrip("")
        if text == "(Fin)":
            break
        line = text.split("|")
        assert len(line) == 3, "Log file sector doesn't have 3 pieces"
        epoch = int(line[0])
        loss = float(line[1])
        # make sure to drop the "<<" on the end
        data = json.loads(line[2][: line[2].find("<<")])
        data |= {"epoch": epoch, "loss": loss}

        out.append(data)
    return hyperparameters, out
