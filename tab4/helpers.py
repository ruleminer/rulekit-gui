import pandas as pd

from common.helpers import format_table

ROUND_DECIMAL_PLACES = 3


def get_mean_table(data: list[pd.DataFrame]) -> pd.DataFrame:
    if len(data) == 1:
        return format_table(data[0])
    data = pd.concat(data, axis=1)
    mean = data.mean(1).round(ROUND_DECIMAL_PLACES).astype(str)
    std = data.std(1).round(ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data = data.replace("nan ± nan", "-")
    data.name = ""
    data = pd.DataFrame(data)
    return data


def format_confusion_matrix(confusion_matrix: pd.DataFrame):
    return confusion_matrix


def get_mean_confusion_matrix(confusion_matrix: list[pd.DataFrame]):
    if len(confusion_matrix) == 1:
        return format_confusion_matrix(confusion_matrix[0])
    confusion_matrix = pd.concat([pd.DataFrame(conf)
                                 for conf in confusion_matrix])
    confusion_matrix = confusion_matrix.reset_index().groupby(
        ["level_0", "level_1"])
    mean = confusion_matrix.mean().round(ROUND_DECIMAL_PLACES).astype(str)
    std = confusion_matrix.std().round(ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data.index.names = [None, None]
    return data
