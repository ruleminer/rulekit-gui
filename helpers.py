import pandas as pd

ROUND_DECIMAL_PLACES = 3


def get_mean_table(data: list[dict]) -> pd.Series:
    data = pd.DataFrame(data)
    mean = data.mean().round(ROUND_DECIMAL_PLACES).astype(str)
    std = data.std().round(ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data = data.replace("nan ± nan", "-")
    data.name = ""
    return data


def get_mean_confusion_matrix(confusion_matrix: list[dict]):
    confusion_matrix = pd.concat([pd.DataFrame(conf)
                                 for conf in confusion_matrix])
    mean = confusion_matrix.groupby("classes").mean().round(
        ROUND_DECIMAL_PLACES).astype(str)
    std = confusion_matrix.groupby("classes").std().round(
        ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data.index.name = "classes"
    return data


def format_table(table: dict):
    table = pd.DataFrame(table, index=[""])
    float_types = table.select_dtypes(['float']).columns
    table[float_types] = table[float_types].map(lambda x: f"{x:.3f}")
    return table.astype(str).T
