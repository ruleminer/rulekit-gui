import pandas as pd

ROUND_DECIMAL_PLACES = 2


def get_mean_table(data: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(data)
    mean = data.mean().round(ROUND_DECIMAL_PLACES).astype(str)
    std = data.std().round(ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data = data.replace("nan ± nan", "-")
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
