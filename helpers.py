import pandas as pd
import streamlit as st

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
    confusion_matrix = confusion_matrix.set_index("classes")
    confusion_matrix.index.name = None
    return confusion_matrix


def get_mean_confusion_matrix(confusion_matrix: list[pd.DataFrame]):
    if len(confusion_matrix) == 1:
        return format_confusion_matrix(confusion_matrix[0])
    confusion_matrix = pd.concat([pd.DataFrame(conf)
                                 for conf in confusion_matrix])
    mean = confusion_matrix.groupby("classes").mean().round(
        ROUND_DECIMAL_PLACES).astype(str)
    std = confusion_matrix.groupby("classes").std().round(
        ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data.index.name = None
    return data


def format_table(table: pd.DataFrame):
    INTEGER_COLUMNS = {"rules count", "total conditions count",
                       "Covered by prediction", "Not covered by prediction"}
    table = table.T
    columns = set(table.columns)
    integer_columns = list(INTEGER_COLUMNS.intersection(columns))
    float_columns = list(columns - INTEGER_COLUMNS)
    table[integer_columns] = table[integer_columns].map(lambda x: f"{x:.0f}")
    table[float_columns] = table[float_columns].map(lambda x: f"{x:.3f}")
    return table.T


def toggle_generation():
    st.session_state.generation = True
    st.session_state.generated_rules = pd.Series(name="Rules")
    st.session_state.statistics = []
    st.session_state.indicators = []
    st.session_state.confusion_matrices = []
