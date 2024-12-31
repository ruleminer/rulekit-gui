import pandas as pd
import streamlit as st

ROUND_DECIMAL_PLACES = 3


def get_mean_table(data: list[dict]) -> pd.DataFrame:
    if len(data) == 1:
        data = _format_table(data[0])
        data = pd.DataFrame(data)
        return data
    data = pd.DataFrame(data)
    mean = data.mean().round(ROUND_DECIMAL_PLACES).astype(str)
    std = data.std().round(ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data = data.replace("nan ± nan", "-")
    data.name = ""
    data = pd.DataFrame(data)
    return data


def get_mean_confusion_matrix(confusion_matrix: list[dict]):
    if len(confusion_matrix) == 1:
        confusion_matrix = pd.DataFrame(
            confusion_matrix[0]).set_index("classes")
        confusion_matrix.index.name = None
        return confusion_matrix
    confusion_matrix = pd.concat([pd.DataFrame(conf)
                                 for conf in confusion_matrix])
    mean = confusion_matrix.groupby("classes").mean().round(
        ROUND_DECIMAL_PLACES).astype(str)
    std = confusion_matrix.groupby("classes").std().round(
        ROUND_DECIMAL_PLACES).astype(str)
    data = mean + " ± " + std
    data.index.name = None
    return data


def _format_table(table: dict):
    table = pd.DataFrame(table, index=[""])
    float_types = table.select_dtypes(['float']).columns
    table[float_types] = table[float_types].map(lambda x: f"{x:.3f}")
    return table.astype(str).T


def toggle_generation():
    st.session_state.generation = True
    st.session_state.generated_rules = pd.Series(name="Rules")
    st.session_state.statistics = []
    st.session_state.indicators = []
    st.session_state.confusion_matrices = []
