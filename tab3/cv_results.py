import pandas as pd
import streamlit as st

from common.helpers import format_table


def display_cross_validation():
    index = [
        f"Fold {i + 1}" for i in range(st.session_state.settings["n_fold"])]
    statistics = [stat.loc["test"] for stat in st.session_state.statistics[1:]]
    ruleset_stats = pd.concat(statistics, axis=1)
    ruleset_stats.columns = index
    st.write("Ruleset statistics")
    st.table(format_table(ruleset_stats))
    st.write("Rules for entire model")
