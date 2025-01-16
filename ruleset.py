import re

import numpy as np
import pandas as pd
import streamlit as st
from decision_rules.survival import SurvivalRuleSet
from matplotlib import pyplot as plt
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder
from st_aggrid import GridUpdateMode

plt.ioff()


def display_ruleset(ruleset):
    df = create_ruleset_df(ruleset)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(wrapText=True, autoHeight=True)
    gb.configure_column("Rule premise", width=360)
    gb.configure_column("Conclusion", width=120)
    gb.configure_column("p", width=60)
    gb.configure_column("n", width=60)
    gb.configure_column("P", width=60)
    gb.configure_column("N", width=60)
    AgGrid(
        df,
        enable_enterprise_modules=False,
        gridOptions=gb.build(),
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        height=_get_height(df),
    )


def display_survival_ruleset(ruleset):
    df = create_ruleset_df(ruleset)
    display_df = df.drop(columns=["plot"])
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_default_column(wrapText=True, autoHeight=True)
    gb.configure_column("Rule premise", width=350)
    gb.configure_column("Median survival time", width=150)
    gb.configure_column("p", width=50)
    gb.configure_column("P", width=50)
    data = AgGrid(
        display_df,
        enable_enterprise_modules=False,
        gridOptions=gb.build(),
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=_get_height(df),
    )
    selected_rows = data["selected_rows"]
    if selected_rows is not None and len(selected_rows) != 0:
        st.write("Kaplan-Meier plot for the selected rule:")
        fig = df.iloc[selected_rows.index]["plot"].values[0]
        st.pyplot(fig)
    else:
        st.write("Select a rule to display its corresponding Kaplan-Meier plot.")


def plot_kaplan_meier(rule):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(rule.conclusion.estimator.times,
            rule.conclusion.estimator.probabilities, "black")
    props = dict(boxstyle="round,pad=1", facecolor="grey", alpha=0.2)
    text_x_coord = (rule.conclusion.estimator.times.max(
    ) - rule.conclusion.estimator.times.min()) * 0.55 + rule.conclusion.estimator.times.min()
    rule_str = get_survival_rule_string(rule).replace(
        "AND ", "AND\n").replace("THEN", "\nTHEN").replace("(p", "\n(p")
    ax.text(text_x_coord, 0.95, rule_str,
            wrap=True, bbox=props, fontsize=12,
            verticalalignment="top")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Survival probability", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, rule.conclusion.estimator.times.max())
    ax.margins(x=0)
    return fig


def get_survival_rule_string(rule):
    rule_str = str(rule).replace(
        rule.conclusion.column_name, "median survival time"
    )
    return re.sub(", [n,N]=\d*", "", rule_str)


def create_ruleset_df(ruleset):
    rows = [
        [
            rule.premise.to_string(rule.column_names),
            _format_conclusion(rule.conclusion.value),
            rule.coverage.p,
            rule.coverage.n,
            rule.coverage.P,
            rule.coverage.N,
        ]
        for rule in ruleset.rules
    ]
    df = pd.DataFrame(rows, columns=[
        "Rule premise",
        "Conclusion",
        "p",
        "n",
        "P",
        "N",
    ], index=range(len(rows)))
    if isinstance(ruleset, SurvivalRuleSet):
        df["plot"] = [plot_kaplan_meier(rule) for rule in ruleset.rules]
        df = df[["Rule premise", "Conclusion", "p", "P", "plot"]]
        df = df.rename(columns={"Conclusion": "Median survival time"})
    return df


def _format_conclusion(conclusion):
    if conclusion == np.inf:
        return "inf"
    elif conclusion == -np.inf:
        return "-inf"
    elif isinstance(conclusion, float):
        return f"{conclusion:.3f}"
    else:
        return conclusion


def _get_height(df):
    row_heights = df["Rule premise"].apply(
        lambda x: len(x.replace(" ", "")) // 40 + 1).sum()
    height = min(int(row_heights) * 30, 500)
    return height
