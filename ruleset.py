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
    AgGrid(df, fit_columns_on_grid_load=True)


def display_survival_ruleset(ruleset):
    df = create_ruleset_df(ruleset)
    display_df = df.drop(columns=["plot"])
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    grid_options = gb.build()
    data = AgGrid(display_df,
                  gridOptions=grid_options,
                  allow_unsafe_jscode=True,
                  update_mode=GridUpdateMode.SELECTION_CHANGED,
                  fit_columns_on_grid_load=True,
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
            _format_conclusion(rule.conclusion.value) if isinstance(
                ruleset, SurvivalRuleSet) else str(rule.conclusion),
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
        return f"{conclusion:.2f}"
    else:
        return conclusion
