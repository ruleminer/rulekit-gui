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


def display_ruleset():
    ruleset = st.session_state.ruleset
    if isinstance(ruleset, SurvivalRuleSet):
        _display_survival_ruleset(ruleset)
    else:
        _display_other_ruleset(ruleset)


def _display_other_ruleset(ruleset):
    df = _create_ruleset_df(ruleset)
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


def _display_survival_ruleset(ruleset):
    df = _create_ruleset_df(ruleset)
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
        rule_index = int(selected_rows.index.values[0])
        st.write("Kaplan-Meier plot for the selected rule:")
        fig = df.loc[rule_index, "plot"]
        st.pyplot(fig)
        rule = ruleset.rules[rule_index]
        rule_str = _get_survival_rule_string(rule)
        st.write(rule_str)
    else:
        st.write("Select a rule to display its corresponding Kaplan-Meier plot.")


def _plot_kaplan_meier(rule, covered_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    times = rule.conclusion.estimator.times
    probabilities = rule.conclusion.estimator.probabilities
    if times[0] != 0.0:
        times = np.insert(times, 0, 0.0)
        probabilities = np.insert(probabilities, 0, 1.0)
    last_covered_row = covered_data.sort_values("survival_time").iloc[-1]
    if last_covered_row["survival_status"] == 1:
        times = np.insert(times, -1, last_covered_row["survival_time"])
        probabilities = np.insert(probabilities, -1, 0.0)
    ax.step(times,
            probabilities, "black", where="post")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Survival probability", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, times.max() + 1)
    ax.margins(x=0)
    return fig


def _get_survival_rule_string(rule):
    rule_str = str(rule).replace(
        rule.conclusion.column_name, "median survival time"
    )
    return re.sub(", [n,N]=[0-9]*", "", rule_str)


def _create_ruleset_df(ruleset):
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
        x, y = st.session_state.test[0]
        data = pd.concat([x, y], axis=1)
        coverage_matrix = ruleset.calculate_coverage_matrix(x)
        df["plot"] = [_plot_kaplan_meier(rule, data[coverage_matrix[:, i]])
                      for i, rule in enumerate(ruleset.rules)]
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
