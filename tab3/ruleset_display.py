import re

import numpy as np
import pandas as pd
import streamlit as st
from decision_rules.survival import SurvivalRuleSet
from matplotlib import pyplot as plt
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder
from st_aggrid import GridUpdateMode

from tab3.kaplan_meier import get_kaplan_meier
from tab3.kaplan_meier import plot_kaplan_meier

plt.ioff()


def display_ruleset():
    ruleset = st.session_state.ruleset
    if isinstance(ruleset, SurvivalRuleSet):
        _display_survival_ruleset(ruleset)
    else:
        _display_ruleset(ruleset)


def _display_ruleset(ruleset):
    df = _create_ruleset_df(ruleset)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(wrapText=True, autoHeight=True)
    gb.configure_column("ID", width=40)
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
    display_df = df.drop(columns=["KM"])
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_default_column(wrapText=True, autoHeight=True)
    gb.configure_column("ID", width=50)
    gb.configure_column("Rule premise", width=300)
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
        selection = selected_rows.index.astype(int)
        st.write("Kaplan-Meier plot for selected rules")
        plot_kaplan_meier(df.iloc[selection].sort_values("ID"))
    else:
        st.write("Select a rule to display its corresponding Kaplan-Meier plot.")


def _get_survival_rule_string(rule):
    rule_str = str(rule).replace(
        rule.conclusion.column_name, "median survival time"
    )
    return re.sub(", [n,N]=[0-9]*", "", rule_str)


def _create_ruleset_df(ruleset):
    rows = [
        [
            f"r{i+1}",
            rule.premise.to_string(rule.column_names),
            _format_conclusion(rule.conclusion.value),
            rule.coverage.p,
            rule.coverage.n,
            rule.coverage.P,
            rule.coverage.N,
        ]
        for i, rule in enumerate(ruleset.rules)
    ]
    df = pd.DataFrame(rows, columns=[
        "ID",
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
        df["KM"] = [get_kaplan_meier(rule, data[coverage_matrix[:, i]])
                    for i, rule in enumerate(ruleset.rules)]
        df = df[["ID", "Rule premise", "Conclusion", "p", "P", "KM"]]
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
        lambda x: len(x) // 50 + 1).sum()
    height = min(32 + (int(row_heights)) * 30, 480)
    return height
