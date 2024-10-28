import streamlit as st

from const import MEASURE_SELECTION


def get_metrics_selection_dict():
    met_ind_cluss = st.selectbox(
        "Induction measure", MEASURE_SELECTION, index=6)
    ind_cluss = MEASURE_SELECTION.Desc[MEASURE_SELECTION["Metric"]
                                       == met_ind_cluss].values[0]

    met_prun_cluss = st.selectbox(
        "Pruning measure", MEASURE_SELECTION, index=6)
    prun_cluss = MEASURE_SELECTION.Desc[MEASURE_SELECTION["Metric"]
                                        == met_prun_cluss].values[0]

    met_vot_cluss = st.selectbox("Voting measure", MEASURE_SELECTION, index=6)
    vot_cluss = MEASURE_SELECTION.Desc[MEASURE_SELECTION["Metric"]
                                       == met_vot_cluss].values[0]

    dictionary = {"ind_cluss": ind_cluss,
                  "prun_cluss": prun_cluss,
                  "vot_cluss": vot_cluss, }

    return dictionary
