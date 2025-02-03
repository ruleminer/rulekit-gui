import streamlit as st

from common.choices import EvaluationType
from tab4.helpers import get_mean_confusion_matrix
from tab4.helpers import get_mean_table


def display_results():
    st.subheader("Ruleset statistics")
    model_type = st.session_state.settings["model_type"]
    is_train_test = st.session_state.settings["eval_type"] != EvaluationType.ONLY_TRAINING

    ruleset_stats = get_mean_table(st.session_state.statistics)
    if not is_train_test:
        ruleset_stats = ruleset_stats.drop(
            "test", level=0).reset_index(0, drop=True)
    else:
        ruleset_stats = ruleset_stats.unstack().T.reset_index(0, drop=True)[
            ["train", "test"]]
    st.write(ruleset_stats.to_html(
        header=is_train_test), unsafe_allow_html=True)

    if model_type and st.session_state.confusion_matrices:
        st.subheader("Confusion matrix")
        confusion_matrix = get_mean_confusion_matrix(
            st.session_state.confusion_matrices)
        if not is_train_test:
            confusion_matrix = confusion_matrix.drop(
                "test", level=0).reset_index(0, drop=True)
        else:
            confusion_matrix = confusion_matrix.T[["train", "test"]]
        st.write(confusion_matrix.to_html(),
                 unsafe_allow_html=True)

    st.subheader("Prediction indicators")
    prediction_indicators = get_mean_table(st.session_state.indicators)
    if not is_train_test:
        prediction_indicators = prediction_indicators.drop(
            "test", level=0).reset_index(0, drop=True)
    else:
        prediction_indicators = prediction_indicators.unstack(
        ).T.reset_index(0, drop=True)[["train", "test"]]
    st.write(prediction_indicators.to_html(
        header=is_train_test), unsafe_allow_html=True)
