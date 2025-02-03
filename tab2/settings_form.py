import streamlit as st

from common.choices import DivType
from common.choices import EvaluationType
from common.choices import ModelType


def get_training_settings():
    # Model type
    model_type = st.radio(
        "Model type",
        ModelType.choices(),
        index=0,
    )
    st.write("")

    # Evaluation type
    eval_choices = EvaluationType.choices()
    if model_type == ModelType.REGRESSION:
        eval_choices.remove(EvaluationType.CROSS_VALIDATION)
    eval_type = st.radio(
        "Evaluation parameters", eval_choices, index=0)

    n_fold = None
    per_div = None
    div_type = None

    if eval_type == EvaluationType.TRAIN_TEST:
        per_div = st.number_input(
            "Insert a percentage of the test set", value=0.20)
    if eval_type == EvaluationType.CROSS_VALIDATION:
        n_fold = st.number_input("Insert a number of folds", value=5)

    if eval_type in [EvaluationType.TRAIN_TEST, EvaluationType.CROSS_VALIDATION]:
        if model_type == ModelType.CLASSIFICATION:
            div_type = DivType.STRATIFIED
        else:
            div_type = st.radio(
                "Hold out type", DivType.choices(), index=1)

    return model_type, eval_type, div_type, per_div, n_fold
