import pandas as pd
import streamlit as st
from decision_rules.ruleset_factories import ruleset_factory
from rulekit._operator import BaseOperator

from common.choices import DivType
from common.choices import EvaluationType
from common.choices import ModelType
from common.session import commit_current_model_settings
from common.session import reset_results_in_session
from tab3.dataset import process_data
from tab3.helpers import make_splits
from tab3.listener import MyProgressListener
from tab3.statistics import calculate_iteration_results


def train_and_evaluate_all(
        data: pd.DataFrame,
        model_type: ModelType,
        eval_type: EvaluationType,
        div_type: DivType,
        per_div: float,
        n_fold: int,
        clf: BaseOperator,
        fit_expert_params: dict,
):
    """
    Main method for training and evaluating the ruleset.
    The dataset is divided into independent and dependent variables
    and then split according to training settings (only training, train-test split or k-fold cross-validation).
    The model is trained on each split and the ruleset is created.
    Only in the case of k-fold cross-validation, there is in fact more than one iteration, but this structure allows
    for consistency between settings.
    The first ruleset (the only one in only training and train-test split settings, and the one trained on full data in CV case)
    is then stored in the session state for later display.
    """
    commit_current_model_settings(
        model_type=model_type,
        eval_type=eval_type,
        div_type=div_type,
        per_div=per_div,
        n_fold=n_fold,
    )
    reset_results_in_session()
    x, y = process_data(data, model_type)
    st.session_state.x, st.session_state.y = x, y
    listener = MyProgressListener(
        eval_type, n_fold)
    clf.add_event_listener(listener)
    make_splits(x, y, eval_type, div_type, per_div, n_fold)
    rulesets = []
    for (x_train, y_train), (x_test, y_test) in zip(st.session_state.train, st.session_state.test):
        try:
            if fit_expert_params:
                clf.fit(x_train, y_train, **fit_expert_params)
            else:
                clf.fit(x_train, y_train)
        except:
            st.error(
                "An error occurred during model training. Make sure the parameters are correct.")
            st.session_state.generation = False
            st.session_state.ruleset = None
            st.stop()
        listener.finish()
        if clf.model is not None and clf.model.rules:
            try:
                ruleset = ruleset_factory(clf, x_train, y_train)
                rulesets.append(ruleset)
                calculate_iteration_results(
                    ruleset, x_train, y_train, x_test, y_test)
            except:
                st.error(
                    "An unexpected error occurred during ruleset processing. "
                    "If you are performing expert induction, make sure you set expert rules, as well as preferred and forbidden conditions/attributes in the correct format.")
                st.session_state.generation = False
                st.session_state.ruleset = None
                st.stop()

    st.session_state.generation = False
    if rulesets:
        st.session_state.ruleset = rulesets[0]
    else:
        st.session_state.ruleset = None
        st.error("An empty ruleset was generated. Try changing the model parameters.")
    listener.empty()
