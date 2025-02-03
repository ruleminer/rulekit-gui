import streamlit as st
from decision_rules.ruleset_factories import ruleset_factory

from common.expert_params import parse_expert_params_to_fit
from common.session import commit_current_model_settings
from common.session import reset_results_in_session
from tab3.dataset import process_data
from tab3.helpers import make_splits
from tab3.listener import MyProgressListener
from tab3.statistics import calculate_iteration_results


def train_and_evaluate_all(
        data, model_type, eval_type, div_type, per_div, n_fold, clf, on_expert,
):
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
            if on_expert:
                clf.fit(x_train, y_train, **parse_expert_params_to_fit())
            else:
                clf.fit(x_train, y_train)
        except:
            st.error(
                "An error occurred during model training. Make sure the parameters are correct.")
            st.stop()
        listener.finish()
        if clf.model.rules:
            ruleset = ruleset_factory(clf, x_train, y_train)
            rulesets.append(ruleset)
            calculate_iteration_results(
                ruleset, x_train, y_train, x_test, y_test)

    st.session_state.generation = False
    st.session_state.ruleset = rulesets[0]
