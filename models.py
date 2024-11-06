import streamlit as st
from rulekit.classification import ExpertRuleClassifier
from rulekit.classification import RuleClassifier
from rulekit.regression import ExpertRuleRegressor
from rulekit.regression import RuleRegressor
from rulekit.survival import ExpertSurvivalRules
from rulekit.survival import SurvivalRules

from choices import ModelType
from expert_params import define_fit_expert_params
from expert_params import get_common_expert_params
from induction_params import get_classification_params
from induction_params import get_common_params
from induction_params import get_regression_params
from metrics import get_measures_selection_dict


def define_model(model_type: ModelType):
    if model_type == ModelType.CLASSIFICATION:
        return _define_model_classification()
    elif model_type == ModelType.REGRESSION:
        return _define_model_regression()
    elif model_type == ModelType.SURVIVAL:
        return _define_model_survival()


def _define_model_classification():
    metric = get_measures_selection_dict()
    param = get_common_params()
    class_param = get_classification_params()

    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.expert_rules_list = []
        clf = RuleClassifier(
            **metric,
            **param,
            **class_param,
        )
    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_expert_params()
        expert_params["consider_other_classes"] = st.toggle(
            "Induce rules for the remaining classes", value=False)
        clf = ExpertRuleClassifier(
            **metric,
            **param,
            **class_param,
            **expert_params,
        )
        define_fit_expert_params()

    return clf, metric["induction_measure"], on_expert


def _define_model_regression():
    metric = get_measures_selection_dict()
    param = get_common_params()
    reg_param = get_regression_params()

    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.expert_rules_list = []
        clf = RuleRegressor(
            **metric,
            **param,
            **reg_param,
        )
    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_expert_params()
        clf = ExpertRuleRegressor(
            **metric,
            **param,
            **reg_param,
            **expert_params,
        )
        define_fit_expert_params()

    return clf, metric["induction_measure"], on_expert


def _define_model_survival():
    param = get_common_params()

    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.expert_rules_list = []
        clf = SurvivalRules(
            survival_time_attr='survival_time',
            **param,
        )
    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_expert_params()
        clf = ExpertSurvivalRules(
            survival_time_attr='survival_time',
            **param,
            **expert_params,
        )
        define_fit_expert_params()

    return clf, None, on_expert
