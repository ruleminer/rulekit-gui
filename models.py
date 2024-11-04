import streamlit as st
from rulekit.classification import ExpertRuleClassifier
from rulekit.classification import RuleClassifier
from rulekit.regression import ExpertRuleRegressor
from rulekit.regression import RuleRegressor
from rulekit.survival import ExpertSurvivalRules
from rulekit.survival import SurvivalRules

from choices import ModelType
from expert_params import define_expert_preferred_extend
from expert_params import define_expert_preferred_induction
from induction_params import get_classification_params
from induction_params import get_common_params
from induction_params import get_common_params_expert
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
        st.session_state.ind_exp_list = []

        clf = RuleClassifier(
            **metric,
            **param,
            **class_param,
        )

    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_params_expert()

        clf = ExpertRuleClassifier(
            **metric,
            **param,
            **class_param,
            consider_other_classes=st.toggle(
                "Consider other classes", value=False),
        )
        _update_expert_params_in_session(expert_params)

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
        st.session_state.ind_exp_list = []

        clf = RuleRegressor(
            **metric,
            **param,
            **reg_param,
        )

    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_params_expert()

        clf = ExpertRuleRegressor(
            **metric,
            **param,
            **reg_param,
            **expert_params,
        )

        _update_expert_params_in_session(expert_params)

    return clf, metric["induction_measure"], on_expert


def _define_model_survival():
    param = get_common_params()

    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.ind_exp_list = []

        clf = SurvivalRules(
            survival_time_attr='survival_time',
            **param,
        )

    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_params_expert()

        clf = ExpertSurvivalRules(
            survival_time_attr='survival_time',
            **param,
            **expert_params,
        )

        _update_expert_params_in_session(expert_params)

    return clf, None, on_expert


def _update_expert_params_in_session(expert_params):
    if expert_params["extend_using_preferred"]:
        st.session_state.pref_list, st.session_state.forb_list = define_expert_preferred_extend(
            st.session_state.pref_list, st.session_state.forb_list)
    else:
        st.session_state.pref_list = []
        st.session_state.forb_list = []

    if expert_params["induce_using_preferred"]:
        st.session_state.ind_exp_list = define_expert_preferred_induction(
            st.session_state.ind_exp_list)
    else:
        st.session_state.ind_exp_list = []
