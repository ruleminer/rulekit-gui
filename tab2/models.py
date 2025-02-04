import streamlit as st
from rulekit.classification import ExpertRuleClassifier
from rulekit.classification import RuleClassifier
from rulekit.regression import ExpertRuleRegressor
from rulekit.regression import RuleRegressor
from rulekit.survival import ExpertSurvivalRules
from rulekit.survival import SurvivalRules

from common.choices import ModelType
from common.expert_params import define_fit_expert_params
from common.expert_params import get_common_expert_params
from tab2.induction_params import get_classification_params
from tab2.induction_params import get_common_params
from tab2.induction_params import get_regression_params
from tab2.metrics import get_measures_selection_dict


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
    st.text("")
    on_expert = st.toggle(
        "Do you want to perform expert induction?", value=False)

    if not on_expert:
        clf = RuleClassifier(
            **metric,
            **param,
            **class_param,
        )
    else:
        st.write("Expert induction parameters")
        expert_params = get_common_expert_params()
        expert_params["consider_other_classes"] = st.toggle(
            "Induce rules for the remaining classes", value=False)
        define_fit_expert_params()
        clf = ExpertRuleClassifier(
            **metric,
            **param,
            **class_param,
            **expert_params,
        )

    return clf, metric["induction_measure"], on_expert


def _define_model_regression():
    metric = get_measures_selection_dict()
    param = get_common_params()
    reg_param = get_regression_params()
    st.write("")
    on_expert = st.toggle(
        "Do you want to perform expert induction?", value=False)

    if not on_expert:
        clf = RuleRegressor(
            **metric,
            **param,
            **reg_param,
        )
    else:
        st.write("Expert induction parameters")
        expert_params = get_common_expert_params()
        define_fit_expert_params()
        clf = ExpertRuleRegressor(
            **metric,
            **param,
            **reg_param,
            **expert_params,
        )

    return clf, metric["induction_measure"], on_expert


def _define_model_survival():
    param = get_common_params()
    st.text("")
    on_expert = st.toggle(
        "Do you want to perform expert induction?", value=False)

    if not on_expert:
        clf = SurvivalRules(
            survival_time_attr="survival_time",
            **param,
        )
    else:
        st.write("Expert induction parameters")
        expert_params = get_common_expert_params()
        define_fit_expert_params()
        clf = ExpertSurvivalRules(
            survival_time_attr="survival_time",
            **param,
            **expert_params,
        )

    return clf, None, on_expert
