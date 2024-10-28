import streamlit as st
from rulekit.classification import ExpertRuleClassifier
from rulekit.classification import RuleClassifier
from rulekit.regression import ExpertRuleRegressor
from rulekit.regression import RuleRegressor
from rulekit.survival import ExpertSurvivalRules
from rulekit.survival import SurvivalRules

from expert_params import define_expert_preferred_extend
from expert_params import define_expert_preferred_induction
from induction_params import get_classification_params
from induction_params import get_common_params
from induction_params import get_common_params_expert
from induction_params import get_regression_params
from metrics import get_metrics_selection_dict


def define_model_classification():
    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    metric = get_metrics_selection_dict()
    param = get_common_params()
    class_param = get_classification_params()

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.ind_exp_list = []

        clf = RuleClassifier(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            control_apriori_precision=class_param["control_apriori_precision"],
            approximate_bins_count=class_param["approximate_bins_count"],
            approximate_induction=class_param["approximate_induction"])

    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_params_expert()

        clf = ExpertRuleClassifier(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            control_apriori_precision=class_param["control_apriori_precision"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            approximate_bins_count=class_param["approximate_bins_count"],
            approximate_induction=class_param["approximate_induction"],
            extend_using_preferred=expert_params["extend_using_preferred"],
            extend_using_automatic=expert_params["extend_using_automatic"],
            induce_using_preferred=expert_params["induce_using_preferred"],
            induce_using_automatic=expert_params["induce_using_automatic"],
            consider_other_classes=st.toggle(
                "Consider other classes", value=False),
            preferred_conditions_per_rule=expert_params["preferred_conditions_per_rule"],
            preferred_attributes_per_rule=expert_params["preferred_attributes_per_rule"]
        )

        _update_expert_params_in_session(expert_params)

    return clf, metric["ind_cluss"], on_expert


def define_model_regression():

    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    metric = get_metrics_selection_dict()
    param = get_common_params()
    reg_param = get_regression_params()

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.ind_exp_list = []

        clf = RuleRegressor(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            mean_based_regression=reg_param["mean_based_regression"])

    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_params_expert()

        clf = ExpertRuleRegressor(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            mean_based_regression=reg_param["mean_based_regression"],
            extend_using_preferred=expert_params["extend_using_preferred"],
            extend_using_automatic=expert_params["extend_using_automatic"],
            induce_using_preferred=expert_params["induce_using_preferred"],
            induce_using_automatic=expert_params["induce_using_automatic"],
            preferred_conditions_per_rule=expert_params["preferred_conditions_per_rule"],
            preferred_attributes_per_rule=expert_params["preferred_attributes_per_rule"]
        )

        _update_expert_params_in_session(expert_params)

    return clf, metric["ind_cluss"], on_expert


def define_model_survival():
    on_expert = st.toggle(
        'Do you want to perform expert induction?', value=False)

    param = get_common_params()

    if not on_expert:
        st.session_state.pref_list = []
        st.session_state.forb_list = []
        st.session_state.ind_exp_list = []

        clf = SurvivalRules(
            survival_time_attr='survival_time',
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"])

    else:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = get_common_params_expert()

        clf = ExpertSurvivalRules(
            survival_time_attr='survival_time',
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            extend_using_preferred=expert_params["extend_using_preferred"],
            extend_using_automatic=expert_params["extend_using_automatic"],
            induce_using_preferred=expert_params["induce_using_preferred"],
            induce_using_automatic=expert_params["induce_using_automatic"],
            preferred_conditions_per_rule=expert_params["preferred_conditions_per_rule"],
            preferred_attributes_per_rule=expert_params["preferred_attributes_per_rule"]
        )

        _update_expert_params_in_session(expert_params)

    return clf, on_expert


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
