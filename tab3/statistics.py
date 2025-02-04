from functools import partial

import pandas as pd
import streamlit as st
from decision_rules.classification import ClassificationRuleSet
from decision_rules.classification.prediction_indicators import calculate_for_classification
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.helpers import correct_p_values_fdr
from decision_rules.helpers import get_significant_fraction
from decision_rules.regression import RegressionRuleSet
from decision_rules.regression.prediction_indicators import calculate_for_regression
from decision_rules.survival import SurvivalRuleSet
from decision_rules.survival.prediction_indicators import calculate_for_survival


def calculate_iteration_results(ruleset, x_train, y_train, x_test, y_test):
    train_ruleset_stats = calculate_ruleset_stats(
        ruleset, x_train, y_train)
    test_ruleset_stats = calculate_ruleset_stats(
        ruleset, x_test, y_test)
    train_prediction_indicators = calculate_prediction_indicators(
        ruleset, x_train, y_train)
    test_prediction_indicators = calculate_prediction_indicators(
        ruleset, x_test, y_test)
    if isinstance(ruleset, ClassificationRuleSet):
        train_prediction_indicators, train_confusion_matrix = train_prediction_indicators
        test_prediction_indicators, test_confusion_matrix = test_prediction_indicators
        confusion_matrix = _convert_to_train_test(
            train_confusion_matrix, test_confusion_matrix)
        st.session_state.confusion_matrices.append(confusion_matrix)
    ruleset_stats = _convert_to_train_test(
        train_ruleset_stats, test_ruleset_stats)
    prediction_indicators = _convert_to_train_test(
        train_prediction_indicators, test_prediction_indicators)
    st.session_state.statistics.append(ruleset_stats)
    st.session_state.indicators.append(prediction_indicators)


def calculate_ruleset_stats(ruleset: AbstractRuleSet, X: pd.DataFrame, y: pd.Series):
    coverage_matrix = ruleset.calculate_rules_coverages(X, y)
    stats = ruleset.calculate_ruleset_stats()
    fraction_examples_covered: float = coverage_matrix.any(1).mean()
    stats["fraction_examples_covered"] = fraction_examples_covered
    if isinstance(ruleset, SurvivalRuleSet):
        stats = {
            key.replace("_", " "): value for key, value in stats.items()
        }
        return pd.DataFrame(stats, index=["metric"]).T
    if isinstance(ruleset, RegressionRuleSet):
        p_values = ruleset.calculate_p_values(y)
    else:
        p_values = ruleset.calculate_p_values()
    adjusted_p_values = correct_p_values_fdr(p_values)
    stats["fraction_significant"] = get_significant_fraction(p_values, 0.05)
    stats["fraction_FDR_significant"] = get_significant_fraction(
        adjusted_p_values, 0.05)
    stats = {
        key.replace("_", " "): value for key, value in stats.items()
    }
    return pd.DataFrame(stats, index=["metric"]).T


def calculate_prediction_indicators(ruleset: AbstractRuleSet, X: pd.DataFrame, y: pd.Series):
    y_pred = ruleset.predict(X)
    INDICATOR_FUNCTIONS = {
        ClassificationRuleSet: calculate_for_classification,
        RegressionRuleSet: calculate_for_regression,
        SurvivalRuleSet: partial(calculate_for_survival, ruleset, X),
    }
    indicators = INDICATOR_FUNCTIONS[type(ruleset)](y, y_pred)["general"]
    indicators = {
        key.replace("_", " "): value for key, value in indicators.items()
    }
    if isinstance(ruleset, ClassificationRuleSet):
        confusion_matrix = indicators.pop("Confusion matrix")
        confusion_matrix = pd.DataFrame(confusion_matrix).set_index("classes")
        confusion_matrix.index.name = None
        return pd.DataFrame(indicators, index=["indicator"]).T, confusion_matrix
    else:
        return pd.DataFrame(indicators, index=["indicator"]).T


def _convert_to_train_test(train_df, test_df):
    train_df.index = pd.MultiIndex.from_product([["train"], train_df.index])
    test_df.index = pd.MultiIndex.from_product([["test"], test_df.index])
    return pd.concat([train_df, test_df])
