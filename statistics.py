from functools import partial

import pandas as pd
from decision_rules.classification import ClassificationRuleSet
from decision_rules.classification.prediction_indicators import calculate_for_classification
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.helpers import correct_p_values_fdr
from decision_rules.helpers import get_significant_fraction
from decision_rules.regression import RegressionRuleSet
from decision_rules.regression.prediction_indicators import calculate_for_regression
from decision_rules.survival import SurvivalRuleSet
from decision_rules.survival.prediction_indicators import calculate_for_survival


def calculate_ruleset_stats(ruleset: AbstractRuleSet, X: pd.DataFrame, y: pd.Series):
    coverage_matrix = ruleset.calculate_rules_coverages(X, y)
    stats = ruleset.calculate_ruleset_stats()
    fraction_examples_covered: float = coverage_matrix.any(1).mean()
    stats['fraction_examples_covered'] = fraction_examples_covered
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
    stats['fraction_significant'] = get_significant_fraction(p_values, 0.05)
    stats['fraction_FDR_significant'] = get_significant_fraction(
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
        return pd.DataFrame(indicators, index=["indicator"]).T, pd.DataFrame(confusion_matrix)
    else:
        return pd.DataFrame(indicators, index=["indicator"]).T
