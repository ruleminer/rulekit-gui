import math

import numpy as np
import pandas as pd
from sklearn import metrics


def get_prediction_metrics(measure: str, y_pred, y_true, classification_metrics: dict) -> tuple[pd.DataFrame, np.ndarray]:
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)

    dictionary = {
        'Measure': measure,
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'MAE': metrics.mean_absolute_error(y_true, y_pred),
        'Kappa': metrics.cohen_kappa_score(y_true, y_pred),
        'Balanced accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'Logistic loss': metrics.log_loss(y_true, y_pred),
        'Precision': metrics.log_loss(y_true, y_pred),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'NPV': npv,
        'PPV': ppv,
        'psep': ppv + npv - 1,
        'Fall-out': fp / (fp + tn),
        "Youden's J statistic": sensitivity + specificity - 1,
        'Lift': (tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn)),
        'F-measure': 2 * tp / (2 * tp + fp + fn),
        'Fowlkes-Mallows index': metrics.fowlkes_mallows_score(y_true, y_pred),
        'False positive': fp,
        'False negative': fn,
        'True positive': tp,
        'True negative': tn,
        'Rules per example': classification_metrics['rules_per_example'],
        'Voting conflicts': classification_metrics['voting_conflicts'],
        'Negative voting conflicts': classification_metrics['negative_voting_conflicts'],
        'Geometric mean': math.sqrt(specificity * sensitivity),
    }
    return pd.DataFrame.from_records([dictionary], index='Measure'), confusion_matrix


def get_ruleset_stats_class(measure: str, model) -> pd.DataFrame:
    return pd.DataFrame.from_records([{'Measure': measure, **model.stats.__dict__}], index='Measure')


def get_regression_metrics(measure: str, y_pred, y_true) -> pd.DataFrame:
    relative_error = 0
    squared_relative_error = 0
    relative_error_lenient = 0
    relative_error_strict = 0
    nae_denominator = 0
    avg = sum(y_true) / len(y_pred)

    for i in range(0, len(y_pred)):
        true = y_true[i]
        predicted = y_pred[i]

        relative_error += abs((true - predicted) / true)
        squared_relative_error += abs((true - predicted) / true) * \
            abs((true - predicted) / true)
        relative_error_lenient += abs((true -
                                      predicted) / max(true, predicted))
        relative_error_strict += abs((true - predicted) / min(true, predicted))
        nae_denominator += abs(avg - true)
    relative_error /= len(y_pred)
    squared_relative_error /= len(y_pred)
    relative_error_lenient /= len(y_pred)
    relative_error_strict /= len(y_pred)
    nae_denominator /= len(y_pred)
    correlation = np.mean(np.corrcoef(y_true, y_pred))

    dictionary = {
        'Measure': measure,
        'absolute_error': metrics.mean_absolute_error(y_true, y_pred),
        'relative_error': relative_error,
        'relative_error_lenient': relative_error_lenient,
        'relative_error_strict': relative_error_strict,
        'normalized_absolute_error': metrics.mean_absolute_error(y_true, y_pred) / nae_denominator,
        'squared_error': metrics.mean_squared_error(y_true, y_pred),
        'root_mean_squared_error': metrics.mean_squared_error(y_true, y_pred, squared=False),
        'root_relative_squared_error': math.sqrt(squared_relative_error),
        'correlation': correlation,
        'squared_correlation': np.power(correlation, 2),
    }
    return pd.DataFrame.from_records([dictionary], index='Measure')


def get_ruleset_stats_reg(measure: str, model) -> pd.DataFrame:
    tmp = model.parameters.__dict__
    del tmp['_java_object']
    return pd.DataFrame.from_records([{'Measure': measure, **tmp, **model.stats.__dict__}], index='Measure')


def get_ruleset_stats_surv(model) -> pd.DataFrame:
    tmp = model.parameters.__dict__
    del tmp['_java_object']
    return pd.DataFrame.from_records([{**tmp, **model.stats.__dict__}])
