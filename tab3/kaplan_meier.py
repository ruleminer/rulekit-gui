from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from matplotlib import pyplot as plt


@dataclass
class _KMPlotData:
    times: np.ndarray
    probabilities: np.ndarray


def get_kaplan_meier(rule, covered_data):
    times = rule.conclusion.estimator.times
    probabilities = rule.conclusion.estimator.probabilities
    times, probabilities = _add_missing_km_points(
        covered_data, times, probabilities)
    return _KMPlotData(times, probabilities)


def plot_kaplan_meier(rules_km_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    rules_km_data.apply(lambda x: _plot_kaplan_meier_for_rule(x, ax), axis=1)
    dataset = pd.concat([st.session_state.x, st.session_state.y], axis=1)
    max_time = dataset["survival_time"].max()
    dataset_km = _get_kaplan_meier_for_dataset(dataset)
    ax.step(dataset_km.times, dataset_km.probabilities,
            "black", where="post", label="dataset",)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Survival probability", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_time + 1)
    ax.margins(x=0)
    ax.legend(fontsize=12, loc="upper right")
    st.pyplot(fig)


def _plot_kaplan_meier_for_rule(rule, ax):
    ax.step(rule["KM"].times, rule["KM"].probabilities,
            where="post", label=rule["ID"])


def _get_kaplan_meier_for_dataset(df: pd.DataFrame):
    estimator = KaplanMeierEstimator()
    estimator.fit(df["survival_time"].to_numpy(),
                  df["survival_status"].to_numpy())
    times, probabilities = _add_missing_km_points(
        df, estimator.surv_info.time, estimator.surv_info.probability)
    return _KMPlotData(times, probabilities)


def _add_missing_km_points(data: pd.DataFrame, times: np.ndarray, probabilities: np.ndarray):
    if times[0] != 0.0:
        times = np.insert(times, 0, 0.0)
        probabilities = np.insert(probabilities, 0, 1.0)
    max_surv_time = data["survival_time"].max()
    data_with_max_surv_time = data[data["survival_time"] == max_surv_time]
    if len(data_with_max_surv_time) == data_with_max_surv_time["survival_status"].sum():
        times = np.append(times, max_surv_time)
        probabilities = np.append(probabilities, 0.0)
    return times, probabilities
