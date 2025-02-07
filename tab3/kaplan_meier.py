from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt


@dataclass
class _KMPlotData:
    times: np.ndarray
    probabilities: np.ndarray


def get_kaplan_meier(rule, covered_data):
    times = rule.conclusion.estimator.times
    probabilities = rule.conclusion.estimator.probabilities
    if times[0] != 0.0:
        times = np.insert(times, 0, 0.0)
        probabilities = np.insert(probabilities, 0, 1.0)
    last_covered_row = covered_data.sort_values("survival_time").iloc[-1]
    if last_covered_row["survival_status"] == 1:
        times = np.insert(times, -1, last_covered_row["survival_time"])
        probabilities = np.insert(probabilities, -1, 0.0)
    return _KMPlotData(times, probabilities)


def _plot_kaplan_meier_for_rule(rule, ax):
    ax.step(rule["KM"].times, rule["KM"].probabilities,
            where="post", label=rule["ID"])


def plot_kaplan_meier(rules_km_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    max_times = rules_km_data["KM"].apply(lambda x: max(x.times)).max()
    rules_km_data.apply(lambda x: _plot_kaplan_meier_for_rule(x, ax), axis=1)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Survival probability", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_times + 1)
    ax.margins(x=0)
    ax.legend(title="Rule ID", title_fontsize=15, fontsize=12)
    st.pyplot(fig)
