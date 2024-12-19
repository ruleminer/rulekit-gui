import re

from matplotlib import pyplot as plt

plt.ioff()


def plot_kaplan_meier(rule):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(rule.conclusion.estimator.times,
            rule.conclusion.estimator.probabilities, "black")
    props = dict(boxstyle="round,pad=1", facecolor="grey", alpha=0.2)
    text_x_coord = (rule.conclusion.estimator.times.max(
    ) - rule.conclusion.estimator.times.min()) * 0.55 + rule.conclusion.estimator.times.min()
    rule_str = get_survival_rule_string(rule).replace(
        "AND ", "AND\n").replace("THEN", "\nTHEN").replace("(p", "\n(p")
    ax.text(text_x_coord, 0.95, rule_str,
            wrap=True, bbox=props, fontsize=12,
            verticalalignment="top")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Survival probability", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.margins(x=0)
    return fig


def get_survival_rule_string(rule):
    rule_str = str(rule).replace(
        rule.conclusion.column_name, "median survival time"
    )
    return re.sub(", [n,N]=\d*", "", rule_str)
