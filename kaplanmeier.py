from matplotlib import pyplot as plt

plt.ioff()


def plot_kaplan_meier(rule):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rule.conclusion.estimator.times,
            rule.conclusion.estimator.probabilities, "black")
    props = dict(boxstyle="round,pad=1", facecolor="grey", alpha=0.2)
    text_x_coord = (rule.conclusion.estimator.times.max(
    ) - rule.conclusion.estimator.times.min()) * 0.6 + rule.conclusion.estimator.times.min()
    text_y_coord = (rule.conclusion.estimator.probabilities.max(
    ) - rule.conclusion.estimator.probabilities.min()) * 0.6 + rule.conclusion.estimator.probabilities.min()
    rule_str = str(rule).replace("AND ", "AND\n").replace(
        "THEN", "\nTHEN").replace("(p", "\n(p")
    ax.text(text_x_coord, text_y_coord, rule_str,
            wrap=True, bbox=props, fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Survival probability", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.margins(x=0)
    return fig
