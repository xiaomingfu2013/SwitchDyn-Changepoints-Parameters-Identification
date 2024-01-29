import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino"] + plt.rcParams["font.serif"]


def plot_timeseries(jsol_list, true_df, figsize=(3.5, 2.2), sample_id1 = 0,
    sample_id2 = 6):

    ts = np.linspace(0, 50, 101)
    # plot example
    fig, ax = plt.subplots(figsize=figsize)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax.plot(ts, jsol_list[sample_id1]['P(t)'], drawstyle="steps-pre", color=colors[0], alpha=0.8)
    ax.axvline((true_df.iloc[sample_id1, 2]), color=colors[0], linestyle="dashed", alpha=0.8)
    ax.plot(ts, jsol_list[sample_id2]['P(t)'], drawstyle="steps-pre", color=colors[3], alpha=0.8)
    ax.axvline((true_df.iloc[sample_id2, 2]), color=colors[3], linestyle="dashed", alpha=0.8)
    ax.set_xlabel("Time", fontsize=7)
    ax.set_ylabel("# Protein", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)

    legend_elements = [
        Line2D([0], [0], color=colors[0], label="Gene 1", alpha=0.8),
        Line2D([0], [0], color=colors[3], label="Gene 2", alpha=0.8),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # turn oof the upper and right axes
    ax.spines[["right", "top"]].set_visible(False)

    ax.text(
        12,
        45,
        "τ",
        horizontalalignment="left",
        verticalalignment="center",
        size=7,
        color=colors[0],
        style="italic",
    )
    ax.text(
        17,
        45,
        "τ",
        horizontalalignment="left",
        verticalalignment="center",
        size=7,
        color=colors[3],
        style="italic",
    )
    ax.text(
        3,
        45,
        "k₁",
        horizontalalignment="left",
        verticalalignment="center",
        size=7,
        color=colors[0],
        style="italic",
    )
    ax.text(
        5,
        25,
        "k₁",
        horizontalalignment="left",
        verticalalignment="center",
        size=7,
        color=colors[3],
        style="italic",
    )
    ax.text(
        33,
        27,
        "k₂",
        horizontalalignment="left",
        verticalalignment="center",
        size=7,
        color=colors[0],
        style="italic",
    )
    ax.text(
        33,
        5,
        "k₂",
        horizontalalignment="left",
        verticalalignment="center",
        size=7,
        color=colors[3],
        style="italic",
    )
    fig.tight_layout()
    return fig