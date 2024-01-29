import seaborn as sns
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib
import logging

# settings for plotting style
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino"] + plt.rcParams["font.serif"]
matplotlib.rc("font", size=4)
post_color_ours = "#8f2727"
post_color_mcp = "#1f78b4"


# plot time series
def plot_time_series(time_series_data, change_point_data, colors=colors, figsize=(4, 2)):
    fig, ax = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 10]}
    )

    # specify legend
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=1, label="LNA mean U"),
        Line2D([0], [0], color=colors[3], lw=1, label="LNA mean V"),
        Patch(facecolor=colors[0], edgecolor="none", alpha=0.2, label="LNA Std. U"),
        Patch(facecolor=colors[3], edgecolor="none", alpha=0.2, label="LNA Std. V"),
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            color=colors[0],
            label="Observed U",
            markersize=2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            color=colors[3],
            label="Observed V",
            markersize=2,
        ),
        Line2D(
            [0],
            [0],
            linestyle="solid",
            color="black",
            label="True change point",
            lw=0.5,
            markersize=1,
            markeredgewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            linestyle="dashed",
            color="grey",
            label="Estim. change point",
            lw=1,
            markersize=1,
            markeredgewidth=1.5,
        ),
    ]

    ax[0].legend(handles=legend_elements, loc="center", ncol=4, fontsize=5)
    ax[0].tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    ax[0].spines[["right", "top", "bottom", "left"]].set_visible(False)
    ts = time_series_data["date"]
    ax[1].scatter(
        ts,
        time_series_data["U_true"],
        color=colors[0],
        zorder=0,
        s=0.5,
        alpha=0.6,
    )
    ax[1].plot(
        ts,
        time_series_data["U_mean"],
        linewidth=1,
        color=colors[0],
        zorder=1,
        alpha=0.8,
    )
    ax[1].fill_between(
        ts,
        y1=time_series_data["U_mean"] + time_series_data["U_std"],
        y2=time_series_data["U_mean"] - time_series_data["U_std"],
        alpha=0.2,
        linewidth=0,
        color=colors[0],
        zorder=1,
    )
    ax[1].scatter(
        ts,
        time_series_data["V_true"],
        color=colors[3],
        zorder=1,
        s=0.5,
        alpha=0.6,
    )
    ax[1].plot(
        ts,
        time_series_data["V_mean"],
        linewidth=1,
        color=colors[3],
        zorder=0,
        alpha=0.8,
    )
    ax[1].fill_between(
        ts,
        y1=time_series_data["V_mean"] + time_series_data["V_std"],
        y2=time_series_data["V_mean"] - time_series_data["V_std"],
        alpha=0.2,
        linewidth=0,
        color=colors[3],
        zorder=0,
    )
    ax[1].axvline(x=change_point_data["true"], lw=0.5, color="black", zorder=2)
    ax[1].axvline(
        x=change_point_data["estim"], lw=1, color="grey", linestyle="dashed", zorder=2
    )
    ax[1].set_ylabel("Numbers", fontsize=7)
    ax[1].set_xlabel("Time", fontsize=7)
    ax[1].set_xlim(xmin=0, xmax=50)
    ax[1].tick_params(axis="both", labelsize=7)
    ax[1].spines[["right", "top"]].set_visible(False)

    fig.tight_layout()

    return fig




# plot publication-ready histograms of rank statistics for simulation-based calibration (SBC) checks
def plot_histogram(
    data_posterior,
    data_true,
    param_names,
    fig_size=None,
    num_bins=5,
    binomial_interval=0.99,
    label_fontsize=7,
    title_fontsize=7,
    tick_fontsize=7,
    hist_color="#a34f4f",
):
    n_sim, n_draws, n_params = data_posterior.shape
    ratio = int(n_sim / n_draws)

    # Log a warning if N/B ratio recommended by Talts et al. (2018) < 20
    if ratio < 20:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(
            f"The ratio of simulations / posterior draws should be > 20 "
            + f"for reliable variance reduction, but your ratio is {ratio}.\
                    Confidence intervals might be unreliable!"
        )

    # Set n_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 5

    # Determine n params and param names if None given
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine n_subplots dynamically
    n_row = int(np.ceil(n_params / 6))
    n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = ((1.45 * n_col), (1.8 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    # Compute ranks (using broadcasting)
    ranks = np.sum(data_posterior < data_true[:, np.newaxis, :], axis=1)

    # Compute confidence interval and mean
    N = int(data_true.shape[0])
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(binomial_interval, N, 1 / num_bins)
    mean = N / num_bins  # corresponds to binom.mean(N, 1 / num_bins)

    # Plot marginal histograms in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(len(param_names)):
        ax[j].axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.3)
        ax[j].axhline(mean, color="gray", zorder=0, alpha=0.9)
        sns.histplot(
            ranks[:, j],
            kde=False,
            ax=ax[j],
            color=hist_color,
            bins=num_bins,
            alpha=0.85,
            lw=0.5,
        )
        ax[j].set_title(param_names[j], fontsize=title_fontsize, style="italic")
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].get_yaxis().set_ticks([])
        if j == 0:
            ax[j].set_ylabel("Count", size=tick_fontsize)
        else:
            ax[j].set_ylabel("")
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = (
        axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    )
    for _ax in bottom_row:
        _ax.set_xlabel("Rank statistic", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f

