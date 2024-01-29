import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib
import datetime
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.neighbors import KernelDensity


# settings for plotting style
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino"] + plt.rcParams["font.serif"]
matplotlib.rc("font", size=7)
post_color1 = "#8f2727"
post_color2 = "#1f78b4"
prior_color = "gray"
prior_alpha = 0.9


def compute_change_points(post_df):
    tau_beta_1 = np.median(post_df["$τ_{β₁}$"])
    tau_beta_2 = np.median(post_df["$τ_{β₂}$"])
    tau_mu_1 = np.median(post_df["$τ_{μ₁}$"])
    try:
        tau_mu_2 = np.median(post_df["$τ_{μ₂}$"])
    except:
        tau_mu_2 = np.nan
    return tau_beta_1, tau_beta_2, tau_mu_1, tau_mu_2


# plot time series
def germany_time_series(
    ribbon,
    change_points,
    start_date="2020-03-01",
    figsize=(3.5, 3),
    layout=(2, 1),
    no_legend=False,
    scale_y = 1.35,
):
    date_list = pd.date_range(start_date, periods=len(ribbon))
    tau_beta_1, tau_beta_2, tau_mu_1, tau_mu_2 = change_points

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
    axes[0].scatter(date_list, ribbon["var 1 data"], color=colors[0], alpha=0.7, s=1)
    axes[0].plot(date_list, ribbon["var 1 median"], color=colors[0], linewidth=0.5)
    axes[0].fill_between(
        date_list,
        y1=ribbon["var 1 alpha 0.5 high"],
        y2=ribbon["var 1 alpha 0.5 low"],
        alpha=0.2,
        linewidth=0,
        color=colors[0],
    )
    axes[0].fill_between(
        date_list,
        y1=ribbon["var 1 alpha 0.1 high"],
        y2=ribbon["var 1 alpha 0.1 low"],
        alpha=0.2,
        linewidth=0,
        color=colors[0],
    )
    axes[0].fill_between(
        date_list,
        y1=ribbon["var 1 alpha 0.05 high"],
        y2=ribbon["var 1 alpha 0.05 low"],
        alpha=0.2,
        linewidth=0,
        color=colors[0],
    )
    axes[0].axvline(
        datetime.timedelta(days=tau_beta_1) + date_list[0],
        color="grey",
        linestyle="dashed",
        ymax=1/scale_y,
    )
    axes[0].axvline(
        datetime.timedelta(days=tau_beta_2) + date_list[0],
        color="grey",
        linestyle="dashed",
        ymax=1/scale_y,
    )
    axes[0].set_ylabel("Cases", fontsize=7)
    axes[0].set_xlabel("Date", fontsize=7)
    axes[0].tick_params(axis="both", which="major", labelsize=7)
    axes[0].set_xlim(xmin=date_list[0], xmax=date_list[len(ribbon) - 1])
    axes[0].set_ylim(ymin=0)
    axes[0].set_title("Cumulative infected cases per million", fontsize=7)
    axes[0].xaxis.set_major_locator(mdates.DayLocator((1)))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_ylim(ymax=axes[0].get_ylim()[1] * scale_y)
    axes[0].spines[["right", "top"]].set_visible(False)

    if not no_legend:
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=1, label="Estim. infected"),
            Line2D(
                [0],
                [0],
                marker="o",
                lw=0,
                color=colors[0],
                label="Observed infected",
                markersize=2,
                alpha=0.7,
            ),
            Line2D(
                [0],
                [0],
                linestyle="dashed",
                color="grey",
                label="Estim. change point $τ_{β}$",
                lw=1,
                markersize=10,
                markeredgewidth=1.5,
            ),
            Patch(
                facecolor=colors[0],
                edgecolor="none",
                alpha=0.6,
                label="50% credible interval",
            ),
            Patch(
                facecolor=colors[0],
                edgecolor="none",
                alpha=0.4,
                label="90% credible interval",
            ),
            Patch(
                facecolor=colors[0],
                edgecolor="none",
                alpha=0.2,
                label="95% credible interval",
            ),
        ]

        axes[0].legend(handles=legend_elements, loc="upper left", fontsize=6, ncol=2, frameon=False)

    axes[1].scatter(date_list, ribbon["var 2 data"], color=colors[3], alpha=0.7, s=1)
    axes[1].plot(date_list, ribbon["var 2 median"], color=colors[3], linewidth=0.5)
    axes[1].fill_between(
        date_list,
        y1=ribbon["var 2 alpha 0.5 high"],
        y2=ribbon["var 2 alpha 0.5 low"],
        alpha=0.2,
        linewidth=0,
        color=colors[3],
    )
    axes[1].fill_between(
        date_list,
        y1=ribbon["var 2 alpha 0.1 high"],
        y2=ribbon["var 2 alpha 0.1 low"],
        alpha=0.2,
        linewidth=0,
        color=colors[3],
    )
    axes[1].fill_between(
        date_list,
        y1=ribbon["var 2 alpha 0.05 high"],
        y2=ribbon["var 2 alpha 0.05 low"],
        alpha=0.2,
        linewidth=0,
        color=colors[3],
    )
    axes[1].axvline(
        datetime.timedelta(days=tau_mu_1) + date_list[0],
        color="grey",
        linestyle="dashed",
        ymax=1/scale_y,
    )
    if not np.isnan(tau_mu_2):
        axes[1].axvline(
            datetime.timedelta(days=tau_mu_2) + date_list[0],
            color="grey",
            linestyle="dashed",
        )

    axes[1].set_ylabel("Cases", fontsize=7)
    axes[1].set_xlabel("Date", fontsize=7)
    axes[1].set_title("Cumulative dead cases per million", fontsize=7)
    axes[1].tick_params(axis="both", which="major", labelsize=7)
    axes[1].set_xlim(xmin=date_list[0], xmax=date_list[len(ribbon) - 1])
    axes[1].set_ylim(ymin=0)
    axes[1].xaxis.set_major_locator(mdates.DayLocator((1)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].set_ylim(ymax=axes[1].get_ylim()[1] * scale_y)
    axes[1].spines[["right", "top"]].set_visible(False)
    if not no_legend:
        legend_elements = [
            Line2D([0], [0], color=colors[3], lw=1, label="Estim. dead"),
            Line2D(
                [0],
                [0],
                marker="o",
                lw=0,
                color=colors[3],
                label="Observed dead",
                markersize=2,
                alpha=0.7,
            ),
            Line2D(
                [0],
                [0],
                linestyle="dashed",
                color="grey",
                label="Estim. change point $τ_{μ}$",
                lw=1,
                markersize=10,
                markeredgewidth=1.5,
            ),
            Patch(
                facecolor=colors[3],
                edgecolor="none",
                alpha=0.6,
                label="50% credible interval",
            ),
            Patch(
                facecolor=colors[3],
                edgecolor="none",
                alpha=0.4,
                label="90% credible interval",
            ),
            Patch(
                facecolor=colors[3],
                edgecolor="none",
                alpha=0.2,
                label="95% credible interval",
            ),
        ]

        axes[1].legend(handles=legend_elements, loc="upper left", fontsize=6, ncol=2, frameon=False)

    fig.tight_layout()

    return fig


# compute MAP value
def estimate_map(samples):
    bw = 1.06 * samples.std() * samples.size ** (-1 / 5.0)
    scores = (
        KernelDensity(bandwidth=bw)
        .fit(samples.values.reshape(-1, 1))
        .score_samples(samples.values.reshape(-1, 1))
    )
    max_i = scores.argmax()
    map_i = samples[max_i]
    return map_i



# plot parameter posteriors vs. priors
def plot_posterior_prior_compare(
    prior_df,
    post_df1,
    post_df2,
    true_params,
    param_long,
    n_row,
    figsize=None,
    compare_method="compare",
    compare_changepoints={},
    actual_changepoints={},
    no_MAP_Med=False,
):
    n_col = int(np.ceil((true_params.size + 1) / n_row))
    if figsize is None:
        figsize = ((1.6 * n_col), (1.2 * n_row))
    f, ax = plt.subplots(n_row, n_col, figsize=figsize)
    # Generate fake data
    for i in range(true_params.size):
        sns.histplot(
            post_df1.iloc[:, i],
            stat="density",
            fill=True,
            color=post_color1,
            alpha=0.6,
            element="step",
            lw=0.5,
            bins=20,
            ax=ax.flat[i],
        )
        if i < len(post_df2.columns):
            sns.histplot(
                post_df2.iloc[:, i],
                stat="density",
                fill=True,
                color=post_color2,
                alpha=0.6,
                element="step",
                lw=0.5,
                bins=20,
                ax=ax.flat[i],
            )
        else:
            # plot the vertical line for the change point
            if i in compare_changepoints.keys():
                ax.flat[i].axvline(
                    compare_changepoints[i],
                    zorder=2,
                    label=f"{compare_method} change point",
                    ymax=1.0,
                    color=post_color2,
                    linestyle="solid",
                    lw=1,
                )
            if i in actual_changepoints.keys():
                ax.flat[i].axvline(
                    actual_changepoints[i],
                    zorder=2,
                    label="Actual change point",
                    ymax=1.0,
                    color="black",
                    linestyle="solid",
                    lw=1,
                )

        sns.histplot(
            prior_df.iloc[:, i],
            stat="density",
            fill=True,
            color=prior_color,
            alpha=0.3,
            bins=20,
            lw=0,
            ax=ax.flat[i],
        )
        MED = str("%s" % float("%.3g" % np.median(post_df1.iloc[:, i])))
        if float(MED) < 0.1:
            MED = str(f"{float(MED):.2e}")
        # ax.flat[i].axvline(float(MED), ls="--", linewidth=1, c="#8f2727", zorder=2)
        MAP = str("%s" % float("%.3g" % estimate_map(post_df1.iloc[:, i])))
        if float(MAP) < 0.1:
            MAP = str(f"{float(MAP):.2e}")
        if not no_MAP_Med:
            ax.flat[i].annotate(
                "$MAP$ = " + MAP,
                xy=(0.8, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=5,
            )
            ax.flat[i].annotate(
                "$Med$ = " + MED,
                xy=(0.8, 0.75),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=5,
            )

        ax.flat[i].set_title(param_long[i], fontsize=6)
        ax.flat[i].set_xlabel("")
        ax.flat[i].set_ylabel("")
        ax.flat[i].tick_params(axis="both", which="major", labelsize=7)
        ax.flat[i].ticklabel_format(
            style="sci", scilimits=(-2, 3), axis="both", useOffset=False
        )
        ax.flat[i].set_yticks([])
        ax.flat[i].xaxis.offsetText.set_fontsize(5)
        ax.flat[i].yaxis.offsetText.set_fontsize(5)
        ax.flat[i].spines[["right", "top"]].set_visible(False)
        ax.flat[i].set_ylim(ymax=ax.flat[i].get_ylim()[1])

    legend_elements = [
        Patch(facecolor=prior_color, label="Prior", alpha=0.3),
        Patch(facecolor=post_color1, label="Posterior ours", alpha=0.6),
        Patch(facecolor=post_color2, label=f"Posterior {compare_method}", alpha=0.6),
    ]

    ax.flat[i + 1].legend(
        handles=legend_elements, loc="center", fontsize=7, frameon=False
    )
    ax.flat[i + 1].tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    ax.flat[i + 1].spines[["right", "top", "bottom", "left"]].set_visible(False)

    legend_elements2 = [
        # Line2D(
        #     [0],
        #     [0],
        #     linestyle="dashed",
        #     color="#8f2727",
        #     label="Median of posterior",
        #     lw=1,
        #     markersize=10,
        #     markeredgewidth=1.5,
        # ),
        Line2D(
            [0],
            [0],
            linestyle="solid",
            color=post_color2,
            label=f"Change points: {compare_method}",
            lw=1,
            markersize=10,
            markeredgewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            linestyle="solid",
            color="black",
            label=f"Actual effective change points",
            lw=1,
            markersize=10,
            markeredgewidth=1.5,
        ),
    ]
    ax.flat[i + 2].legend(
        handles=legend_elements2, loc="center", fontsize=7, frameon=False
    )
    ax.flat[i + 2].tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    ax.flat[i + 2].spines[["right", "top", "bottom", "left"]].set_visible(False)

    for _ax in ax.flat[(post_df1.shape[-1] + 2) :]:
        _ax.remove()

    f.tight_layout()

    return f
