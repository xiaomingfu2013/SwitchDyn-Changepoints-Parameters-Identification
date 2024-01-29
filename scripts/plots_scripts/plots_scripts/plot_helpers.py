from plots_scripts.utils_least_square import total_least_squares, ordinary_least_squares
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino"] + plt.rcParams["font.serif"]



def plot_recovery(
    data_posterior,
    data_true,
    param_names,
    color,
    point_agg=np.median,
    uncertainty_agg=median_abs_deviation,
    fig_size=None,
    label_fontsize=7,
    title_fontsize=7,
    metric_fontsize=7,
    tick_fontsize=7,
    add_corr=True,
    add_r2=True,
    n_row=None,
    n_col=None,
    trend_line='TLS',
    boundary={}, # boundary for ploting each parameter, e.g. boundary = {'1': [0, 1], '2': [0, 1]}
):
    est = point_agg(data_posterior, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(data_posterior, axis=1)

    # Determine n params and param names if None given
    n_params = data_posterior[:, :].shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (1.45 * n_col, 2 * n_row)
    f2, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr_it = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr_it = [axarr]

    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break
        x = data_true[:, i]
        y = est[:, i]
        # Add scatter and errorbars
        if uncertainty_agg is not None:
            _ = ax.errorbar(
                x,
                y,
                yerr=u[:, i],
                fmt="o",
                alpha=0.3,
                color=color,
                ms=2,
                elinewidth=1,
            )
        else:
            _ = ax.scatter(x, y, alpha=0.3, color=color, s=2)

        
                
                    
        # Make plots quadratic to avoid visual illusions
        # if lower has no key i, then set lower to min of data_true and est, else set lower to lower[i], same for upper
        if i not in boundary.keys():
            lower = min(x.min(), y.min())
            upper = max(x.max(), y.max())
        else:
            lower = boundary[i][0]
            upper = boundary[i][1]
            
        eps = (upper - lower) * 0.1
        # xticks and yticks to be the same
        
        
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        # ax.set_xticks(ax.get_yticks())
        # ax.set_yticks(ax.get_xticks())
        ax.locator_params(axis="both", nbins=5)
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )
        
        if trend_line is not None:
            if trend_line == "OLS":
                # Fit linear regression with ordinary least squares
                slope, intercept = ordinary_least_squares(x, y)
            elif trend_line == "TLS":
                # Fit linear regression with total least squares
                slope, intercept = total_least_squares(x, y)
            else:
                raise ValueError(
                    f"trend_line must be either 'OLS' or 'TLS' or None, but is {trend_line}."
                )
            linear_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
            ax.plot(
                linear_x,
                intercept + slope * linear_x,
                color='gray',
                alpha=1.0,
                lw=1.5,
                linestyle="solid",
                zorder=1,
            )
            # plot the trend line equation at the bottom right corner
            # intercept_sign = "+" if intercept >= 0 else "-"
            # ax.text(
            #     0.1,
            #     0.9,
            #     f"$y={slope:.2f}x{intercept_sign}{np.abs(intercept):.2f}$",
            #     horizontalalignment="left",
            #     verticalalignment="center",
            #     transform=ax.transAxes,
            #     size=metric_fontsize * 0.8,
            # )
        # axis equal ratio 
        ax.set_aspect("equal", "box")
        


        # Add optional metrics and title
        if add_corr:
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.9,
                0.1,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_r2:
            r2 = r2_score(x, y)
            ax.text(
                0.9,
                0.2,
                "$R^2$ = {:.3f}".format(r2),
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        
        ax.set_title(param_names[i], fontsize=title_fontsize, style="italic")

        # Prettify
        sns.despine(ax=ax)
        # ax.grid(alpha=0.5) # turn off grid
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax.ticklabel_format(
            style="sci", scilimits=(-2, 2), axis="both", useOffset=False
        )
        ax.xaxis.offsetText.set_fontsize(5)
        ax.yaxis.offsetText.set_fontsize(5)


    bottom_row = (
        axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    )
        
    for _ax in bottom_row:
        _ax.set_xlabel("Ground truth", fontsize=label_fontsize)
        _ax.xaxis.set_label_coords(0.5, -0.3)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr_it[0].set_ylabel("Estimated", fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr_it[:, 0]:
            _ax.set_ylabel("Estimated", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    f2.tight_layout()
    return f2
