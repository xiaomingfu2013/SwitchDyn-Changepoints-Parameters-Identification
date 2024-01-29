using DataFrames, Plots, SciMLBase
# visualize the results
function plot_solutions(
    sol::ODESolution;
    ts=sol.t,
    color_palette=["#1f78b4" "#ff7f00" "#e31a1c"],
    loss_function_kwargs,
    ribbon_ratio=1.0,
    loss="",
    name="",
    event_times=[],
    event_times_true=[],
)
    @unpack data_true, mean_idxs, var_idxs = loss_function_kwargs
    plt_list = []
    for (i, (mean_idx, var_idx)) in enumerate(zip(mean_idxs, var_idxs))
        mean = sol[mean_idx, :]
        var = sol[var_idx, :]
        observed = data_true[i]
        plt = Plots.plot(
            ts,
            mean;
            label="LNA cases",
            xlabel="time",
            ylabel="numbers",
            ribbon=sqrt.(var) * ribbon_ratio,
            color=color_palette[i],
            left_margin=10Plots.mm,
            legend=:topleft,
            lw=2,
        )
        Plots.plot!(
            ts, observed; label="observed", color=color_palette[i], st=:scatter, ms=2
        )
        push!(plt_list, plt)
        if i == 1
            title!(plt, "$name $loss"; titlefont=12)
        end
        if length(event_times_true) > 0
            plot!(
                event_times_true[i];
                label="true change point",
                st=:vline,
                color="#e31a1c",
                linestyle=:solid,
                lw=2,
            )
        end
        if length(event_times) > 0
            plot!(
                event_times[i];
                label="estimated change point",
                st=:vline,
                color=:gray,
                linestyle=:dash,
                lw=3,
            )
        end
    end
    plt = Plots.plot(
        plt_list...; layout=(Int(length(plt_list)), 1), size=(800, 200 * length(plt_list))
    )
    return plt
end

function single_plot_solutions(
    sol::ODESolution;
    size=(800, 300),
    ts=sol.t,
    color_palette=["#1f78b4" "#ff7f00" "#e31a1c"],
    loss="",
    name="",
    ribbon_ratio=1.0,
    loss_function_kwargs,
    event_times=[],
    event_times_true=[],
)
    @unpack data_true, mean_idxs, var_idxs = loss_function_kwargs
    plt = Plots.plot(; size=size)
    for (i, (mean_idx, var_idx)) in enumerate(zip(mean_idxs, var_idxs))
        mean = sol[mean_idx, :]
        var = sol[var_idx, :]
        observed = data_true[i]
        Plots.plot!(
            ts,
            mean;
            label="LNA cases",
            xlabel="time",
            ylabel="numbers",
            ribbon=sqrt.(var) * ribbon_ratio,
            color=color_palette[i],
            left_margin=10Plots.mm,
            legend=:topleft,
            lw=2,
        )
        Plots.plot!(
            ts, observed; label="observed", color=color_palette[i], st=:scatter, ms=2
        )
        if length(event_times) > 0
            plot!(
                event_times_true;
                label="true change point",
                st=:vline,
                color="#e31a1c",
                linestyle=:solid,
                lw=2,
                legend=:outerright,
            )
            plot!(
                event_times;
                label="estimated change point",
                st=:vline,
                color=:gray,
                linestyle=:dash,
                lw=3,
                legend=:outerright,
            )
        end
    end
    title!(plt, "$name $loss"; titlefont=12)
    return plt
end
