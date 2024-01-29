function posterior_predict(
    save_margs, loss_function_kwargs, color_palette=["#1f78b4" "#e31a1c"]
)
    @unpack ts, tspan, scaled_chain, unscaled_chain = save_margs
    @unpack data_true, mean_idxs, var_idxs, unflatten = loss_function_kwargs
    @unpack alg, u0_expanded, solvesettings = loss_function_kwargs

    median_ps = quantile(unscaled_chain).nt.var"50.0%"
    #  = build_ps(unflatten(median_ps))
    p_var, p_invar, event_times = build_ps(unflatten(median_ps); return_array=false)
    ps = [p_var; p_invar; event_times]
    print(ps)
    model = loss_function_kwargs.prob.f
    prob_embed = ODEProblem(model, u0_expanded, tspan, ps)
    sol_median = solve(prob_embed, alg; saveat=ts, tstops=event_times, solvesettings...)

    plt_list = []
    event_times_set = [event_times[1:2], event_times[3:end]]
    df_quantile_sol = DataFrame()
    df_quantile_sol[!, "time"] = ts
    for (i, idx) in enumerate(mean_idxs)
        quantile_median = sol_median[idx, :]
        plt = plot(
            ts,
            data_true[i];
            legend=:topleft,
            color=color_palette[i],
            size=(600, 300),
            label="Observed data",
            st=:scatter,
            ms=3,
        )
        # add event times lines
        for event_time in event_times_set[i]
            vline!(
                [event_time],
                color="gray",
                label="event time",
                linestyle=:dash,
                linewidth=3,
            )
        end
        for alpha in [0.50, 0.10, 0.05]
            hpd_res = MCMCChains.hpd(unscaled_chain; alpha=alpha)
            low_ps = build_ps(unflatten(hpd_res[:, 2]))
            high_ps = build_ps(unflatten(hpd_res[:, 3]))

            event_times_low = low_ps[(end - 3):end]
            sol_low = solve(
                prob_embed,
                alg;
                saveat=ts,
                tstops=event_times_low,
                solvesettings...,
                p=low_ps,
            )
            event_times_high = high_ps[(end - 3):end]
            sol_high = solve(
                prob_embed,
                alg;
                saveat=ts,
                tstops=event_times_high,
                solvesettings...,
                p=high_ps,
            )

            quantile_low = sol_low[idx, :]
            quantile_high = sol_high[idx, :]
            #plot time series
            Plots.plot!(
                ts,
                quantile_median;
                legend=:topleft,
                label="CI: $(round(Int, (1-alpha)*100)) %",
                ribbon=(quantile_median .- quantile_low, quantile_high .- quantile_median),
                fillalpha=0.25,
                color=color_palette[i],
            )
            df_quantile_sol[!, "var $i alpha $alpha low"] = quantile_low
            df_quantile_sol[!, "var $i alpha $alpha high"] = quantile_high
        end
        # push!(df_quantile_sol, "var $i median" => quantile_median)
        df_quantile_sol[!, "var $i median"] = quantile_median
        df_quantile_sol[!, "var $i data"] = data_true[i]
        push!(plt_list, plt)
    end
    df_quantile_sol
    plt1 = plot(plt_list...; layout=(2, 1), size=(600, 400), legend=:topleft)
    return df_quantile_sol, plt1
end
