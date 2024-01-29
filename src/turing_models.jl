
function turing_change_point(
    priors_tuple::NamedTuple;
    loss_function_kwargs,
    logpdf_custom=(mean, var, data) -> logpdf(MvNormal(mean, Diagonal(var)), data),
    solvesettings=loss_function_kwargs.solvesettings,
)
    @unpack data_true, u0_expanded, sensealg, prob, alg, ts, set_of_p_var, unflatten =
        loss_function_kwargs
    @unpack mean_idxs, var_idxs = loss_function_kwargs
    sym_priors = create_symbol_dict(priors_tuple)
    pdist = product_distribution(collect(values(sym_priors)))
    # num_ts = length(ts)
    # num_idxs = length(mean_idxs)
    # num = num_ts * num_idxs # shall we ignore this?
    println("Prior distributions: ")
    display(sym_priors)
    # pdist_ = product_distribution(NamedDist.(pdist, syms))
    @model function model(data_true, ::Type{T}=Float64) where {T<:Real}
        # pprior = Vector{T}(undef, length(pdist)) # mutating array
        # pprior .~ NamedDist.(pdist, syms)
        # for i in eachindex(pdist)
        #     pprior[i] ~ NamedDist(pdist[i], syms[i])
        # end
        pprior ~ pdist
        p_var, p_invar, event_times = build_ps(unflatten(pprior); return_array=false)
        # u0_ = convert.(T, u0_expanded)
        cp = ChangePointParameters(u0_expanded, p_var, p_invar, event_times, set_of_p_var)
        sol = solve_event_times(cp, ts, prob, alg, sensealg; solvesettings...)

        failure = size(sol, 2) < length(ts)

        if failure
            Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
            Turing.@addlogprob! -Inf
            return nothing
        end

        # loss = zero(T)
        # σ ~ InverseGamma(2, 3)
        for (i, (mean_idx, var_idx)) in enumerate(zip(mean_idxs, var_idxs))
            mean = sol[mean_idx, :]
            var = max.(sol[var_idx, :], one(T))
            observed = data_true[i]
            if eltype(observed) == Missing
                continue
            else
                Turing.@addlogprob! logpdf_custom(mean, var, observed)
            end
            # num_σ = sum(predict[var_idx, :] .< σ)
            # Turing.@addlogprob! logpdf_custom(mean, var, observed) - num_σ * log(σ)
        end
        return nothing
    end
    turing_model = model(data_true)
    return turing_model
end

# design a loss function that is equivalent to the one used in the optimisation
function loss_function_turing(
    ps::AbstractArray{T},
    p_null=SciMLBase.NullParameters();
    loss_function_kwargs,
    logpdf_custom=(mean, var, data) -> logpdf(MvNormal(mean, Diagonal(var)), data),
    solvesettings=loss_function_kwargs.solvesettings,
) where {T<:Real}
    @unpack data_true, u0_expanded, sensealg, prob, alg, ts, set_of_p_var, unflatten =
        loss_function_kwargs
    @unpack mean_idxs, var_idxs = loss_function_kwargs

    p_var, p_invar, event_times = build_ps(unflatten(ps); return_array=false)
    # u0_ = convert.(T, u0_expanded)
    cp = ChangePointParameters(u0_expanded, p_var, p_invar, event_times, set_of_p_var)
    sol = solve_event_times(cp, ts, prob, alg, sensealg; solvesettings...)
    loss = zero(T)
    for (i, (mean_idx, var_idx)) in enumerate(zip(mean_idxs, var_idxs))
        mean = sol[mean_idx, :]
        var = max.(sol[var_idx, :], one(T))
        observed = data_true[i]
        if eltype(observed) == Missing
            continue
        else
            loss += -logpdf_custom(mean, var, observed)
        end
        # num_σ = sum(predict[var_idx, :] .< σ)
        # Turing.@addlogprob! logpdf_custom(mean, var, observed) - num_σ * log(σ)
    end
    return loss
end
