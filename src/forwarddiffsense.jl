include("extract_local_sensitivity.jl")

@inline convert_tspan(::ForwardDiffSensitivity{CS, CTS}) where {CS, CTS} = CTS

# Generic Fallback for ForwardDiff
function generate_forwarddiff_callback(event_times, f_iip, nump_var_invar, cp)
    calibrate_grad =
        let f_iip = f_iip, nump_var_invar = nump_var_invar, event_times = event_times
            function (integrator)
                u_cur = ForwardDiff.value.(integrator.u)

                partial_cur = zero(ForwardDiff.partials.(integrator.u)) # get partials values: a vector of length u_cur, and each element is a vector of length p
                partial_val = reduce.(vcat, partial_cur) # convert the a vector of tuple to a vector of vector


                δu = compute_changepoint_derivative(
                    u_cur, f_iip, ForwardDiff.value.(integrator.p), integrator.t
                )
                event_time_idxs =
                    findall(isequal(integrator.t), event_times) .+ nump_var_invar # in case that s1 = t1 two event times are identical

                    for u_idx in eachindex(δu), event_time_idx in event_time_idxs
                    partial_val[u_idx][event_time_idx] = -δu[u_idx]
                end

                T1 = eltype(u_cur)
                T2 = typeof(first(integrator.u))
                partial_modify = Vector{T2}(undef, length(integrator.u))
                for u_idx in eachindex(integrator.u)
                    partial_modify[u_idx] = T2(
                        zero(T1), ForwardDiff.Partials(Tuple(partial_val[u_idx]))
                    )
                end

                integrator.u .+= partial_modify

                # event_time += 1
                return nothing
            end
        end
    cb_calibrate = PresetTimeCallback(
        event_times, calibrate_grad; save_positions=(false, false)
    )
    return cb_calibrate
end

function ChainRulesCore.rrule(
    ::typeof(solve_event_times),
    cp,
    ts,
    prob,
    alg,
    sensealg::ForwardDiffSensitivity{CS,CTS};
    save_idxs=nothing,
    solvesettings=Dict(),
    kwargs...,
) where {CS,CTS}
    @info "solve_event_times using ForwardDiffSensitivity experimental" maxlog = 1
    @unpack u0, p_var, p_invar, event_times, set_of_p_var = cp
    u0 = eltype(p_var).(u0)
    p = [p_var; p_invar; event_times]
    saveat = ts
    if saveat isa Number
        _saveat = prob.tspan[1]:saveat:prob.tspan[2]
    else
        _saveat = saveat
    end

    # set_of_event_times = haskey(cp.kwargs, :set_of_event_times) ? cp.kwargs[:set_of_event_times] : 1

    sol = solve_event_times(
        cp, ts, prob, alg, sensealg; save_idxs=save_idxs, (solvesettings...), kwargs...
    )

    # saveat values
    # need all values here. Not only unique ones.
    # if a callback is saving two times in primal solution, we also need to get it at least
    # two times in the solution using dual numbers.
    ts = sol.t
    # numu0 = length(u0)
    num_p_var = length(p_var)
    num_p_invar = length(p_invar)
    nump_var_invar = num_p_var + num_p_invar  # num lambda, num p,
    f_iip = prob.f.f
    # odeadjprob = ODEForwardSensitivityProblem(_f, u0, prob.tspan, ps, sensealg)

    function forward_sensitivity_backpass(Δ)
        if !(p === nothing || p === DiffEqBase.NullParameters())
            dp = @thunk begin
                chunk_size = length(p)

                num_chunks = length(p) ÷ chunk_size
                num_chunks * chunk_size != length(p) && (num_chunks += 1)

                pparts = typeof(p[1:1])[]
                for j in 0:(num_chunks - 1)
                    local chunk
                    if ((j + 1) * chunk_size) <= length(p)
                        chunk = ((j * chunk_size + 1):((j + 1) * chunk_size))
                        pchunk = vec(p)[chunk]
                        pdualpart = seed_duals(
                            pchunk, prob.f, ForwardDiff.Chunk{chunk_size}()
                        )
                    else
                        chunk = ((j * chunk_size + 1):length(p))
                        pchunk = vec(p)[chunk]
                        pdualpart = seed_duals(
                            pchunk, prob.f, ForwardDiff.Chunk{length(chunk)}()
                        )
                    end

                    pdualvec = if j == 0
                        vcat(pdualpart, p[((j + 1) * chunk_size + 1):end])
                    elseif j == num_chunks - 1
                        vcat(p[1:(j * chunk_size)], pdualpart)
                    else
                        vcat(
                            p[1:(j * chunk_size)],
                            pdualpart,
                            p[(((j + 1) * chunk_size) + 1):end],
                        )
                    end

                    pdual = ArrayInterface.restructure(p, pdualvec)
                    u0dual = convert.(eltype(pdualvec), u0)

                    if (
                        convert_tspan(sensealg) === nothing && ((
                            haskey(kwargs, :callback) &&
                            has_continuous_callback(kwargs[:callback])
                        ))
                    ) || (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
                        tspandual = convert.(eltype(pdual), prob.tspan)
                    else
                        tspandual = prob.tspan
                    end

                    ## Force recompile mode because it won't handle the duals
                    ## Would require a manual tag to be applied
                    if typeof(prob.f) <: ODEFunction
                        if prob.f.jac_prototype !== nothing
                            _f = ODEFunction{
                                SciMLBase.isinplace(prob.f),SciMLBase.FullSpecialize
                            }(
                                unwrapped_f(prob.f);
                                jac_prototype=convert.(
                                    eltype(u0dual), prob.f.jac_prototype
                                ),
                            )
                        else
                            _f = ODEFunction{
                                SciMLBase.isinplace(prob.f),SciMLBase.FullSpecialize
                            }(
                                unwrapped_f(prob.f)
                            )
                        end
                    else
                        _f = prob.f
                    end
                    _prob = remake(prob; f=_f, u0=u0dual, p=pdual, tspan=tspandual)

                    if saveat isa Number
                        _saveat = prob.tspan[1]:saveat:prob.tspan[2]
                    else
                        _saveat = saveat
                    end
                    cb_calibrate = generate_forwarddiff_callback(
                        event_times, f_iip, nump_var_invar, cp
                    )
                    _sol = solve(
                        _prob,
                        alg;
                        saveat=ts,
                        callback=cb_calibrate,
                        tstops=event_times,
                        (solvesettings...),
                        kwargs...,
                    )
                    _, du = extract_local_sensitivities(_sol, sensealg, Val(true))

                    if haskey(kwargs, :callback)
                        # handle bounds errors: ForwardDiffSensitivity uses dual numbers, so there
                        # can be more or less time points in the primal solution
                        # than in the solution using dual numbers when adaptive solvers are used.
                        # First step: filter all values, so that only time steps that actually occur
                        # in the primal are left. This is for example necessary when `terminate!`
                        # is used.
                        indxs = findall(t -> t ∈ ts, _sol.t)
                        _ts = _sol.t[indxs]
                        # after this filtering step, we might end up with a too large amount of indices.
                        # For example, if a callback saved values in the primal, then we now potentially
                        # save it by `saveat` and by `save_positions` of the callback.
                        # Second step. Drop these duplicates values.
                        if length(indxs) != length(ts)
                            for i in (length(_ts) - 1):-1:2
                                if _ts[i] == _ts[i + 1] && _ts[i] == _ts[i - 1]
                                    deleteat!(indxs, i)
                                end
                            end
                        end
                        _du = @view du[indxs]
                    else
                        _du = du
                    end

                    _dp = sum(eachindex(_du)) do i
                        J = _du[i]
                        if Δ isa AbstractVector ||
                            Δ isa DESolution ||
                            Δ isa AbstractVectorOfArray
                            v = Δ[i]
                        elseif Δ isa AbstractMatrix
                            v = @view Δ[:, i]
                        else
                            v = @view Δ[.., i]
                        end
                        if !(Δ isa NoTangent)
                            if u0 isa Number
                                ForwardDiff.value.(J'v)
                            else
                                ForwardDiff.value.(J'vec(v))
                            end
                        else
                            zero(p)
                        end
                    end
                    push!(pparts, vec(_dp))
                end
                ArrayInterface.restructure(p, reduce(vcat, pparts))
            end
        else
            dp = nothing
        end
        if sensealg.du0_sense
            du0 = @thunk begin
                chunk_size = length(p)

                num_chunks = length(u0) ÷ chunk_size
                num_chunks * chunk_size != length(u0) && (num_chunks += 1)

                du0parts = u0 isa Number ? typeof(u0)[] : typeof(u0[1:1])[]

                local _du0

                for j in 0:(num_chunks - 1)
                    local chunk
                    if u0 isa Number
                        u0dualpart = seed_duals(u0, prob.f, ForwardDiff.Chunk{chunk_size}())
                    elseif ((j + 1) * chunk_size) <= length(u0)
                        chunk = ((j * chunk_size + 1):((j + 1) * chunk_size))
                        u0chunk = vec(u0)[chunk]
                        u0dualpart = seed_duals(
                            u0chunk, prob.f, ForwardDiff.Chunk{chunk_size}()
                        )
                    else
                        chunk = ((j * chunk_size + 1):length(u0))
                        u0chunk = vec(u0)[chunk]
                        u0dualpart = seed_duals(
                            u0chunk, prob.f, ForwardDiff.Chunk{length(chunk)}()
                        )
                    end

                    if u0 isa Number
                        u0dual = u0dualpart
                    else
                        u0dualvec = if j == 0
                            vcat(u0dualpart, u0[((j + 1) * chunk_size + 1):end])
                        elseif j == num_chunks - 1
                            vcat(u0[1:(j * chunk_size)], u0dualpart)
                        else
                            vcat(
                                u0[1:(j * chunk_size)],
                                u0dualpart,
                                u0[(((j + 1) * chunk_size) + 1):end],
                            )
                        end

                        u0dual = ArrayInterface.restructure(u0, u0dualvec)
                    end

                    if p === nothing || p === DiffEqBase.NullParameters()
                        pdual = p
                    else
                        pdual = convert.(eltype(u0dual), p)
                    end

                    if (
                        convert_tspan(sensealg) === nothing && ((
                            haskey(kwargs, :callback) &&
                            has_continuous_callback(kwargs[:callback])
                        ))
                    ) || (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
                        tspandual = convert.(eltype(pdual), prob.tspan)
                    else
                        tspandual = prob.tspan
                    end

                    ## Force recompile mode because it won't handle the duals
                    ## Would require a manual tag to be applied
                    if typeof(prob.f) <: ODEFunction
                        if prob.f.jac_prototype !== nothing
                            _f = ODEFunction{
                                SciMLBase.isinplace(prob.f),SciMLBase.FullSpecialize
                            }(
                                unwrapped_f(prob.f);
                                jac_prototype=convert.(eltype(pdual), prob.f.jac_prototype),
                            )
                        else
                            _f = ODEFunction{
                                SciMLBase.isinplace(prob.f),SciMLBase.FullSpecialize
                            }(
                                unwrapped_f(prob.f)
                            )
                        end
                    else
                        _f = prob.f
                    end

                    _prob = remake(prob; f=_f, u0=u0dual, p=pdual, tspan=tspandual)

                    if saveat isa Number
                        _saveat = prob.tspan[1]:saveat:prob.tspan[2]
                    else
                        _saveat = saveat
                    end
                    #   cb_calibrate = generate_forwarddiff_callback(event_times, f_iip, nump_var_invar, prob)
                    _sol = solve(
                        _prob,
                        alg;
                        saveat=ts,
                        tstops=event_times,
                        (solvesettings...),
                        kwargs...,
                    )
                    _, du = extract_local_sensitivities(_sol, sensealg, Val(true))

                    if haskey(kwargs, :callback)
                        # handle bounds errors: ForwardDiffSensitivity uses dual numbers, so there
                        # can be more or less time points in the primal solution
                        # than in the solution using dual numbers when adaptive solvers are used.
                        # First step: filter all values, so that only time steps that actually occur
                        # in the primal are left. This is for example necessary when `terminate!`
                        # is used.
                        indxs = findall(t -> t ∈ ts, _sol.t)
                        _ts = _sol.t[indxs]
                        # after this filtering step, we might end up with a too large amount of indices.
                        # For example, if a callback saved values in the primal, then we now potentially
                        # save it by `saveat` and by `save_positions` of the callback.
                        # Second step. Drop these duplicates values.
                        if length(indxs) != length(ts)
                            for i in (length(_ts) - 1):-1:2
                                if _ts[i] == _ts[i + 1] && _ts[i] == _ts[i - 1]
                                    deleteat!(indxs, i)
                                end
                            end
                        end
                        _du = @view du[indxs]
                    else
                        _du = du
                    end

                    _du0 = sum(eachindex(_du)) do i
                        J = _du[i]
                        if Δ isa AbstractVector ||
                            Δ isa DESolution ||
                            Δ isa AbstractVectorOfArray
                            v = Δ[i]
                        elseif Δ isa AbstractMatrix
                            v = @view Δ[:, i]
                        else
                            v = @view Δ[.., i]
                        end
                        if !(Δ isa NoTangent)
                            if u0 isa Number
                                ForwardDiff.value.(J'v)
                            else
                                ForwardDiff.value.(J'vec(v))
                            end
                        else
                            zero(u0)
                        end
                    end

                    if !(u0 isa Number)
                        push!(du0parts, vec(_du0))
                    end
                end

                if u0 isa Number
                    first(_du0)
                else
                    ArrayInterface.restructure(u0, reduce(vcat, du0parts))
                end
            end
            du0 = unthunk(du0)
        else
            du0 = zero(u0)
            @info "ForwardDiffSensitivity with `du0_sense = false` return zeros" maxlog = 1
        end
        dp = unthunk(dp)
        Δp_var = dp[1:num_p_var]
        Δp_invar = dp[(num_p_var + 1):(num_p_var + num_p_invar)]
        Δevent_times = dp[(num_p_var + num_p_invar + 1):end]
        Δcp = (
            u0=du0,
            p_var=Δp_var,
            p_invar=Δp_invar,
            event_times=Δevent_times,
            set_of_p_var=NoTangent(),
            kwargs=NoTangent(),
        )
        return (NoTangent(), Δcp, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return sol, forward_sensitivity_backpass
end
