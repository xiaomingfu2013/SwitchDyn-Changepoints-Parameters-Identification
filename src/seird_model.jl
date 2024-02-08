function rs_seird()
    symbol_ps = @parameters λ, σ, γ, μ, N₀
    @variables t
    @species S(t), E(t), I(t), R(t), D(t), cumI(t)
    rxs = [
        Reaction(λ / N₀, [S, I], [I, E]),
        Reaction(σ, [E], [I, cumI]),
        Reaction(γ, [I], [R]),
        Reaction(μ, [I], [D]),
    ]
    @named rs_seird = ReactionSystem(
        rxs, t, [S, E, I, R, D, cumI], symbol_ps; combinatoric_ratelaws=false
    )
    lna_sys = LNASystem(rs_seird)
    return lna_sys
end
function get_seird_lna_model(nt::NamedTuple)
    lna_sys = rs_seird()
    f_ = ODEFunction(lna_sys.odesys)
    f_embed = f_.f.f_iip
    nt = nt
    model_ = function sir_lna_model_embed!(du, u, p, t; nt=nt, f_embed=f_embed)
        @unpack num_event, len_split_ps, len_split_event_times, len_split_event_times_cum = nt
        p_var, p_invar, event_times = split_vector(p, len_split_ps)
        event_time_idxs = find_event_time(t, event_times, len_split_event_times)
        event_time_λ, event_time_μ = event_time_idxs

        λ = p_var[event_time_λ]
        μ = p_var[len_split_event_times_cum[1] + event_time_μ]

        σ, γ = p_invar[1:2]
        S, E, I, R, D, cumI = u[1:6]
        N0 = S + E + I + R
        p_simplify = [λ, σ, γ, μ, N0] # make sure this is the right order as in the ODESystem parameters
        return f_embed(du, u, p_simplify, t)
    end
    return model_, lna_sys
end
