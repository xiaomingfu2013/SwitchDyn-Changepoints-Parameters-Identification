using Catalyst
using ModelingToolkit
using LinearNoiseApproximation

function rs_Lotka_Volterra()
    symbol_ps = @parameters α, β, δ
    @variables t
    @species U(t), V(t)
    rxs = [
        Reaction(α, [U], [U], [1], [2]),
        Reaction(β, [U, V], [V], [1, 1], [2]),
        Reaction(δ, [V], nothing),
    ]
    @named rs_lv = ReactionSystem(rxs, t, [U, V], symbol_ps; combinatoric_ratelaws=false)
    return rs_lv
end


function get_Lotka_Volterra_model_change_delta(nt::NamedTuple)
    @unpack num_event = nt
    rn = rs_Lotka_Volterra()
    sys = LNASystem(rn)
    sys_ = sys.odesys
    sys_return = sys
    f_ = ODEFunction(sys_; jac=true)
    f_embed = f_.f.f_iip
    f_jac = f_.jac
    model_ = function lv_model_embed!(
        du,
        u,
        p,
        t;
        num_event=num_event,
        set_of_p_var=2,
        event_time=nothing,
        f_embed=f_embed,
    )
        p_var, p_invar, event_times = unpack_params(p, num_event, set_of_p_var)
        if event_time === nothing
            event_time = searchsortedfirst(event_times, t)
        end
        α, δ = p_var[event_time], p_var[(set_of_p_var - 1) * (num_event + 1) + event_time]
        β = p_invar[1]
        p_simplify = [α, β, δ] # make sure this is the right order as in the ODESystem parameters
        return f_embed(du, u, p_simplify, t)
    end
    model_jac =
        (du, u, p, t; event_time=nothing) ->
            model_(du, u, p, t; event_time=event_time, f_embed=f_jac)
    model = ODEFunction(
        (du, u, p, t; event_time=nothing) -> model_(du, u, p, t; event_time=event_time);
        jac=model_jac,
    )
    return model, sys_return
end

function get_Lotka_Volterra_model_change_beta(nt::NamedTuple)
    @unpack num_event = nt
    rn = rs_Lotka_Volterra()
    sys = LNASystem(rn)
    sys_ = sys.odesys
    sys_return = sys
    f_ = ODEFunction(sys_; jac=true)
    f_embed = f_.f.f_iip
    f_jac = f_.jac
    model_ = function lv_model_embed!(
        du,
        u,
        p,
        t;
        num_event=num_event,
        set_of_p_var=2,
        event_time=nothing,
        f_embed=f_embed,
    )
        p_var, p_invar, event_times = unpack_params(p, num_event, set_of_p_var)
        if event_time === nothing
            event_time = searchsortedfirst(event_times, t)
        end
        α, β = p_var[event_time], p_var[(set_of_p_var - 1) * (num_event + 1) + event_time]
        δ = p_invar[1]
        p_simplify = [α, β, δ] # make sure this is the right order as in the ODESystem parameters
        return f_embed(du, u, p_simplify, t)
    end
    model_jac =
        (du, u, p, t; event_time=nothing) ->
            model_(du, u, p, t; event_time=event_time, f_embed=f_jac)
    model = ODEFunction(
        (du, u, p, t; event_time=nothing) -> model_(du, u, p, t; event_time=event_time);
        jac=model_jac,
    )
    return model, sys_return
end
