using Catalyst
using LinearNoiseApproximation
##======
# with kG1, kG2, τ changing in the jump process
##======
function rs_stoch_gene_poisson_true_post()
    symbol_ps = @parameters kG, kG_, dP, τ # this is for the jump system callback
    @variables t
    @species P(t)
    rxs = [Reaction(kG, nothing, [P]), Reaction(dP, [P], nothing)]
    @named rs_stoch = ReactionSystem(rxs, t, [P], symbol_ps; combinatoric_ratelaws=false)
    lna_sys = LNASystem(rs_stoch)
    return lna_sys
end
function get_stoch_gene_model_poisson_true_post(nt::NamedTuple)
    @unpack num_event = nt
    lna_sys = rs_stoch_gene_poisson_true_post()
    f_ = ODEFunction(lna_sys.odesys; jac=true)
    f_embed = f_.f.f_iip
    f_jac = f_.jac
    model_ = function stoch_gene_lna_model_embed!(
        du, u, p, t; num_event=num_event, set_of_p_var=1, f_embed=f_embed
    )
        p_var, p_invar, event_times = unpack_params(p, num_event, set_of_p_var)
        event_time_idx = searchsortedfirst(event_times, t)
        kG = p_var[event_time_idx]
        dP = p_invar[1]
        # make sure this is the right order as in the ODESystem parameters
        p_simplify = [kG, 0.0, dP, 0.0] # kG, kG2, dP, τ, kG2 will not be use, same as for τ
        return f_embed(du, u, p_simplify, t)
    end
    model_jac = (du, u, p, t) -> model_(du, u, p, t; f_embed=f_jac)
    model = ODEFunction(model_; jac=model_jac)
    return model, lna_sys
end

function event_callback_reaction_jumpprob_all_params_in_condition(
    num_event, set_of_p_var; save_positions=(false, false)
)
    # num_event = length(event_times)
    # set_of_p_var
    function condition(u, t, integrator)
        p_var, p_invar, event_times = unpack_params(integrator.p, num_event, set_of_p_var)
        return t in event_times
    end
    function affect!(integrator)
        p_var, p_invar, event_times = unpack_params(integrator.p, num_event, set_of_p_var)
        event_time = searchsortedfirst(event_times, integrator.t) + 1
        for i in 1:set_of_p_var
            integrator.p[(i - 1) * (num_event + 1) + 1] = p_var[(i - 1) * (num_event + 1) + event_time] # always change the first one as the input for the jump system
        end
        return reset_aggregated_jumps!(integrator) # reset the jump rates, this is very important https://docs.sciml.ai/JumpProcesses/dev/faq/#How-do-I-use-callbacks-with-ConstantRateJump-or-MassActionJump-systems?
    end
    cb = DiscreteCallback(condition, affect!; save_positions=save_positions)
    return cb
end
