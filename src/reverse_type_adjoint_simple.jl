using UnPack, ChainRulesCore
using DiffEqBase
using Enzyme
using SciMLBase
import SciMLBase: unwrapped_f
using Adapt

"""
    default autojacvec is `:ZygoteVJP` with better compatibility
    use `:EnzymeVJP` for better performance. However, Enzyme is still a work in progress pakage, therefore may not be stable.
"""
struct SimpleAdjoint
    autojacvec::Symbol
end

SimpleAdjoint() = SimpleAdjoint(:ZygoteVJP)



include("solve_event_times.jl")


struct AdjointFunctor{C,solType,uType,pType,fType,sType}
    y::uType
    forward_sol::solType
    prob::pType
    diffcache::C
    f::fType
    sensealg::sType
end

"""
    customized chain rule for `solve_event_times` function
"""
function ChainRulesCore.rrule(::typeof(solve_event_times), cp, ts, prob, alg, sensealg::SimpleAdjoint; kwargs...)
    @info "solve_event_times using SimpleAdjoint" maxlog = 1
    @unpack u0, p_var, p_invar, event_times = cp

    forward_sol = solve_event_times(cp, eltype(ts)[], prob, alg, sensealg; kwargs...)
    only_end = length(ts) == 1 && ts[1] == prob.tspan[2]

    function pullback(Δ)
        function dgdu_iip(_out, u, p, t, i)
            outtype = typeof(_out) <: SubArray ?
                      DiffEqBase.parameterless_type(_out.parent) :
                      DiffEqBase.parameterless_type(_out)
            if only_end
                eltype(Δ) <: NoTangent && return
                if typeof(Δ) <: AbstractArray{<:AbstractArray} && length(Δ) == 1 && i == 1
                    vec(_out) .= adapt(outtype, vec(Δ[1]))
                else
                    vec(_out) .= adapt(outtype, vec(Δ))
                end
            else
                !Base.isconcretetype(eltype(Δ)) &&
                    (Δ[i] isa NoTangent || eltype(Δ) <: NoTangent) && return
                if typeof(Δ) <: AbstractArray{<:AbstractArray} || typeof(Δ) <: DESolution
                    x = Δ[i]
                    vec(_out) .= vec(x)
                else
                    vec(_out) .= vec(adapt(outtype, reshape(Δ, prod(size(Δ)[1:(end-1)]), size(Δ)[end])[:, i]))
                end
            end
        end
        num_p_var = length(p_var)
        num_p_invar = length(p_invar)
        du0, adj = backward_pass_with_event_times(prob, forward_sol, dgdu_iip, ts, cp, alg, sensealg; kwargs...)

        Δp_var = adj[1:num_p_var]
        Δp_invar = adj[num_p_var+1:num_p_var+num_p_invar]
        Δevent_times = adj[num_p_var+num_p_invar+1:end]

        Δcp = (u0=du0, p_var=Δp_var, p_invar=Δp_invar, event_times=Δevent_times, set_of_p_var=NoTangent(), kwargs=NoTangent())

        return (NoTangent(), Δcp, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return forward_sol(ts), pullback
end



"""
    backward_pass_with_event_times(prob, sol, dgdu_discrete, ts, u0, p_var, p_invar, event_times, alg, sensealg)

- prob : ODEProblem
- sol : forward solution
- dgdu_discrete : discrete : e.g. dg(out, u, p, t, i) = (out .= 2*(u .- true_data[:,i]))
- ts : time points to be compared with the true data
- u0 : initial condition
- p_var : parameters that are varied
- p_invar : parameters that are invariant
- event_times : event times
- alg : algorithm for solving the ODEProblem
- sensealg : sensitivity algorithm
"""
function backward_pass_with_event_times(prob, forward_sol, dgdu_discrete, ts, cp, alg, sensealg::SimpleAdjoint; kwargs...)
    cb_calibrate = reverse_callback(prob, forward_sol, cp, sensealg)
    # if kwargs has callback, then add them together
    cb_kwarg = get(kwargs, :callback, CallbackSet())
    cb_merge = CallbackSet(cb_calibrate, cb_kwarg)
    du0, dp = adjoint_sensitivities_direct(forward_sol, alg; ts=ts, dgdu_discrete=dgdu_discrete, callback=cb_merge, sensealg=sensealg, kwargs...)

    idx_start = length(cp.p_var) + length(cp.p_invar) + 1
    copyto!(dp, idx_start, cb_calibrate.affect!.user_affect!.dLdt_)
    return du0, dp
end


function (s::AdjointFunctor)(da, a, p, t)
    @unpack y, forward_sol, sensealg = s
    num_u0 = length(y)
    λ = @view a[1:num_u0]

    dλ = @view da[1:num_u0]
    dgrad = @view da[num_u0+1:end]

    y = forward_sol(t, continuity=:right) # update the solution at time t
    vjp = sensealg.autojacvec # symbol
    vecjacobian!(dλ, y, λ, p, t, s, vjp; dgrad=dgrad)
    return nothing
end

function adjoint_problem_direct(forward_sol, sensealg; kwargs...)
    prob = forward_sol.prob
    ps = prob.p
    tspan = prob.tspan

    # make sure prob.f is inplace function
    pf = let f = prob.f
        if DiffEqBase.isinplace(prob)
            function (out, u, _p, t)
                f(out, u, _p, t)
                nothing
            end
        else
            function (out, u, _p, t)
                out .= f(u, _p, t)
                nothing
            end
        end
    end

    y = copy(forward_sol.u[end])
    if typeof(prob.p) <: DiffEqBase.NullParameters
        _p = similar(y, (0,))
    else
        _p = prob.p
    end
    if typeof(prob.p) <: DiffEqBase.NullParameters
        paramjac_config = zero(y), prob.p, zero(y), zero(y), zero(y)
    else
        paramjac_config = zero(y), zero(_p), zero(y), zero(y), zero(y)
    end
    sense_functor = AdjointFunctor(y, forward_sol, prob, paramjac_config, pf, sensealg)
    # initial condition for adjoint equation: zeros(num_u0) with zeros(length(ps))
    ini = vcat(zero(prob.u0), zero(ps))
    odefunc = ODEFunction{true,true}(sense_functor)
    adprob = ODEProblem(
        odefunc,
        ini,
        reverse(tspan),
        ps,
    )
    # return sense_functor
    return adprob
end

function adjoint_sensitivities_direct(forward_sol, alg; ts=nothing, sensealg=SimpleAdjoint(), dgdu_discrete=nothing, callback=CallbackSet(), kwargs...)

    num_u0 = length(forward_sol.u[end])
    adprob = adjoint_problem_direct(forward_sol, sensealg; kwargs...)
    sense_functor = adprob.f.f
    cb_dg = dg_callback(sense_functor, dgdu_discrete, ts)
    #!! be careful about the order of callback
    # the correct order is reverse call then dg dg_callback, because we need to use λ before the jump
    cb = CallbackSet(callback, cb_dg)
    adsol = solve(
        adprob,
        alg;
        save_everystep=false,
        save_start=false,
        saveat=eltype(ts)[],
        tstops=ts,
        callback=cb,
        kwargs...
    )
    return adsol[end][1:num_u0], adsol[end][num_u0+1:end]
end


function dg_callback(sense_functor, dgdu_discrete, ts)

    affect! =
        let cur_time = Ref(length(ts)),
            ts = ts,
            dgdu_discrete = dgdu_discrete

            sense_functor = sense_functor

            function (integrator)
                p, u = integrator.p, integrator.u
                y = sense_functor.y
                gᵤ = similar(y)
                dgdu_discrete(gᵤ, y, p, ts[cur_time[]], cur_time[])
                u[1:length(y)] .+= gᵤ
                u_modified!(integrator, true)
                cur_time[] -= 1
                return nothing
            end
        end
    cb_dg = PresetTimeCallback(ts, affect!; save_positions=(false, false))
    return cb_dg
end


function reverse_callback(prob, forward_sol, cp, ::SimpleAdjoint)
    @unpack u0, p_var, p_invar, event_times, set_of_p_var, kwargs = cp
    numu0 = length(u0)

    f_iip = prob.f.f
    calibrate_grad = let dLdt_ = Vector{eltype(u0)}(undef, length(event_times)), forward_sol = forward_sol, f_iip = f_iip, event_times = event_times
        function (integrator)
            λ = integrator.u[1:numu0]
            y = forward_sol(integrator.t)
            event_time_idxs = findall(isequal(integrator.t), event_times)
            dLdt = compute_shift_timepoint_grad_which(y, λ, f_iip, integrator.p, integrator.t)
            for event_time_idx in event_time_idxs
                dLdt_[event_time_idx] = dLdt
            end
            return nothing
        end
    end

    cb_calibrate = PresetTimeCallback(event_times, calibrate_grad, save_positions=(false, false))
    return cb_calibrate
end



vecjacobian!(dλ, y, λ, p, t, s, alg::Symbol; dgrad=nothing) = vecjacobian!(dλ, y, λ, p, t, s, Val{alg}, dgrad=dgrad)

function vecjacobian!(dλ, y, λ, p, t, s, ::Type{Val{:ZygoteVJP}}; dgrad=dgrad)
    f = unwrapped_f(s.f.f)
    _, back = Zygote.pullback(y, p) do u, p
        out_ = Zygote.Buffer(similar(u))
        f(out_, u, p, t)
        vec(copy(out_))
    end
    dLdu, dLdp = back(-λ) # dLdu = -λ'*dfdu, dLdp = -λ'*dfdp
    copyto!(dλ, dLdu)
    dgrad !== nothing && copyto!(dgrad, dLdp)
    return nothing
end

# use Enzyme to compute the jacobian vector product
function vecjacobian!(dλ, y, λ, p, t, s, ::Type{Val{:EnzymeVJP}}; dgrad=nothing)
    f = unwrapped_f(s.f.f)
    tmp1, tmp2, tmp3, tmp4, ytmp = s.diffcache
    tmp1 .= 0 # should be removed for dλ
    ytmp .= y
    dup = if !(tmp2 isa DiffEqBase.NullParameters)
        tmp2 .= 0
        Enzyme.Duplicated(p, tmp2)
    else
        Enzyme.Const(p)
    end
    tmp3 .= 0

    vec(tmp4) .= vec(-λ) # !! notice the negative sign

    Enzyme.autodiff(
        Enzyme.Reverse,
        f,
        Enzyme.Const,
        Enzyme.Duplicated(tmp3, tmp4),
        Enzyme.Duplicated(ytmp, tmp1),
        dup,
        Enzyme.Const(t)
    )

    dλ .= tmp1
    dgrad !== nothing && (dgrad[:] .= vec(tmp2))


    return nothing
end
