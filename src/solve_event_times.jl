using Flux
"""
    ChangePointParameters
Input Arguments:
- u0: initial condition
- p_var: parameters that are varied
- p_invar: parameters that are not varied
- event_times: event times
- set_of_p_var: number of parameters that are varied
- kwargs: keyword arguments
"""
struct ChangePointParameters{T1,K1}
    u0::T1
    p_var::T1
    p_invar::T1
    event_times::T1
    set_of_p_var::Int
    kwargs::K1
end

function ChangePointParameters(u0, p_var, p_invar, event_times)
    return ChangePointParameters{typeof(u0),Dict}(
        u0, p_var, p_invar, event_times, 1, Dict()
    )
end
function ChangePointParameters(u0, p_var, p_invar, event_times, set_of_p_var)
    return ChangePointParameters{typeof(u0),Dict}(
        u0, p_var, p_invar, event_times, set_of_p_var, Dict()
    )
end

Flux.@functor ChangePointParameters
function Flux.trainable(cp::ChangePointParameters)
    return (u0=cp.u0, p_var=cp.p_var, p_invar=cp.p_invar, event_times=cp.event_times)
end

# when the parameters are stored in a matrix, but the user can store it in a vector for good performance
# ChangePointParameters(u0, p_var, p_invar, event_times, set_of_p_var, kwargs) = ChangePointParameters(u0, p_var, p_invar, event_times, set_of_p_var, kwargs)

"""
    solve_event_times(cp, ts, prob, alg, sensealg, solvesettings)
Solve the ODE with the given parameters and return the solution at the given time points.
Input Arguments:
- cp : ChangePointParameters
- ts : time points
- prob : ODE problem
- alg : ODE solver
- sensealg : sensitivity algorithm
- kwargs : keyword arguments
- du0_sense : whether to compute the sensitivity of the initial condition
"""
function solve_event_times(
    cp::ChangePointParameters, ts, prob, alg, sensealg; kwargs...
)
    @unpack u0, p_var, p_invar, event_times = cp
    ps = [p_var; p_invar; event_times]
    if prob.p isa AbstractArray
        @assert length(ps) == length(prob.p)
    end
    @assert length(u0) == length(prob.u0)

    prob_ = remake(prob; u0=u0, p=ps)
    tstops_ = haskey(kwargs, :tstops) ? sort([event_times; kwargs[:tstops]]) : event_times
    sol = solve(prob_, alg; saveat=ts, kwargs..., tstops=tstops_)
    return sol
end
