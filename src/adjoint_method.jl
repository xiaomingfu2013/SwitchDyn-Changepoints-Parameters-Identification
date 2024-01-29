using ChainRulesCore
using Adapt
using LinearAlgebra
using DiffEqBase
using FunctionWrappersWrappers
using SciMLBase
import SciMLBase: unwrapped_f
using ForwardDiff
using UnPack
using Flux
using ArrayInterface

include("solve_event_times.jl")
include("forwarddiffsense.jl")
include("reverse_type_adjoint_simple.jl")


"""
    compute_shift_timepoint_grad(u_cur, λ, f_simple, p1, p2)
Compute the gradient of the loss function with respect to the shift time point.
- u_cur : current state
- λ : adjoint variable
- f_iip : the in-place function f(du, u, p, t)
- p : parameters
- t : time point
"""
function compute_changepoint_derivative(u_cur, f_iip, p, t)
    du1 = similar(u_cur)
    du2 = similar(u_cur)
    f_iip(du1, u_cur, p, t)
    f_iip(du2, u_cur, p, t + eps(t)) # next time point, this is should be treated carefully
    return du2 - du1
end

function compute_shift_timepoint_grad_which(u_cur, λ, f_iip, p, t)
    Δu = compute_changepoint_derivative(u_cur, f_iip, p, t)
    dLdt = -LinearAlgebra.dot(λ, Δu)
    return dLdt
end
