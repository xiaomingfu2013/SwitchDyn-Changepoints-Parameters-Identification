using Test
using DifferentialEquations
using Zygote
using LinearAlgebra
include("utils.jl")
include("../src/adjoint_method.jl")


t_break_true = [3.0, 5.0]
p_var = [1.5, 1.0, 3.0] # beta1, beta2, beta3
p_invar = [2.0] # alpha
p_true = vcat(p_var, p_invar) # beta1, beta2, beta3, alpha
u0_true = [1.0, 1.0]

nt = (len_split_ps=(3, 1, 2), )
function lv_new!(du, u, p, t; nt=nt)
    x, y = u
    β, α = extract_params(p, t, nt)
    du[1] = α * x - x * y
    du[2] = -y + β * x * y
    return nothing
end

function extract_params(p, t, nt)
    p_var, p_invar, event_times = split_vector(p, nt.len_split_ps)
    event_time = searchsortedfirst(event_times, t)
    β = p_var[event_time]
    α = p_invar[1]
    return β, α
end


tspan = (0.0, 10.0)
ts = 0.0:0.1:10.0
prob = ODEProblem(lv_new!, u0_true, tspan)
alg = Tsit5()
solvesettings = Dict(:abstol => 1e-14, :reltol => 1e-14)
sol = solve(
    prob,
    alg;
    tstops=t_break_true,
    saveat=ts,
    p=[p_true; t_break_true],
    solvesettings...,
)
true_data = Array(sol)

# the goal is to avoid using callback, but to update the gradient to the event time points in the backward pass directly
dgdu(out, u, p, t, i) = (out .= 2 * (u .- true_data[:, i]))




function loss_function(u0, p, event_time; ts=ts, sensealg)
    ps = [p; event_time]
    p_var, p_invar, event_times = split_vector(ps, (3, 1, 2))
    cp = ChangePointParameters(u0, p_var, p_invar, event_times)
    sol = solve_event_times(
        cp, ts, prob, Tsit5(), sensealg; solvesettings...
    )
    return sum(abs2, (true_data .- Array(sol)))
end



u0 = [2.0, 2.0]
t_break = [3.5, 4.5]
p = [1.2, 1.5, 3.0, 2.5]
sensealg1 = ForwardDiffSensitivity(du0_sense=true)
sensealg2 = SimpleAdjoint(:EnzymeVJP)
sensealg3 = SimpleAdjoint(:ZygoteVJP)
argument_combination = Iterators.product((u0, u0_true), (p, p_true), (t_break, t_break_true))

num_paramm = length(p) + length(t_break) + length(u0)

res_total = []
@testset "Accuracy" for args in argument_combination
    res_list = []
    for sensealg in (sensealg1, sensealg2, sensealg3)
        res = vcat(
            Zygote.gradient(
                (u0, p, event_time) ->
                    loss_function(u0, p, event_time; sensealg=sensealg),
                args...,
            )...,
        )
        push!(res_list, res)
        push!(res_total, res)
    end
    # test with rtol = 1e-5
    for i in 1:num_paramm
        @test res_list[2][i] ≈ res_list[1][i] atol=1e-2
        @test res_list[2][i] ≈ res_list[1][i] atol=1e-2
    end
end
