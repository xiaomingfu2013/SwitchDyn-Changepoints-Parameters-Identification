precompile_time = @elapsed begin
    using DifferentialEquations
    using Plots
    using DataStructures # OrderedDict
    using Distributions
    using ParameterHandling
    const PH = ParameterHandling
    using Random
    using Turing
    using StatsPlots
    using MCMCChains
    using Serialization
    using DrWatson
    using Optimization, OptimizationOptimisers, OptimizationOptimJL, OptimizationNLopt
    using Zygote
    using CSV
    Turing.setadbackend(:zygote)
    theme(:vibrant)
    include("../../../src/adjoint_method.jl")
    include("../../../src/ode_models.jl")
    include("../../../src/utils.jl")
    include("../../../src/turing_models.jl")
end

i = 1
rng = MersenneTwister(i)
var_scaling = 10.0
logpdf_custom =
    (mean, var, data) -> likelihood_lna_dist(mean, var_scaling * sqrt.(var), data)


@info "Precompile finish with time: $precompile_time"
model_name = "get_Lotka_Volterra_model_change_beta"

@info "Model $model_name"
priors_sample = (
    p_var=(
        α=[
            LogNormal(log(0.8), 0.1),
            LogNormal(log(0.6), 0.1),
        ],
        β=[
            LogNormal(log(0.5), 0.1),
            LogNormal(log(0.4), 0.1),
        ],
    ),
    p_invar=(δ=PH.fixed(0.4),),
    event_times=(
        τ₁=Uniform(2.2, 2.8),
    ),
    scaling=(
        SCALING_p_var=(SCALING_α=PH.fixed(ones(2)), SCALING_β=PH.fixed(1e-2 * ones(2))),
        SCALING_p_invar=PH.fixed([1.0]),
        SCALING_event_times=PH.fixed(1e1 * ones(1)),
    ),
)

@info "Setting up initial conditions for synthetic ODE run"
## ====================
# randomly draw a set an ODE initial conditions
## ====================
orig_pt_sample = sample_tuple(rng, priors_sample)
# set up initial conditions
u0 = [120.0, 140.0]
duration = 50.0
tspan = (0.0, duration)
ts = tspan[1]:0.1:tspan[2]
alg = AutoVern7(Rodas5())
sensealg = ForwardDiffSensitivity()
# sensealg = SimpleAdjoint(:EnzymeVJP)
solvesettings = Dict()
set_of_p_var = length(orig_pt_sample.p_var)
p_var_sample, p_invar_sample, event_times_sample = build_ps(
    orig_pt_sample; return_array=false
)
ps_ = [p_var_sample..., p_invar_sample..., event_times_sample...]

##====================
# set up the ODE model
##====================
num_event = length(orig_pt_sample.event_times)
nt = (num_event=num_event,)
model, exp_sys = odemodel_selection(model_name)(nt);

# lna_sys.odesys.eqs
u0_expanded = expand_initial_conditions(exp_sys, u0)
odemodel_embed = model
prob_embed_sample = ODEProblem(odemodel_embed, u0_expanded, tspan, ps_)
sol_embed_sample = solve(
    prob_embed_sample, alg; saveat=ts, tstops=event_times_sample, solvesettings...
)

cp_sample = ChangePointParameters(
    u0_expanded, p_var_sample, p_invar_sample, event_times_sample, set_of_p_var, Dict()
)
sol_sample = solve_event_times(
    cp_sample, ts, prob_embed_sample, alg, sensealg; solvesettings...
)

@assert sol_embed_sample[end] == sol_sample[end]

##====================
# set up the synthetic data
##====================
mean_U_idx, var_U_idx = find_states_cov_number(1, exp_sys)
mean_V_idx, var_V_idx = find_states_cov_number(2, exp_sys)

synthetic_data_noisy = (
    rand(
        rng,
        MvNormal(
            sol_embed_sample[mean_U_idx, :],
            Diagonal(var_scaling * sqrt.(sol_embed_sample[var_U_idx, :])),
        ),
    ),
    rand(
        rng,
        MvNormal(
            sol_embed_sample[mean_V_idx, :],
            Diagonal(var_scaling * sqrt.(sol_embed_sample[var_V_idx, :])),
        ),
    ),
)

priors = (
    p_var=(
        α=priors_sample.p_var.α,
        β=priors_sample.p_var.β,
    ),
    p_invar=priors_sample.p_invar,
    event_times=(τ₁=Uniform(0.0, 5.0),),
    scaling=priors_sample.scaling,
)

guess_init_pt = sample_mean_tuple(priors)
guess_params_flatten, unflatten = PH.value_flatten(guess_init_pt);
## ====================
# set up the turing model
## ====================
loss_function_kwargs = (
    data_true=synthetic_data_noisy,
    sensealg=sensealg,
    prob=prob_embed_sample,
    alg=alg,
    ts=ts,
    set_of_p_var=set_of_p_var,
    solvesettings=solvesettings,
    u0_expanded=u0_expanded,
    mean_idxs=[mean_U_idx, mean_V_idx],
    var_idxs=[var_U_idx, var_V_idx],
    unflatten=unflatten,
);

turing_model = turing_change_point(
    priors; loss_function_kwargs=loss_function_kwargs, logpdf_custom=logpdf_custom
);


include("../../../src/abc_module.jl")
loss_function(p) = loss_function_turing(p, loss_function_kwargs=loss_function_kwargs, logpdf_custom=logpdf_custom)

loss_function(guess_params_flatten)

new_names = extend_event_time_symbols(priors)
sampler = Turing.NUTS(0.65; max_depth=10, Δ_max=1000.0, init_ϵ=5e-3)
sample_length = 1000
rng = MersenneTwister(i)
chain = Turing.sample(
    rng,
    turing_model,
    sampler,
    sample_length;
    progress=true,
    init_params=guess_params_flatten,
)

# result using ForwardDiffSensitivity()
# Chains MCMC chain (1000×17×1 Array{Float64, 3}):
# Iterations        = 501:1:1500
# Number of chains  = 1
# Samples per chain = 1000
# Wall duration     = 65.7 seconds
# Compute duration  = 65.7 seconds
# parameters        = pprior[1], pprior[2], pprior[3], pprior[4], pprior[5]
# internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

change_names = Dict(chain.name_map.parameters .=> new_names)
new_chain = replacenames(chain, change_names)

new_chain.name_map.parameters # check the indices of the fixed variables

new_chain_ = scale_chain(new_chain, unflatten, [5]) # 5, 6 are the indices of δ and τ

new_chain_df = DataFrames.DataFrame(new_chain_)

summarize(new_chain_)
plt_chn = plot(new_chain_)
true_ps_list = build_ps(orig_pt_sample)

hpd_res = quantile(chain; q=[0.5])[:, 2] # 0.5 is the median
estim_pt = unflatten(hpd_res)

p_var2, p_invar2, event_times2 = build_ps(estim_pt; return_array=false)
cp_rec = ChangePointParameters(
    u0_expanded, p_var2, p_invar2, event_times2, set_of_p_var, Dict()
)

sol_rec = solve_event_times(cp_rec, ts, prob_embed_sample, alg, sensealg; solvesettings...)

plt_rec = single_plot_solutions(
    sol_rec;
    loss_function_kwargs=loss_function_kwargs,
    loss=round(loss_estim_params; digits=1),
    event_times=event_times2,
    event_times_true=event_times_sample,
    name="reconstructed from median value of posterior",
)

true_ps = wrapper_scaled_param_dict(orig_pt_sample)
true_ps_list = build_ps(orig_pt_sample)
estim_list = build_ps(estim_pt)
param_dict = DataFrames.DataFrame(
    "symbol" => extend_event_time_symbols(priors; all_symbols=true),
    "true" => true_ps_list,
    "estim" => estim_list,
    "relative_error" => abs.(estim_list .- true_ps_list) ./ true_ps_list,
)


loss_function(guess_params_flatten)
loss_estim_params = loss_function(hpd_res)

abc_setup = setup_optimization_abc(priors, loss_function; target=4600.0,  nparticles=1000, ϵ1=1e5, MaxFuncEvals=1e6, α = 0.2, convergence = 0.0005)
# abc_setup.simfunc(guess_params_flatten, nothing, nothing)

abc_data = hcat(synthetic_data_noisy...)
@time res_abc = runabc(abc_setup, abc_data; parallel = true, progress=true)

# Preparing to run in parallel on 4 processors
# Preparing to run in parallel on 4 processors
# 106.056492 seconds (5.58 G allocations: 484.607 GiB, 30.45% gc time, 0.03% compilation time)
# Total number of simulations: 1.07e+06
# Cumulative number of simulations = [1004, 33409, 262625, 827767, 994298, 1071682]
# Acceptance ratio: 9.33e-04
# Tolerance schedule = [5188.98, 4896.34, 4791.22, 4705.59, 4651.84, 4624.57]

# Median (95% intervals):
# Parameter 1: 0.83 (0.80,0.86)
# Parameter 2: 0.64 (0.58,0.69)
# Parameter 3: 0.47 (0.46,0.49)
# Parameter 4: 0.40 (0.36,0.44)
# Parameter 5: 2.59 (2.38,3.06)

# Total number of simulations: 1.07e+06
# Cumulative number of simulations = [1004, 33409, 262625, 827767, 994298, 1071682]
# Acceptance ratio: 9.33e-04
# Tolerance schedule = [5188.98, 4896.34, 4791.22, 4705.59, 4651.84, 4624.57]

# Median (95% intervals):
# Parameter 1: 0.83 (0.80,0.86)
# Parameter 2: 0.64 (0.58,0.69)
# Parameter 3: 0.47 (0.46,0.49)
# Parameter 4: 0.40 (0.36,0.44)
# Parameter 5: 2.59 (2.38,3.06)


# Total number of simulations: 1.07e+06
# Cumulative number of simulations = [1004, 33409, 262625, 827767, 994298, 1071682]
# Acceptance ratio: 9.33e-04
# Tolerance schedule = [5188.98, 4896.34, 4791.22, 4705.59, 4651.84, 4624.57]

# Median (95% intervals):
# Parameter 1: 0.83 (0.80,0.86)
# Parameter 2: 0.64 (0.58,0.69)
# Parameter 3: 0.47 (0.46,0.49)
# Parameter 4: 0.40 (0.36,0.44)
# Parameter 5: 2.59 (2.38,3.06)

# Total number of simulations: 1.07e+06
# Cumulative number of simulations = [1004, 33409, 262625, 827767, 994298, 1071682]
# Acceptance ratio: 9.33e-04
# Tolerance schedule = [5188.98, 4896.34, 4791.22, 4705.59, 4651.84, 4624.57]

# Median (95% intervals):
# Parameter 1: 0.83 (0.80,0.86)
# Parameter 2: 0.64 (0.58,0.69)
# Parameter 3: 0.47 (0.46,0.49)
# Parameter 4: 0.40 (0.36,0.44)
# Parameter 5: 2.59 (2.38,3.06)
plot(res_abc)

res_abc

writeoutput(res_abc)

output = DataFrame(res_abc.parameters, new_chain.name_map.parameters)
output[!, :loss] .= map(x -> x.distance, res_abc.particles)
output[!, :weights] .= map(x -> x.weight, res_abc.particles)
output




save_margs = OrderedDict(
    "seed" => i,
    "model_name" => model_name,
    "u0" => u0,
    "ts" => ts,
    "tspan" => tspan,
    "priors" => priors,
    "param_dict" => param_dict,
    #"turing_model" => turing_model,
    # "loss_function_kwargs" => loss_function_kwargs,
    "var_scaling" => var_scaling,
)

push!(
    save_margs,
    "sol_rec" => DataFrames.DataFrame(sol_rec),
    "scaled_chain" => new_chain_,
    "unscaled_chain" => chain,
    "sampler_info" => sampler,
    "sample_length" => sample_length,
    "abc_res" => output,
)

@quickactivate "SwitchDyn-Changepoints-Parameters-Identification"
save_path = "plots_scripts/lv/"


serialize(joinpath(save_path, "lv_results_$(i)_abc_res.jls"), save_margs)
using JSON

post_df = DataFrames.DataFrame(new_chain_)

json_data = Dict(
    "sol_rec" => DataFrames.DataFrame(sol_rec),
    "mean_idxs" => loss_function_kwargs.mean_idxs,
    "var_idxs" => loss_function_kwargs.var_idxs,
    "data_true" => loss_function_kwargs.data_true,
    "param_dict" => DataFrames.DataFrame(param_dict),
    "post_df" => post_df,
    "param_list" => new_names,
    "abc_res" => output,
)

json_string = JSON.json(json_data)

open("$save_path/lv_data_$(i)_abc_compare.json", "w") do f
    return JSON.print(f, json_string)
end



# easy visualization using Julia
abc_res_scaled = deepcopy(output)
abc_res_scaled[!, :β₁] = output[!, :β₁] * 1e-2
abc_res_scaled[!, :β₂] = output[!, :β₂] * 1e-2
abc_res_scaled[!, :τ₁] = output[!, :τ₁] * 1e1

abc_res_simple = abc_res_scaled[!, new_names]
weights = abc_res_scaled[!, :weights]

ours_res_simple = post_df[!, new_names]

# plot the two histogram together
using StatsPlots
plt_list = []
for i in new_names
    plt = histogram(
        abc_res_simple[!, i],
        weights=weights,
        label="ABC",
        xlabel=i,
        ylabel="Density",
        bins=20,
        alpha=0.5,
        legend=:topleft,
        normalize=:pdf,
    )
    histogram!(
        ours_res_simple[!, i],
        label="Ours",
        bins=20,
        alpha=0.5,
        legend=:topleft,
        normalize=:pdf,
    )
    push!(plt_list, plt)
end

plot(plt_list..., layout=(3, 2), size=(800, 600))
# abc_res_scaled
