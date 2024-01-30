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

# for i in 1:100
local_machine = isempty(ARGS)
i = local_machine ? 1 : parse(Int, ARGS[1])
rng = MersenneTwister(i)



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
pairs(orig_pt_sample)
# set up initial conditions
u0 = [120.0, 140.0]
duration = 50.0
tspan = (0.0, duration)
ts = tspan[1]:0.1:tspan[2]
alg = AutoVern7(Rodas5())
sensealg = ForwardDiffSensitivity()
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
var_scaling = 10.0
logpdf_custom =
    (mean, var, data) -> likelihood_lna_dist(mean, var_scaling * sqrt.(var), data)
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
    event_times=(τ₁=Uniform(0.0, 5.0),),
    scaling=(
        SCALING_p_var=(SCALING_α=PH.fixed(ones(2)), SCALING_β=PH.fixed(1e-2 * ones(2))),
        SCALING_p_invar=PH.fixed([1.0]),
        SCALING_event_times=PH.fixed(1e1 * ones(1)),
    ),
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
optim_prob = optim_problem(turing_model, MAP(); autoad=Optimization.AutoZygote());
loss_function(p::Vector) = optim_prob.prob.f.f(p, SciMLBase.NullParameters())
loss_function(guess_params_flatten)


## Turing Chain
new_names = extend_event_time_symbols(priors)
sampler = Turing.NUTS(0.65; max_depth=10, Δ_max=1000.0, init_ϵ=5e-3)
sample_length = 500
rng = MersenneTwister(i)
chain = Turing.sample(
    rng,
    turing_model,
    sampler,
    sample_length;
    progress=true,
    init_params=guess_params_flatten
)
change_names = Dict(chain.name_map.parameters .=> new_names)
new_chain = replacenames(chain, change_names)
new_chain_ = scale_chain(new_chain, unflatten, [5]) # 5 is the index of δ
summarize(new_chain_)
plt_chn = plot(new_chain_)
true_ps_list = build_ps(orig_pt_sample)



hpd_res = quantile(chain; q=[0.5])[:, 2] # 0.5 is the median second row is the value vector
estim_pt = unflatten(hpd_res)
p_var2, p_invar2, event_times2 = build_ps(estim_pt; return_array=false)
cp_rec = ChangePointParameters(
    u0_expanded, p_var2, p_invar2, event_times2, set_of_p_var, Dict()
)
loss_estim_params = loss_function(hpd_res)
sol_rec = solve_event_times(cp_rec, ts, prob_embed_sample, alg, sensealg; solvesettings...)
plt_rec = single_plot_solutions(
    sol_rec;
    loss_function_kwargs=loss_function_kwargs,
    loss=round(loss_estim_params; digits=1),
    event_times=event_times2,
    event_times_true=event_times_sample,
    name="reconstructed from median value of posterior"
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

@quickactivate "SwitchDyn-Changepoints-Parameters-Identification"
save_path = datadir(
    "sims/inference_lv/",
)
!isdir(save_path) && mkdir(save_path)
save_margs = OrderedDict(
    "seed" => i,
    "model_name" => model_name,
    "u0" => u0,
    "ts" => ts,
    "tspan" => tspan,
    "priors" => priors,
    "param_dict" => param_dict,
    "turing_model" => turing_model,
    "loss_function_kwargs" => loss_function_kwargs,
    "var_scaling" => var_scaling,
)

push!(
    save_margs,
    "sol_rec" => DataFrames.DataFrame(sol_rec),
    "chain" => new_chain_,
    "sampler_info" => sampler,
    "sample_length" => sample_length,
)

new_chain_100 = Turing.sample(rng, new_chain_, 100)
post_samples = DataFrames.DataFrame(new_chain_100)[:, new_chain_100.name_map.parameters]
post_samples = Matrix(post_samples)
true_param = param_dict[∈(new_chain_100.name_map.parameters).(param_dict.symbol), "true"]
post_df_ = DataFrames.DataFrame(new_chain_)[:, new_names]
param_dict = DataFrames.DataFrame(param_dict)
@unpack data_true, mean_idxs, var_idxs = loss_function_kwargs
data_true = hcat(data_true...)


using JSON
json_data = Dict(
    "sol_rec" => DataFrames.DataFrame(sol_rec),
    "mean_idxs" => mean_idxs,
    "var_idxs" => var_idxs,
    "data_true" => data_true,
    "param_dict" => DataFrames.DataFrame(param_dict),
    "post_df" => post_df_,
    "true_param" => true_param,
    "param_list" => new_names,
    "post_samples" => post_samples,
    "loss_estim_params" => loss_estim_params,
)

json_string = JSON.json(json_data)

open("$save_path/lv_data_$(i).json", "w") do f
    return JSON.print(f, json_string)
end
