precompile_time = @elapsed begin
    using DifferentialEquations
    using Plots
    using DataStructures # OrderedDict
    using Distributions
    using ParameterHandling
    const PH = ParameterHandling
    using Random
    using Turing
    Turing.setadbackend(:zygote)
    theme(:vibrant)
    include("../../../src/adjoint_method.jl")
    include("../../../src/ode_models.jl")
    include("../../../src/utils.jl")
    include("../../../src/turing_models.jl")
    include("../../../src/get_country_data.jl")
end

per_million_flag = true
country_data, df_cum_daily_, country_population = get_germany_timeseries_data_daily(
    viz=true, per_million=per_million_flag,
)


df_cum_daily = filter_df_portion(
    df_cum_daily_, 0.0, 120, country_population; per_million=per_million_flag
)

true_data_tuple = (
    cum_daily_cases=deepcopy(df_cum_daily.cum_daily_cases),
    cum_daily_deaths=deepcopy(df_cum_daily.cum_daily_deaths),
)
# true_data_tuple

@info "Precompile finish with time: $precompile_time"
model_name = "get_seird_lna_model"

@info "Training model $model_name"
priors = (
    p_var=(
        λ=[LogNormal(log(0.5), 0.2), LogNormal(log(0.25), 0.2), LogNormal(log(0.05), 0.2)],
        μ=[LogNormal(log(0.1), 0.1), LogNormal(log(0.4), 0.1)],
    ),
    p_invar=(σ=LogNormal(log(0.25), 0.1), γ=PH.fixed(1 / 14)),
    event_times=(
        τ_λ1=Uniform(0.10, 0.35), τ_λ2=Uniform(0.35, 0.60), τ_μ1=Uniform(0.10, 0.60)
    ),
    scaling=(
        SCALING_p_var=(SCALING_λ=PH.fixed(ones(3)), SCALING_μ=PH.fixed(1e-2 * ones(2))),
        SCALING_p_invar=PH.fixed([1.0, 1.0]),
        SCALING_event_times=PH.fixed(1e2 * ones(3)),
    ),
)

@info "Setting up initial conditions for synthetic ODE run"
## ====================
# randomly draw a set an ODE initial conditions
## ====================
guess_pt = sample_mean_tuple(priors)
len_split_event_times = (2, 1)
len_split_event_times_cum = (3, 5)
len_split_ps = compute_len_split_ps(guess_pt)
info_nt = (
    num_event=length(guess_pt.event_times),
    model_name=model_name,
    len_split_ps=len_split_ps,
    len_split_event_times=len_split_event_times,
    len_split_event_times_cum=len_split_event_times_cum,
)
model, lna_sys = get_seird_lna_model(info_nt);

@info "Setting up initial conditions for initial ODE run"
N0 = country_population
per_million = per_million_flag ? 1e6 / N0 : 1.0
I0 = max.(df_cum_daily.cum_daily_cases[1], per_million) # avoid I0 = 0
E0 = I0 * (1 + PH.value(guess_pt.p_invar.σ) / PH.value(guess_pt.p_invar.γ)) # (1 + σ / γi) * # where σ / γi is the ratio of the incubation period to the average infectious period.
R0 = 0.0
D0 = df_cum_daily.cum_daily_deaths[1]
u0 = [(N0 * per_million - E0 - I0 - R0 - D0), E0, I0, R0, D0, I0]  # S, E, I, R, D, cumI

@assert sum(u0[1:5]) ≈ N0 * per_million
duration = (df_cum_daily.date[end] - df_cum_daily.date[1]).value + 1.0
tspan = (1.0, duration)
ts = tspan[1]:1.0:tspan[2]

alg = AutoVern7(Rodas5())
sensealg = ForwardDiffSensitivity()
solvesettings = Dict(:maxiters => 1e5, :abstol => 1e-6, :reltol => 1e-6)

u0_expanded = expand_initial_conditions(lna_sys, u0)
ps = build_ps(guess_pt)
p_var, p_invar, event_times = build_ps(guess_pt; return_array=false)

init_cb_affect! =
    let λ_list = Float64[],
        μ_list = Float64[],
        event_times = event_times,
        len_split_event_times = len_split_event_times,
        num_event = info_nt.num_event,
        len_split_event_times_cum = len_split_event_times_cum,
        len_split_ps = len_split_ps

        function (integrator)
            event_time_idxs = find_event_time(
                integrator.t, event_times, len_split_event_times
            )
            event_time_λ, event_time_μ = event_time_idxs
            p_var, _, _ = split_vector(integrator.p, len_split_ps)
            λ = p_var[event_time_λ]
            μ = p_var[len_split_event_times_cum[1]+event_time_μ]
            append!(λ_list, λ)
            return append!(μ_list, μ)
        end
    end
init_cb = PresetTimeCallback(ts, init_cb_affect!; save_positions=(false, false))

odemodel_embed = model
prob_embed = ODEProblem(odemodel_embed, u0_expanded, tspan, ps)
@time sol_embed = solve(
    prob_embed, alg; saveat=ts, tstops=event_times, callback=init_cb, solvesettings...
)

mean_cumI_idx, var_cumI_idx = find_states_cov_number(6, lna_sys)
mean_cumD_idx, var_cumD_idx = find_states_cov_number(5, lna_sys)
mean_idxs = [mean_cumI_idx, mean_cumD_idx]
var_idxs = [var_cumI_idx, var_cumD_idx]
set_of_p_var = length(guess_pt.p_var)
cp = ChangePointParameters(u0_expanded, p_var, p_invar, event_times, set_of_p_var, Dict())
sol_event = solve_event_times(
    cp, ts, prob_embed, alg, sensealg; tstops=ts, (solvesettings...)
)
@assert sol_event[end] == sol_embed[end]

guess_params_flatten, unflatten = PH.value_flatten(guess_pt);
guess_params_flatten

loss_function_kwargs = (
    data_true=true_data_tuple,
    sensealg=sensealg,
    prob=prob_embed,
    alg=alg,
    ts=ts,
    set_of_p_var=set_of_p_var,
    solvesettings=solvesettings,
    u0_expanded=u0_expanded,
    mean_idxs=mean_idxs,
    var_idxs=var_idxs,
    unflatten=unflatten,
);

plt_guess = plot_solutions(sol_embed; loss_function_kwargs=loss_function_kwargs)

var_scaling = 1e-2
logpdf_custom =
    (mean, var, data) -> logpdf(MvNormal(mean, Diagonal(var_scaling * var)), data)
turing_model = turing_change_point(
    priors; loss_function_kwargs=loss_function_kwargs, logpdf_custom=logpdf_custom
);

using Optimization, OptimizationOptimisers, OptimizationNLopt
using Zygote

lb, _ = PH.value_flatten(sample_quantile_tuple(priors, 1e-16));
ub, _ = PH.value_flatten(sample_quantile_tuple(priors, 1 - 1e-16));

optim_prob = optim_problem(turing_model, MLE(); autoad=Optimization.AutoZygote());
loss_function(p::Vector) = optim_prob.prob.f.f(p, SciMLBase.NullParameters())

guess_loss = round(loss_function(guess_params_flatten); digits=1)

@info "Starting optimization"
new_prob = remake(optim_prob.prob; u0=guess_params_flatten)
optimization_time = @elapsed estim_params_local = solve(
    new_prob, OptimizationOptimisers.Adam(1e-3); progress=true, maxiters=5e3
)
local_loss = loss_function(estim_params_local.minimizer)
prob_with_bounds = remake(new_prob; u0=guess_params_flatten, lb=lb, ub=ub)
optimization_time = @elapsed estim_params_local2 = solve(
    prob_with_bounds, NLopt.LN_SBPLX(); progress=true, maxiters=Int(5e3)
)

@info "Optimization finished in $optimization_time seconds"
loss_estim_params = round(loss_function(estim_params_local2.minimizer); digits=1)
opt_estim_pt = unflatten(estim_params_local2.minimizer)

orig_ps = wrapper_scaled_param_dict(guess_pt)
opt_estim = wrapper_scaled_param_dict(opt_estim_pt)
new_names = extend_event_time_symbols(priors)
lb_ = build_ps(unflatten(lb))
ub_ = build_ps(unflatten(ub))

orig_ps_list = build_ps(guess_pt)
opt_estim_list = build_ps(opt_estim_pt)

## Turing Chain
using StatsPlots
using MCMCChains
max_depth = 7
sampler = NUTS(0.65; max_depth=max_depth, init_ϵ=1e-3)
sample_length = 1000
rng = Random.MersenneTwister(1234)

chain = Turing.sample(
    rng, turing_model, sampler, sample_length; init_params=estim_params_local2.minimizer
)
change_names = Dict(chain.name_map.parameters .=> new_names)
new_chain = replacenames(chain, change_names)
scaled_chain = scale_chain(new_chain, unflatten, [7])

summarize(scaled_chain)
chn_plt = plot(scaled_chain)

hpd_res = quantile(chain; q=[0.5])[:, 2] # 0.5 is the median
estim_pt = unflatten(hpd_res)
# store the median loss
median_final_loss = loss_function(hpd_res)

p_var2, p_invar2, event_times2 = build_ps(estim_pt; return_array=false)
cp_rec = ChangePointParameters(
    u0_expanded, p_var2, p_invar2, event_times2, set_of_p_var, Dict()
)
sol_rec = solve_event_times(cp_rec, ts, prob_embed, alg, sensealg; solvesettings...)
bayes_estim_list = build_ps(estim_pt)

# for the python plot save to json
prior_chain_ = Turing.sample(rng, turing_model, Prior(), 1000)
change_names = Dict(prior_chain_.name_map.parameters .=> new_names)
prior_chain__ = replacenames(prior_chain_, change_names)
prior_chain = scale_chain(prior_chain__, unflatten, [7])

param_dict = DataFrame(
    "symbol" => extend_event_time_symbols(priors; all_symbols=true),
    "orig" => orig_ps_list,
    "lower_bound" => lb_,
    "opt_estim" => opt_estim_list,
    "bayes_estim" => bayes_estim_list,
    "upper_bound" => ub_,
)
save_margs = Dict(
    "model_name" => model_name,
    "u0" => u0,
    "ts" => ts,
    "tspan" => tspan,
    "priors" => priors,
    "param_dict" => param_dict,
    "unscaled_chain" => chain,
    "prior_chain" => prior_chain,
    "scaled_chain" => scaled_chain,
    "sampler_info" => sampler,
    "sample_length" => sample_length,
    "estim_sol" => DataFrame(sol_rec),
    "logpdf_custom" => logpdf_custom,
    "var_scaling" => var_scaling,
    "max_depth" => max_depth,
    "df_cum_daily" => df_cum_daily,
)


prior_df = DataFrames.DataFrame(prior_chain)[:, new_names]
post_df = DataFrames.DataFrame(scaled_chain)[:, new_names]
true_param = param_dict[in.(param_dict.symbol, [new_names]), "orig"]

include("chain_model_posterior_predicts_function.jl")

df_quantile_sol, plt1 = posterior_predict(save_margs, loss_function_kwargs)
plt1

df_quantile_sol

using DrWatson
@quickactivate "ChangePointCodeRepo"
save_path = datadir("sims/inference_epid/")
!isdir(save_path) && mkdir(save_path)
savefig(chn_plt, "$save_path/seird_chain_plot4_depth$(max_depth)_scaling$(var_scaling).png")
savefig(
    plt1, "$save_path/seird_posterior_predicts4_depth$(max_depth)_scaling$(var_scaling).png"
)
using JSON
json_data = Dict(
    "sol_rec" => DataFrame(sol_rec),
    "data_true" => true_data_tuple,
    "param_dict" => param_dict,
    "true_param" => true_param,
    "prior_df" => prior_df,
    "post_df" => post_df,
    "df_quantile_sol" => df_quantile_sol,
    "mean_idxs" => mean_idxs,
    "var_idxs" => var_idxs,
    "loss" => median_final_loss,
)
json_string = JSON.json(json_data)

open("data/sims/inference_epid/epid_data.json", "w") do f
    return JSON.print(f, json_string)
end
