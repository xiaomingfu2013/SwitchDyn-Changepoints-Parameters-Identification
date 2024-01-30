# in this example: we have the true distribution of three different parameters
# we want draw from the true distribution of all three:
# heterogenous initiation rates: kG1, kG2
# asynchronized change point: tau_1
# use the prior distribution to accurately estimate the true parameters
# this script is for each single set of parameters for HPC

local_machine = isempty(ARGS)
i = local_machine ? 2 : parse(Int, ARGS[1])
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
    using Optimization, OptimizationOptimisers, OptimizationOptimJL, OptimizationNLopt
    using Zygote
    Turing.setadbackend(:zygote)
    theme(:vibrant)
    include("../../../src/adjoint_method.jl")
    include("../../../src/ode_models.jl")
    include("../../../src/utils.jl")
    include("../../../src/turing_models.jl")
end

using DrWatson
@quickactivate "SwitchDyn-Changepoints-Parameters-Identification"


maxiters = local_machine ? 1e3 : 5e3
learning_rate = 1e-3


rng = MersenneTwister(i)
@info "Precompile finish with time: $precompile_time"
model_name = "get_stoch_gene_model_poisson_true_post"

@info "Training maxiters: $maxiters with model $model_name"
# ========================================
# set up the JumpProblem, Markovian model
# ========================================
sample_priors = (
    p_var=(kG=[
        LogNormal(log(3.0), 0.1),
        LogNormal(log(1.5), 0.1),
    ],),
    p_invar=(dP=PH.fixed(1.0),),
    event_times=(
        τ₁=LogNormal(log(1.5), 0.1),
    ),
    scaling=(
        SCALING_p_var=(SCALING_kG=PH.fixed(1e1 * ones(2)),),
        SCALING_p_invar=PH.fixed([1.0]),
        SCALING_event_times=PH.fixed(1e1 * ones(1)),
    ),
)
@info "setting up jump problem"
true_pt = sample_tuple(rng, sample_priors)
nt = (num_event=length(true_pt.event_times),)
ode_model, lna_sys = odemodel_selection(model_name)(nt);

rn = lna_sys.rn
u0 = [5.0]
duration = 50.0
tspan = (0.0, duration)
ts = tspan[1]:0.5:tspan[2]
num_event = length(true_pt.event_times)
set_of_p_var = length(true_pt.p_var)
true_ps_var, true_ps_invar, true_ps_event_times = build_ps(true_pt; return_array=false)
true_ps = [true_ps_var..., true_ps_invar..., true_ps_event_times...]
dprob = DiscreteProblem(rn, u0, tspan, true_ps)
jprob = JumpProblem(rn, dprob, Direct(); save_positions=(false, false))
cb = event_callback_reaction_jumpprob_all_params_in_condition(num_event, set_of_p_var)
jsol = solve(
    jprob, SSAStepper(); tstops=true_ps_event_times, callback=cb, saveat=ts, seed=i
)

plt_onejump = Plots.plot(
    jsol; lw=2, label="P", xlabel="time", ylabel="number", title="Stochastic gene model"
)
synthetic_data_noisy = (jsol[1, :],)

@info "Setting up ODEProblem"
alg = Tsit5()
sensealg = SimpleAdjoint(:EnzymeVJP)
solvesettings = Dict()
nt = (num_event=length(true_pt.event_times),)
u0_expanded = expand_initial_conditions(lna_sys, u0)
prob_embed = ODEProblem(ode_model, u0_expanded, tspan, true_ps)
mean_U_idx, var_U_idx = find_states_cov_number(1, lna_sys)

p_var_true, p_invar_true, event_times_true = build_ps(true_pt; return_array=false)
cp_true = ChangePointParameters(
    u0_expanded, p_var_true, p_invar_true, event_times_true, set_of_p_var, Dict()
)
sol_true = solve_event_times(cp_true, ts, prob_embed, alg, sensealg; solvesettings...)

# ========================================
#  Set up the Bayesian inference problem
# ========================================
priors = (
    p_var=(kG=[Uniform(1.5, 4.5), Uniform(0.5, 2.5)],),
    p_invar=(dP=PH.fixed(1.0),),
    event_times=(τ₁=Uniform(0.0, 5.0),),
    scaling=(
        SCALING_p_var=(SCALING_kG=PH.fixed(1e1 * ones(2)),),
        SCALING_p_invar=PH.fixed([1.0]),
        SCALING_event_times=PH.fixed(1e1 * ones(1)),
    ),
)
guess_pt = sample_tuple(rng, priors)
guess_params_flatten, unflatten = PH.value_flatten(guess_pt);
true_params_flatten, _ = PH.value_flatten(true_pt);

loss_function_kwargs = (
    data_true=synthetic_data_noisy,
    sensealg=sensealg,
    prob=prob_embed,
    alg=alg,
    ts=ts,
    set_of_p_var=set_of_p_var,
    solvesettings=solvesettings,
    u0_expanded=u0_expanded,
    mean_idxs=[mean_U_idx],
    var_idxs=[var_U_idx],
    unflatten=unflatten,
);

logpdf_custom = (mean, var, data) -> logpdf(product_distribution(Poisson.(mean)), data)
turing_model = turing_change_point(
    priors; loss_function_kwargs=loss_function_kwargs, logpdf_custom=logpdf_custom
);
optim_prob = optim_problem(turing_model, MAP(); autoad=Optimization.AutoZygote());
loss_function(p::Vector) = optim_prob.prob.f.f(p, SciMLBase.NullParameters())

guess_loss = round(loss_function(guess_params_flatten), digits=3)
true_loss = round(loss_function(true_params_flatten), digits=3)

## Turing Chain
new_names = extend_event_time_symbols(priors)
sampler = Turing.NUTS(0.65; max_depth=10, Δ_max=1000.0, init_ϵ=5e-3)
sample_length = 500
rng = MersenneTwister(i)

init_params_guess = guess_params_flatten
init_params_guess[3] = mean(sample_priors.event_times.τ₁)
chain = Turing.sample(
    rng,
    turing_model,
    sampler,
    sample_length;
    progress=true,
    init_params=init_params_guess
)
change_names = Dict(chain.name_map.parameters .=> new_names)
new_chain = replacenames(chain, change_names)
new_chain_ = scale_chain(new_chain, unflatten, [3]) # 5 is the index of dP
summarize(new_chain_)
plt_chn = plot(new_chain_)
true_ps_list = build_ps(true_pt)

hpd_res = quantile(chain; q=[0.5])[:, 2] # 0.5 is the median second row is the value vector
loss_estim_params = round(loss_function(hpd_res), digits=3)
estim_pt = unflatten(hpd_res)
p_var2, p_invar2, event_times2 = build_ps(estim_pt; return_array=false)

cp_rec = ChangePointParameters(
    u0_expanded, p_var2, p_invar2, event_times2, set_of_p_var, Dict()
)

sol_rec = solve_event_times(cp_rec, ts, prob_embed, alg, sensealg; solvesettings...)
plt1 = single_plot_solutions(
    sol_rec;
    loss_function_kwargs=loss_function_kwargs,
    event_times=event_times2,
    event_times_true=true_ps_event_times,
    name="Loss: guess $guess_loss true $true_loss reconstructed $loss_estim_params"
)
Plots.plot!(
    jsol; lw=2, label="P", xlabel="time", ylabel="number", color="orange", alpha=0.6
)

estim = wrapper_scaled_param_dict(estim_pt)
true_ps_list = build_ps(true_pt)
estim_list = build_ps(estim_pt)
param_dict = DataFrames.DataFrame(
    "symbol" => extend_event_time_symbols(priors; all_symbols=true),
    "estim" => estim_list,
    "true" => true_ps_list,
    "relative_error" => abs.(estim_list .- true_ps_list) ./ true_ps_list,
)


save_path = datadir(
    "sims/inference_stoch/bayesian/"
)
!isdir(save_path) && mkdir(save_path)

# ========================================
# prepare the 100 samples from the posterior
# ========================================
new_chain_100 = Turing.sample(rng, new_chain_, 100)
post_samples = Matrix(DataFrame(new_chain_100)[:, new_chain_100.name_map.parameters])

prior_chain_ = Turing.sample(rng, turing_model, Prior(), 1000)
change_names = Dict(prior_chain_.name_map.parameters .=> new_names)
prior_chain__ = replacenames(prior_chain_, change_names)
prior_chain = scale_chain(prior_chain__, unflatten, [3])
prior_df = DataFrames.DataFrame(prior_chain)[:, new_names]

post_df_ = DataFrames.DataFrame(new_chain_)[:, new_names]
post_df = DataFrames.DataFrame(post_df_)
param_dict = DataFrames.DataFrame(param_dict)
@unpack data_true, mean_idxs, var_idxs = loss_function_kwargs
data_true = hcat(data_true...)



using JSON
json_data = OrderedDict(
    "seed" => i,
    "priors" => priors,
    "sample_priors" => sample_priors,
    "param_dict" => param_dict,
    "estim_sol" => DataFrame(sol_rec),
    "true_jsol" => DataFrame(jsol),
    "rec_loss" => loss_estim_params,
    "post_samples" => post_samples,
    "post_df" => post_df,
    "prior_df" => prior_df,
)
json_string = JSON.json(json_data)
open("$save_path/stoch_data_$(i).json", "w") do f
    return JSON.print(f, json_string)
end
# savefig(plt1, "$save_path/stoch_reconstructed_$(i).png")
