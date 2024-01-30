using JSON
using DrWatson
@quickactivate "SwitchDyn-Changepoints-Parameters-Identification"
number_of_sims = 300
save_path = datadir(
    "sims/inference_stoch"
)
# save_path = datadir(
#     "sims/inference_stoch/compare_cp_strucchange"
# )
load_path = save_path
tmp1_ = JSON.parsefile("$load_path/stoch_data_1.json")

tmp1 = JSON.parse(tmp1_)
param_idxs = [1, 2, 4] # the 3rd parameter is fixed
param_names = tmp1["param_dict"]["symbol"][param_idxs]
sample_priors = tmp1["sample_priors"]

estim_ps_list = []
true_ps_list = []
jsol_list = []

for i in 1:number_of_sims
    tmp_ = JSON.parsefile("$load_path/stoch_data_$i.json")
    tmp = JSON.parse(tmp_)
    jsol = tmp["true_jsol"]
    estim_tmp = tmp["param_dict"]["estim"][param_idxs]
    true_tmp = tmp["param_dict"]["true"][param_idxs]
    push!(estim_ps_list, estim_tmp)
    push!(true_ps_list, true_tmp)
    push!(jsol_list, jsol)
end

estim_df = hcat(estim_ps_list...)'
true_df = hcat(true_ps_list...)'


jsol_list
using JSON
json_data = Dict(
    "param_names" => param_names,
    "estim_df" => estim_df,
    "true_df" => true_df,
    "sample_priors" => sample_priors,
    "jsol_list" => jsol_list,
)

json_string = JSON.json(json_data)

open("$load_path/stoch_merge.json", "w") do f
    return JSON.print(f, json_string)
end
