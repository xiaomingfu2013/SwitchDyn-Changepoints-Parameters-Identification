#%%
from plots_scripts.plot_helpers import plot_recovery
from load_helper_functions import load_multi_post_df
import os



dir_path = os.path.dirname(os.path.realpath(__file__))
# read posteriors of specified case using our method
load_path_ours = os.path.join(dir_path, f"../../../data/sims/inference_stoch/bayesian/")
save_path = os.path.join(dir_path, 'stoch')
load_path_strucchange = os.path.join(dir_path, "../../../data/sims/inference_stoch/compare_cp_bayesian_strucchange/")
true_params_all_ours,  post_samples_ours, est_params_ours = load_multi_post_df(load_path_ours)


# %%
true_params_all_struc,  post_samples_struc, est_params_struc = load_multi_post_df(load_path_strucchange)



true_params_var_ours = true_params_all_ours[:, [0, 1, 3]]
# %%
param_names = ["k₁", "k₂", "τ₁"]
# post_color_strucchange = "#1f77b4"
post_color_ours = "#8f2727"




# %%
import numpy as np
import matplotlib.pyplot as plt
import json
struc_estimlist = []

all_files = os.listdir(load_path_strucchange)
json_files = [pos_json for pos_json in all_files if pos_json.endswith(".json")]
for file in json_files:
    with open(load_path_strucchange + file, "r") as f:
        json_string = json.load(f)
        f.close()

    json_data = json.loads(json_string)
    struc_estimlist.append(json_data['param_dict']['estim'][-1])

struc_estimlist = np.array(struc_estimlist)
struc_estimlist = np.expand_dims(struc_estimlist, axis=(1,2))
# repeat the second axis 100 times
struc_estimlist = np.repeat(struc_estimlist, 100, axis=1)
# concatentate post_samples_mcps and mcp_estimlist
post_samples_struc_ex = np.concatenate((post_samples_struc, struc_estimlist), axis=2)

# %%
# missing 273 and 287
true_params_var_struc = true_params_all_struc[:, [0, 1, 3]]
f5 = plot_recovery(post_samples_struc_ex, true_params_var_struc, param_names, color="#1f78b4", trend_line='OLS', add_r2=False)
f5.savefig(save_path+f"fig_recover_stoch_strucchange_OLS.pdf")
# %%
fig1 = plot_recovery(post_samples_ours, true_params_var_ours, param_names, color = post_color_ours, trend_line='OLS', add_r2=False)
fig1.savefig(save_path+f"fig_recover_stoch_bayesian_OLS.pdf")
