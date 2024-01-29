# %%
from load_json_helper_function import load_data
from plot_helper_functions import plot_posterior_prior_compare, germany_time_series
import os 
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

# load json file and create variables
title = "epid_data"
data_path = os.path.join(dir_path, "../../../data/sims/inference_epid/epid_data.json")
data_path_compare = os.path.join(dir_path, "../../../data/sims/inference_epid/epid_data_strucchange.json")

save_path = "./"

column_names = ["β₁", "β₂", "β₃", "μ₁", "μ₂", "σ", "$τ_{β₁}$", "$τ_{β₂}$", "$τ_{μ₁}$"]
column_names_compare = ["β₁", "β₂", "β₃", "μ₁", "μ₂", "σ"]

true_params, prior_df, post_df, _, _ = load_data(data_path, column_names, return_param_dict=True)

_, _, post_df_compare, df_quantile_sol_compare, param_dict_compare = load_data(data_path_compare, column_names_compare, return_param_dict=True)

compare_changepoints_vals = param_dict_compare.loc[:, 'orig'][-3:].values
# %%
# add np.nan to compare_changepoints_vals
cps = np.append(compare_changepoints_vals, np.nan)
f1 = germany_time_series(df_quantile_sol_compare, cps, figsize = (3.2, 3.5), layout=(2, 1), no_legend=False, scale_y=1.6)
f1.savefig(save_path + f"{title}_time_series_compare.pdf")
# %%
param_long = [
    "First transmission rate $β₁$",
    "Second transmission rate $β₂$",
    'Third transmission rate $β₃$',
    "First mortality rate $μ₁$",
    "Second mortality rate $μ₂$",
    "Transfer rate: exposed to infectious $σ$",
    "First change point $τ_{β₁}$",
    "Second change point $τ_{β₂}$",
    "First change point $τ_{μ₁}$",
]

compare_changepoints_keys = [6, 7, 8]
compare_changepoints = dict(zip(compare_changepoints_keys, compare_changepoints_vals))

actual_changepoints_keys = [6, 7]
actual_changepoints_vals = [5 + 14.4, 23 + 14.4]

actual_changepoints = dict(zip(actual_changepoints_keys, actual_changepoints_vals))
f2 = plot_posterior_prior_compare(prior_df, post_df, post_df_compare, true_params, param_long, n_row=4, figsize = (4.8, 4.5), compare_method="strucchange", compare_changepoints=compare_changepoints, actual_changepoints=actual_changepoints, no_MAP_Med=True)
f2.savefig(save_path + f"{title}_param_compare.pdf")
# %%
