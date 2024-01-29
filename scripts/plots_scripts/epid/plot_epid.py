# %%
from load_json_helper_function import load_data
from plot_helper_functions import germany_time_series, compute_change_points
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# load json file and create variables


title = "epid_data"
data_path = os.path.join(dir_path, "../../../data/sims/inference_epid/epid_data.json")
save_path = "./"

column_names = ["β₁", "β₂", "β₃", "μ₁", "μ₂", "σ", "$τ_{β₁}$", "$τ_{β₂}$", "$τ_{μ₁}$"]
true_params, prior_df, post_df, df_quantile_sol, param_dict = load_data(data_path, column_names, return_param_dict=True)




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

cps = compute_change_points(post_df)
f1 = germany_time_series(df_quantile_sol, cps, figsize = (3.2, 3.5), layout=(2, 1), no_legend=True, scale_y=1.1)
f1.savefig(save_path + f"{title}_time_series.pdf")
