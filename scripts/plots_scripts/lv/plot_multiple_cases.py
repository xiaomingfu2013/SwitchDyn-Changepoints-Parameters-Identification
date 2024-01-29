#%%
from load_json_helper_function import load_multi_post_df
from plot_helper_functions import plot_histogram
from plots_scripts.plot_helpers import plot_recovery
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# read posteriors of specified case using our method
load_path_ours = os.path.join(dir_path, f"../../../data/sims/inference_lv/")
save_path = os.path.join(dir_path)
true_params_all, post_samples_ours, param_list = load_multi_post_df(load_path_ours)
# %%
f2 = plot_histogram(post_samples_ours, true_params_all, param_list, num_bins=7)
f2.savefig(save_path+f"/fig_lv_hist.pdf")
# %%
post_color_ours = "#8f2727"
post_color_mcp = "#1f78b4"
f3 = plot_recovery(post_samples_ours, true_params_all, param_list, color = post_color_ours, trend_line='OLS', boundary={0: [0.4, 1.2], 1: [0.4, 1.1], 2 :[3.5*1e-3, 8.5*1e-3], 3: [2.5*1e-3, 6*1e-3],  4: [12, 45]}, add_r2=False)
f3.savefig(save_path+f"/fig_lv_recover.pdf")

