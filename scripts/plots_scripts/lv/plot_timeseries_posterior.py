# %%
from load_json_helper_function import load_other_data
from plot_helper_functions import plot_time_series
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


case_no = 15  # specify case number for plotting
# folder = ""
folder = "_mcp"
# read posteriors of specified case using our method
load_path = os.path.join(dir_path, f"../../../data/sims/inference_lv{folder}/")
save_path = os.path.join(dir_path)

data_true, true_param, sol_rec, param_dict, mean_idxs, var_idxs, estimated_sol = load_other_data(load_path, case_no)

# %%
change_point_data = {'true':param_dict['true'][5], 'estim':param_dict['estim'][5]} # 5 here is the row index of tau
# plot time series
f = plot_time_series(time_series_data = estimated_sol, change_point_data = change_point_data)
f.savefig(save_path + "/fig_lv_time_series_" + str(case_no) + f"{folder}.pdf")

# %%
