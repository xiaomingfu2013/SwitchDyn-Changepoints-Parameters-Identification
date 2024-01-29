#%%
from load_json_helper_function import load_multi_post_df
from plots_scripts.plot_helpers import plot_recovery
import numpy as np
import json
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# read posteriors of specified case using our method
load_path_ours = os.path.join(dir_path,"../../../data/sims/inference_lv/")
load_path_mcp = os.path.join(dir_path,"../../../data/sims/inference_lv_mcp/")
save_path = os.path.join(dir_path,"../../plots_scripts/lv/")

trend_line = 'OLS'
true_params_all, _, param_list = load_multi_post_df(load_path_ours)
_, post_samples_mcp, param_list_mcp = load_multi_post_df(load_path_mcp)
post_color_ours = "#8f2727"
post_color_mcp = "#1f78b4"

mcp_estimlist = []
for i in range(1, 301):
    with open(load_path_mcp + "lv_data_" + str(i) + ".json", "r") as f:
        json_string = json.load(f)
        f.close()

    json_data = json.loads(json_string)
    mcp_estimlist.append(json_data['param_dict']['estim'][-1])

mcp_estimlist = np.array(mcp_estimlist)
mcp_estimlist = np.expand_dims(mcp_estimlist, axis=(1,2))
# repeat the second axis 100 times
mcp_estimlist = np.repeat(mcp_estimlist, 100, axis=1)
# concatentate post_samples_mcps and mcp_estimlist
post_samples_mcp_ex = np.concatenate((post_samples_mcp, mcp_estimlist), axis=2)

# %%
f5 = plot_recovery(post_samples_mcp_ex, true_params_all, param_list, color = post_color_mcp, trend_line=trend_line, boundary={0: [0.4, 1.2], 1: [0.4, 1.1], 2 :[3.5*1e-3, 8.5*1e-3], 3: [2.5*1e-3, 6*1e-3],  4: [12, 45]}, add_r2=False)
f5.savefig(save_path+f"fig_recover_lv_mcp_trend_{trend_line}.pdf")
