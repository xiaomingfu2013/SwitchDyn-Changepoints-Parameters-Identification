import os
import json
import pandas as pd
from plot_helper_functions import plot_timeseries
dir_path = os.path.dirname(os.path.realpath(__file__))
load_path = os.path.join(dir_path, "../../../data/sims/inference_stoch/")

with open(load_path + "stoch_merge.json", "r") as f:
    json_string = json.load(f)
    f.close()

json_data = json.loads(json_string)
jsol_list = json_data["jsol_list"]
param_names = json_data["param_names"]
true_df = pd.DataFrame(json_data["true_df"]).T
true_df.columns = param_names


fig = plot_timeseries(jsol_list, true_df, sample_id1=0)
save_path = os.path.join(dir_path)
fig.savefig(save_path+f"/fig_stoch_timeseries.pdf")