import json
import pandas as pd
import os
state_list = {}
true_change_points = {}
estimated_change_points = {}
dir_path = os.path.dirname(os.path.realpath(__file__))

for id in range(1, 301):

    with open(os.path.join(dir_path, "../../../data/sims/inference_stoch/bayesian/stoch_data_") + str(id) + ".json", "r") as f:
        json_string = json.load(f)
        f.close()
        
    json_data = json.loads(json_string)

    # check the keys
    ts = json_data['true_jsol']['P(t)']
    
    # push the ts to the state list
    state_list[id] = ts
    true_change_points[id] = json_data['param_dict']['true'][-1]
    estimated_change_points[id] = json_data['param_dict']['estim'][-1]
    
# save the state list in to a dataframe
df = pd.DataFrame(state_list)
index = json_data['true_jsol']['timestamp']
# add the index of df to be the timestamp
df.index = index


df.to_csv(os.path.join(dir_path, "../../../data/sims/inference_stoch/stoch_data.csv"))

df_true = pd.DataFrame(true_change_points, index=[0])
df_estim = pd.DataFrame(estimated_change_points, index=[0])
# merge the two dataframes
df_change_points = pd.concat([df_true, df_estim])
# set the index
df_change_points.index = ['true', 'estim']
save_path = os.path.join(dir_path, "../../../data/sims/inference_stoch/stoch_data_change_points.csv")
df_change_points.to_csv(save_path, index=True)