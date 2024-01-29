import json
import pandas as pd
import numpy as np
import os

def load_multi_post_df(load_path):
    """
    load all json files in the directory and return the dataframes named post_samples_all
    """
    # count the number of json files in the directory
    all_files = os.listdir(load_path)
    json_files = [pos_json for pos_json in all_files if pos_json.endswith(".json")]
    
    for i, file_name in zip(range(1, len(json_files)+1), json_files):
        file_path = os.path.join(load_path, file_name)
        with open(file_path, "r") as f:
            json_string = json.load(f)
            f.close()

        json_data = json.loads(json_string)
        tmp_true = np.array(json_data["true_param"])
        tmp_post = np.array(json_data["post_samples"])
        if i == 1:
            true_params_all = tmp_true
            post_samples_all = tmp_post
        else:
            true_params_all = np.dstack((true_params_all, tmp_true))
            post_samples_all = np.dstack((post_samples_all, tmp_post))

    true_params_all = np.transpose(true_params_all, (2, 1, 0))[:, :, 0]
    post_samples_all = np.transpose(post_samples_all, (2, 1, 0))
    param_list = json_data["param_list"]
    return true_params_all, post_samples_all, param_list



def load_other_data(load_path, case_no):
    """
    load json file and return the dataframes
    """
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # load_path = os.path.join(dir_path, '../../') + load_path
    with open(load_path + "lv_data_" + str(case_no) + ".json", "r") as f:
        json_string = json.load(f)
        f.close()

    json_data = json.loads(json_string)
    data_true = pd.DataFrame(json_data["data_true"])
    sol_rec = pd.DataFrame(json_data["sol_rec"])
    param_dict = pd.DataFrame(json_data["param_dict"])
    mean_idxs = json_data["mean_idxs"]
    var_idxs = json_data["var_idxs"]
    true_param = json_data["true_param"]

    estimated_sol = sol_rec.iloc[:, [0] + mean_idxs + var_idxs]
    true_data = data_true.T
    estimated_sol = pd.concat([estimated_sol, true_data], axis=1)
    estimated_sol.columns = [
        "date",
        "U_mean",
        "V_mean",
        "U_var",
        "V_var",
        "U_true",
        "V_true",
    ]
    estimated_sol["U_std"] = np.sqrt(estimated_sol["U_var"])
    estimated_sol["V_std"] = np.sqrt(estimated_sol["V_var"])

    return (
        data_true,
        true_param,
        sol_rec,
        param_dict,
        mean_idxs,
        var_idxs,
        estimated_sol,
    )
