import os
import numpy as np
import pandas as pd
import json


def load_multi_post_df(load_path):
    """
    load all json files in the directory and return the dataframes named post_samples_all
    """
    # count the number of json files in the directory
    all_files = os.listdir(load_path)
    json_files = [pos_json for pos_json in all_files if pos_json.endswith(".json")]
    true_params_all = []
    est_params_all = []
    post_samples_all = []
    len_json = len(json_files)
    for i, file_name in zip(range(1, len_json + 1), json_files):
        file_path = os.path.join(load_path, file_name)
        with open(file_path, "r") as f:
            json_string = json.load(f)
            f.close()
        
        json_data = json.loads(json_string)
        tmp_true = np.array(json_data["param_dict"]["true"])
        tmp_est = np.array(json_data["param_dict"]["estim"])
        tmp_post = np.array(json_data["post_samples"])
        
        true_params_all.append(tmp_true)
        est_params_all.append(tmp_est)
        post_samples_all.append(tmp_post)
            
    # reshape the data length  x n_samples x param
    
    true_params_all = np.stack(true_params_all, axis=0)
    post_samples_all = np.stack(post_samples_all, axis=0)
    est_params_all = np.stack(est_params_all, axis=0)
    
    # change last axis of post_samples_all
    post_samples_all = np.moveaxis(post_samples_all, -2, -1)
    return true_params_all, post_samples_all, est_params_all