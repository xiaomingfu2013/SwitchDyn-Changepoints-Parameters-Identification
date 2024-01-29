import json
import pandas as pd
import numpy as np
import glob

def load_data(data_path, column_names, return_param_dict=False):
    """
    load json file and return the dataframes
    """
    with open(data_path, "r") as f:
        json_string = json.load(f)
        f.close()

    json_data = json.loads(json_string)
    # data_true = pd.DataFrame(json_data["data_true"])
    # sol_rec = pd.DataFrame(json_data["sol_rec"])
    # param_dict = pd.DataFrame(json_data["param_dict"])
    prior_df = pd.DataFrame(json_data["prior_df"])
    post_df = pd.DataFrame(json_data["post_df"])
    post_df.columns = column_names
    true_param = json_data["true_param"]
    true_params = np.transpose(pd.DataFrame(true_param))
    df_quantile_sol = pd.DataFrame(json_data["df_quantile_sol"])
    true_params.columns = column_names
    if return_param_dict:
        param_dict = pd.DataFrame(json_data["param_dict"])
        return (
            # data_true,
            true_params,
            # sol_rec,
            prior_df,
            post_df,
            df_quantile_sol,
            param_dict,
        )
    else:
        return (
            # data_true,
            true_params,
            # sol_rec,
            # param_dict,
            prior_df,
            post_df,
        )