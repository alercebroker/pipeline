import numpy as np
import pandas as pd
import os


model_path = "models/hrf_classifier_20240710-142630"
file_path = os.path.join(model_path, "hierarchical_random_forest_model.pkl")
hrf_dict = pd.read_pickle(file_path)

feature_list = hrf_dict["feature_list"]
rf_name_list = []
importances_list = []
for rf_name, rf_model in hrf_dict["model"].items():
    rf_name_list.append(rf_name)
    importances_list.append(rf_model.feature_importances_)

df_importances = pd.DataFrame(
    data=np.stack(importances_list, axis=1), columns=rf_name_list, index=feature_list
)

for column in df_importances.columns:
    print(df_importances.sort_values(column, ascending=False).iloc[:30][column])
    print("-" * 20)
