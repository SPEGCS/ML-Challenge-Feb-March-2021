#%% this module is to used test different models
import os
import pathlib
import pickle
import random
import time
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from xgboost import XGBClassifier as XGBC
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor as XGB
from sklearn.base import clone
from plot import plot_crossplot, plot_logs_columns, plot_feature_important

SelectedScaler = RobustScaler

from load_pickle import (
    las_data_DTSM_QC2,
    las_lat_lon,
    alias_dict,
    rock_info,
)

# load customized functions and requried dataset
from util import (
    to_pkl,
    read_pkl,
    process_las,
    assign_rock_type,
)

pio.renderers.default = "browser"
rng = np.random.RandomState(42)
#%% load the model '7', '6_1', '6_2' and '6_3, '0_4_DTCO'

model_name = "2_3"
algorithm_name = "xgb"

path = f"predictions/TEST_model_building/{model_name}_{algorithm_name}"

if not os.path.exists(path):
    os.mkdir(path)

# import model best_params, should rerun the RandomizedSearchCV on all data
model_path = f"predictions/tuning_final"
model = read_pkl(f"{model_path}/model_xgb_{model_name}.pickle")

model_dicts = {
    "model": model["zone_9"]["best_estimator"],
    "scaler_x": model["zone_9"]["scaler_x"],
    "scaler_y": model["zone_9"]["scaler_y"],
    "params": model["zone_9"]["best_params"],
    "target_mnemonics": model["zone_9"]["target_mnemonics"],
}
target_mnemonics = model_dicts["target_mnemonics"]

# save the model to pickle file
to_pkl(model_dicts, f"{path}/model_xgb_{model_name}_final.pickle")

print(model_name, target_mnemonics)
print("\n")

#%% retrain the model for model '7'

# get training
las_dict = read_pkl(f"{model_path}/las_dict_{model_name}.pickle")
print("las_dict length:", len(las_dict))

# create training dataset
Xy_train = []
for k, v in las_dict.items():
    Xy_train.append(v)
Xy_train = pd.concat(Xy_train, axis=0)

print("Xy_train:", Xy_train.sample(10))


time0 = time.time()

# create a mdoel with best_params
model = XGB(
    **model_dicts["params"],
    objective="reg:squarederror",
    tree_method="gpu_hist",
    deterministic_histogram=True,
    random_state=rng,
)
print("model:", model)

# select 'target mnemonics' data
X_train = Xy_train[target_mnemonics].iloc[:, :-1]
y_train = Xy_train[target_mnemonics].iloc[:, -1:]  # get the 'DTSM'

# scale training data with existing scalers, which should be the same
scaler_x, scaler_y = (
    RobustScaler(),
    RobustScaler(),
)  # model_dicts["scaler_x"], model_dicts["scaler_y"]
X_train = scaler_x.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)

# train the model with 'rock type' in X_train
model.fit(X_train, y_train)

model_dicts_rocktype = {
    "target_mnemonics": target_mnemonics,
    "model": model,
    "scaler_x": scaler_x,
    "scaler_y": scaler_y,
}

# save the model to pickle file
to_pkl(model_dicts_rocktype, f"{path}/model_xgb_{model_name}_final_refit.pickle")
print("Completed fitting model with rock type.")
#%% retrain the model if adding rock type, this is only for model '7' and '6_1'
# do NOT run for other models like '6_2', '6_3' etc.

# get training
las_dict = read_pkl(f"{model_path}/las_dict_{model_name}.pickle")
print("las_dict length:", len(las_dict))

# create training dataset
Xy_train = []
for k, v in las_dict.items():
    v = assign_rock_type(
        df=v,
        las_name=k,
        info=rock_info,
    )

    Xy_train.append(v)
Xy_train = pd.concat(Xy_train, axis=0)

print("Xy_train:", Xy_train.sample(10))


time0 = time.time()

# create a mdoel with best_params
model = XGB(
    **model_dicts["params"],
    objective="reg:squarederror",
    tree_method="gpu_hist",
    deterministic_histogram=True,
    random_state=rng,
)
print("model:", model)

# select 'target mnemonics' data
X_train = Xy_train[target_mnemonics].iloc[:, :-1]
y_train = Xy_train[target_mnemonics].iloc[:, -1:]  # get the 'DTSM'

# scale training data with existing scalers, which should be the same
scaler_x, scaler_y = model_dicts["scaler_x"], model_dicts["scaler_y"]
X_train = scaler_x.transform(X_train)
y_train = scaler_y.transform(y_train)

# add the rock type after scaling
X_train = np.c_[X_train, Xy_train[["rock_type"]].values.reshape(-1, 1)]
print(
    "X_train sum of rock_type:",
    sum(X_train[:, -1:]),
    len(X_train),
    sum(X_train[:, -1:]) / len(X_train),
)

# train the model with 'rock type' in X_train
model.fit(X_train, y_train)

model_dicts_rocktype = {
    "target_mnemonics": v.columns.tolist(),
    "model": model,
    "scaler_x": model_dicts["scaler_x"],
    "scaler_y": model_dicts["scaler_y"],
}

# save the model to pickle file
to_pkl(model_dicts_rocktype, f"{path}/model_xgb_{model_name}_rocktype_final.pickle")
print("Completed fitting model with rock type.")
