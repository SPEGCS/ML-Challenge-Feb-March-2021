#%% this module is to used test different models
import os
import pathlib
import pickle
import random
import time
import re
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import StackingRegressor as Stack

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor as XGB
from plot import plot_crossplot, plot_logs_columns

from load_pickle import (
    alias_dict,
    las_data_TEST_renamed,
    las_data_DTSM_QC2,
    TEST_neighbors_20,
    feature_model,
)

from copy import deepcopy

# load customized functions and requried dataset
from util import (
    read_pkl,
    to_pkl,
    process_las,
)

pio.renderers.default = "browser"


#%%  test_predict


def test_predict(
    model=None,
    df_TEST=None,
    rock_type=None,
    TEST_folder=None,
):

    # start counting time
    time0 = time.time()
    if not os.path.exists(TEST_folder):
        os.mkdir(TEST_folder)

    # prepare TEST data with terget mnemonics
    df_TEST = df_TEST.copy()
    target_mnemonics = model["target_mnemonics"]
    target_mnemonics = [i for i in target_mnemonics if i != "DTSM"]
    print("target_mnemonics:", target_mnemonics)
    print("df_TEST:\n", df_TEST.sample(5))

    # scale test data and predict, and scale back prediction
    scalers = [model["scaler_x"], model["scaler_y"]]

    if rock_type is None:
        assert "rock_type" not in target_mnemonics
        X_test = df_TEST[target_mnemonics].values
        X_test = scalers[0].transform(X_test)
    else:
        assert "rock_type" in target_mnemonics
        assert isinstance(rock_type, np.ndarray)
        assert len(df_TEST) == len(rock_type)

        target_mnemonics_ = [i for i in target_mnemonics if i != "rock_type"]
        X_test_ = df_TEST[target_mnemonics_].values
        X_test_ = scalers[0].transform(X_test_)
        X_test = np.c_[X_test_, rock_type]

    y_predict = scalers[1].inverse_transform(
        model["model"].predict(X_test).reshape(-1, 1)
    )
    y_predict = pd.DataFrame(y_predict, columns=["DTSM_Pred"])

    y_predict["Depth"] = df_TEST.index
    print(
        f"Completed traing and predicting on TEST data in time {time.time()-time0:.2f} seconds"
    )

    return y_predict


def predict_mini(X_test=None, model=None):

    # X_test should only include X, matches with model
    scalers = [model["scaler_x"], model["scaler_y"]]
    X_test = X_test.copy().values
    X_test = scalers[0].transform(X_test)

    y_predict = scalers[1].inverse_transform(
        model["model"].predict(X_test).reshape(-1, 1)
    )

    return y_predict


#%% fix T01 data
las_data_TEST_QC = dict()

model_DTCO = read_pkl(
    "predictions/TEST_model_building/0_4_DTCO_xgb/model_xgb_0_4_DTCO_final.pickle"
)


TEST_well = "T01-00d02be79f49_TGS"
df = las_data_TEST_renamed[TEST_well].copy()

# plot the prediction and compare
plot_logs_columns(
    df=df,
    DTSM_pred=None,
    well_name=TEST_well,
    alias_dict=alias_dict,
    plot_show=True,
    plot_return=False,
    plot_save_file_name={TEST_well},
    plot_save_path=None,
    plot_save_format=[],
)

# fill DTCO
target_mnemonics_ = model_DTCO["target_mnemonics"][:-1]
y_DTCO = predict_mini(X_test=df[target_mnemonics_], model=model_DTCO)
df.loc[pd.isnull(df["DTCO"]), ["DTCO"]] = y_DTCO[pd.isnull(df["DTCO"])]


# plot the prediction and compare
plot_logs_columns(
    df=df,
    DTSM_pred=None,
    well_name=TEST_well,
    alias_dict=alias_dict,
    plot_show=True,
    plot_return=False,
    plot_save_file_name={TEST_well},
    plot_save_path=None,
    plot_save_format=[],
)

las_data_TEST_QC[TEST_well] = df


#%% predict:regular wells

model = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final.pickle")

for TEST_well in [
    "T06-2f96a5f92418_TGS",
    "T07-302460e3021a_TGS",
    "T08-3369b6f8fb6f_TGS",
    "T09-34a80ab7a5fa_TGS",
    "T10-63250f7d463b_TGS",
    "T12-7595ba9fb314_TGS",
    "T14-8e37531ba266_TGS",
    "T18-eed1e9537976_TGS",
    "T19-fca03aa6acde_TGS",
    "T20-ff7845ea074d_TGS",
]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict = test_predict(
        model=model,
        df_TEST=df,
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% predict: rock type=1, T05

model_rocktype = read_pkl(
    "predictions/TEST_model_building/7_xgb/model_xgb_7_rocktype_final.pickle"
)

# assign rock type
rock_type = dict()
for TEST_well in ["T05-20372701d5e2_TGS"]:

    # assign rock type first
    rock_type[TEST_well] = np.ones((len(las_data_TEST_renamed[TEST_well]), 1))

    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict = test_predict(
        model=model_rocktype,
        df_TEST=df,
        rock_type=rock_type[TEST_well],
        TEST_folder=TEST_folder,
    )

    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% predict: rock type=1, T13, T15

model_rocktype = read_pkl(
    "predictions/TEST_model_building/6_1_xgb/model_xgb_6_1_rocktype_final.pickle"
)

# assign rock type
rock_type = dict()
TEST_well = "T13-84c5fb9cc880_TGS"
rock_type[TEST_well] = np.ones((len(las_data_TEST_renamed[TEST_well]), 1))

TEST_well = "T15-94c1f5cae85c_TGS"
rock_type[TEST_well] = np.ones((len(las_data_TEST_renamed[TEST_well]), 1))
rock_type[TEST_well][las_data_TEST_renamed[TEST_well].index > 20900] = 0

for TEST_well in [
    "T13-84c5fb9cc880_TGS",
    "T15-94c1f5cae85c_TGS",
]:

    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict = test_predict(
        model=model_rocktype,
        df_TEST=df,
        rock_type=rock_type[TEST_well],
        TEST_folder=TEST_folder,
    )

    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T01, use las_data_TEST_QC

model = read_pkl("predictions/TEST_model_building/6_1_xgb/model_xgb_6_1_final.pickle")

for TEST_well in ["T01-00d02be79f49_TGS"]:
    df = las_data_TEST_QC[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict = test_predict(
        model=model,
        df_TEST=df,
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T11

model = read_pkl("predictions/TEST_model_building/6_2_xgb/model_xgb_6_2_final.pickle")

for TEST_well in ["T11-638f2cc65681_TGS"]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict = test_predict(
        model=model,
        df_TEST=df,
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T02

model_7 = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final.pickle")
model_3_4 = read_pkl(
    "predictions/TEST_model_building/3_4_xgb/model_xgb_3_4_final.pickle"
)

for TEST_well in ["T02-0a7822c59487_TGS"]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict_A = test_predict(
        model=model_7,
        df_TEST=df[df.index < 7900],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_B = test_predict(
        model=model_3_4,
        df_TEST=df[df.index >= 7900],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict = pd.concat([y_predict_A, y_predict_B], axis=0)
    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T03

model_6_3 = read_pkl(
    "predictions/TEST_model_building/6_3_xgb/model_xgb_6_3_final.pickle"
)

model_3_2 = read_pkl(
    "predictions/TEST_model_building/3_2_xgb/model_xgb_3_2_final.pickle"
)


for TEST_well in ["T03-113412eec2a6_TGS"]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict_A = test_predict(
        model=model_6_3,
        df_TEST=df[~pd.isnull(df["RHOB"])],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_B = test_predict(
        model=model_3_2,
        df_TEST=df[pd.isnull(df["RHOB"])],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict = pd.concat([y_predict_A, y_predict_B], axis=0)
    y_predict = y_predict.sort_values(by="Depth")
    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T04

model_4_2 = read_pkl(
    "predictions/TEST_model_building/4_2_xgb/model_xgb_4_2_final.pickle"
)

model_7 = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final.pickle")


model_5_4 = read_pkl(
    "predictions/TEST_model_building/5_4_xgb/model_xgb_5_4_final.pickle"
)


for TEST_well in ["T04-1684cc35f399_TGS"]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict_A = test_predict(
        model=model_4_2,
        df_TEST=df[df.index < 7000],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_B = test_predict(
        model=model_7,
        df_TEST=df[(df.index >= 7000) & (df.index <= 11280)],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_C = test_predict(
        model=model_5_4,
        df_TEST=df[df.index > 11280],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict = pd.concat([y_predict_A, y_predict_B, y_predict_C], axis=0)
    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T16

model_7 = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final.pickle")
model_4_4 = read_pkl(
    "predictions/TEST_model_building/4_4_xgb/model_xgb_4_4_final.pickle"
)
model_6_1 = read_pkl(
    "predictions/TEST_model_building/6_1_xgb/model_xgb_6_1_final.pickle"
)

for TEST_well in ["T16-ae16a9f64878_TGS"]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict_A = test_predict(
        model=model_4_4,
        df_TEST=df[df.index <= 6000],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_B = test_predict(
        model=model_7,
        df_TEST=df[(df.index > 6000) & (df.index <= 13850)],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_C = test_predict(
        model=model_6_1,
        df_TEST=df[(df.index > 13850) & (df.index <= 15065)],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_D = test_predict(
        model=model_4_4,
        df_TEST=df[df.index > 15065],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict = pd.concat([y_predict_A, y_predict_B, y_predict_C, y_predict_D], axis=0)
    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% Well: T17

model_2_3 = read_pkl(
    "predictions/TEST_model_building/2_3_xgb/model_xgb_2_3_final.pickle"
)
model_7 = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final.pickle")

for TEST_well in ["T17-ed48bda2217f_TGS"]:
    df = las_data_TEST_renamed[TEST_well]
    TEST_folder = f"predictions/TEST_final/{TEST_well}"

    y_predict_A = test_predict(
        model=model_2_3,
        df_TEST=df[df.index <= 8660],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict_B = test_predict(
        model=model_7,
        df_TEST=df[df.index > 8660],
        rock_type=None,
        TEST_folder=TEST_folder,
    )

    y_predict = pd.concat([y_predict_A, y_predict_B], axis=0)
    y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
    print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

    # plot the prediction and compare
    plot_logs_columns(
        df=df,
        DTSM_pred=y_predict,
        well_name=TEST_well,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{TEST_well}-Prediction",
        plot_save_path=TEST_folder,
        plot_save_format=["png"],
    )

#%% collect all csv and check depth

path = "predictions/TEST_final"

pred = dict()
for f in glob.glob(f"{path}/*/*.csv"):
    f_name = re.split("[/\\\.]", f)[-2][-16:]
    print(f_name)

    df = pd.read_csv(f)
    pred[f_name] = df
print("*" * 50)

#%
path = "to_submit/Team_Name/Leaderboard"

submit = dict()

for f in glob.glob(f"{path}/*.xlsx"):
    f_name = re.split("[/\\\.]", f)[-2]
    print(f_name)

    df = pd.read_excel(f, engine="openpyxl")
    submit[f_name] = df
    # print(df.head())
print("*" * 50)

i = 1
for key in submit.keys():
    print(f"Checking {key}")
    df1 = submit[key]
    df2 = pred[key]
    assert all(df1["Depth"] == df2["Depth"])

    print(df1.head())
    df1["DTSM"] = df2["DTSM_Pred"]
    print(df1.head())
    print("*" * 50)

    df1.to_excel(
        f"to_submit/Team_Geoexpert_Final_Predictions/{key}.xlsx",
        index=False,
        engine="openpyxl",
    )

    # one final check of predictions
    df1.columns = ["Depth", "DTSM_Pred"]
    # plot the prediction and compare
    plot_logs_columns(
        df=las_data_TEST_renamed[f"T{i:0>2}-{key}"],
        DTSM_pred=df1,
        well_name=key,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{key}-Prediction",
        plot_save_path=path,
        plot_save_format=["png"],
    )

    i += 1

print("Fabulous job! You completed the entire ML challenge and output the predictions!")

#%% test 17 wells

# model = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final.pickle")
# # model = read_pkl("predictions/TEST_model_building/7_xgb/model_xgb_7_final_refit.pickle")
# # model = read_pkl("predictions/TEST_model_building/6_3_xgb/model_xgb_6_3_final.pickle")

# for TEST_well in TEST_neighbors_20:

#     TEST_folder = "predictions/evaluate3/17finalvalidation"

#     df = process_las().get_df_by_mnemonics(
#         df=las_data_DTSM_QC2[TEST_well].copy(),
#         target_mnemonics=model["target_mnemonics"],
#         log_mnemonics=["RT"],
#         strict_input_output=True,
#         alias_dict=alias_dict,
#         drop_na=True,
#     )
#     df["CALI"] = np.nan
#     print(df.sample(5))

#     y_predict = test_predict(
#         model=model,
#         df_TEST=df,
#         rock_type=None,
#         TEST_folder=TEST_folder,
#     )

#     y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
#     print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

#     # plot the prediction and compare
#     plot_logs_columns(
#         df=las_data_DTSM_QC2[TEST_well],
#         DTSM_pred=y_predict,
#         well_name=TEST_well,
#         alias_dict=alias_dict,
#         plot_show=False,
#         plot_return=False,
#         plot_save_file_name=f"{TEST_well}-Prediction",
#         plot_save_path=TEST_folder,
#         plot_save_format=["png"],
#     )

#     # plot crossplot to compare y_predict vs y_actual
#     plot_crossplot(
#         y_actual=df["DTSM"].values,
#         y_predict=y_predict["DTSM_Pred"].values,
#         include_diagnal_line=True,
#         text=None,
#         plot_show=False,
#         plot_return=False,
#         plot_save_file_name=f"{TEST_well}-Prediction-Crossplot",
#         plot_save_path=TEST_folder,
#         plot_save_format=["png"],
#     )


# #%% test 17 wells, rock type

# model = read_pkl(
#     "predictions/TEST_model_building/7_xgb/model_xgb_7_rocktype_final.pickle"
# )

# for TEST_well in TEST_neighbors_20:

#     TEST_folder = "predictions/evaluate3/17finalvalidation/{TEST_well}"

#     target_mnemonics_ = [i for i in model["target_mnemonics"] if i != "rock_type"]

#     df = process_las().get_df_by_mnemonics(
#         df=las_data_DTSM_QC2[TEST_well].copy(),
#         target_mnemonics=target_mnemonics_,
#         log_mnemonics=["RT"],
#         strict_input_output=True,
#         alias_dict=alias_dict,
#         drop_na=True,
#     )
#     # print(df.sample(5))

#     if TEST_well == "109-84967b1f42e0_TGS":
#         rock_type = np.ones((len(df), 1))
#     else:
#         rock_type = np.zeros((len(df), 1))

#     y_predict = test_predict(
#         model=model,
#         df_TEST=df,
#         rock_type=rock_type,
#         TEST_folder=TEST_folder,
#     )

#     y_predict.to_csv(f"{TEST_folder}/Final_prediction_{TEST_well}.csv")
#     print(f"Predictions for {TEST_well} are saved at: {TEST_folder}\n\n")

#     # plot the prediction and compare
#     plot_logs_columns(
#         df=las_data_DTSM_QC2[TEST_well],
#         DTSM_pred=y_predict,
#         well_name=TEST_well,
#         alias_dict=alias_dict,
#         plot_show=False,
#         plot_return=False,
#         plot_save_file_name=f"{TEST_well}-Prediction",
#         plot_save_path=TEST_folder,
#         plot_save_format=["png"],
#     )

#     # plot crossplot to compare y_predict vs y_actual
#     plot_crossplot(
#         y_actual=df["DTSM"].values,
#         y_predict=y_predict["DTSM_Pred"].values,
#         include_diagnal_line=True,
#         text=None,
#         plot_show=False,
#         plot_return=False,
#         plot_save_file_name=f"{TEST_well}-Prediction-Crossplot",
#         plot_save_path=TEST_folder,
#         plot_save_format=["png"],
#     )
