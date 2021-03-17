#%% this module is to used test different models
import pickle
import time
import os

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
import xgboost
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor as XGB
from tqdm import tqdm

from load_pickle import (
    alias_dict,
    las_data_DTSM_QC2,
    las_data_DTSM_felix,
    KMeans_model_3clusters,
    KMeans_model_5clusters,
    feature_model,
    TEST_neighbors_20,
)
from plot import plot_crossplot, plot_feature_important, plot_logs_columns

pio.renderers.default = "browser"

# load customized functions and requried dataset
from util import MeanRegressor, process_las, read_pkl, to_pkl, predict_zones


#%% Batch tuning XGB using RandomizedSearchCV
path = f"predictions/tuning_final"
KMeans_model = None

if not os.path.exists(path):
    os.mkdir(path)

# create a dictionary to save all the models

time0 = time.time()
feature_model_ = {"2_3": feature_model["2_3"]}  # on search for one model '7' etc.
for model_name, target_mnemonics in feature_model_.items():

    # try:
    #     las_dict = read_pkl(f"{path}/las_dict_{model_name}.pickle")
    # except:
    # train_test_split among depth rows
    las_dict = process_las().get_compiled_df_from_las_dict(
        las_data_dict=las_data_DTSM_QC2,
        target_mnemonics=target_mnemonics,
        log_mnemonics=["RT"],
        strict_input_output=True,
        outliers_contamination=None,
        alias_dict=alias_dict,
        return_dict=True,
        drop_na=True,
    )

    # save the las_dict
    to_pkl(las_dict, f"{path}/las_dict_{model_name}.pickle")

    Xy_ = []
    # groups = []
    for las_name, df in las_dict.items():
        # make sure no test data used in hyper parameter tuning
        # when testing different models on 17 test wells
        # if (
        #     las_name not in TEST_neighbors_20
        # ):
        Xy_.append(df)
    Xy = pd.concat(Xy_)
    print("Xy:\n", Xy.sample(5))

    # scale before zoning
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train_ = scaler_x.fit_transform(Xy.iloc[:, :-1])
    y_train_ = scaler_y.fit_transform(Xy.iloc[:, -1:])

    # predict zones
    zones = predict_zones(df=Xy, cluster_model=None)
    zone_ids = np.unique(zones)

    model_dict_zones = dict()
    for zone_id in zone_ids:

        # reset model_dict for each zone_id
        model_dict = dict()

        # get data for each specific zone and scale the data

        # scale after zoning
        # scaler_x, scaler_y = RobustScaler(), RobustScaler()
        # X_train = scaler_x.fit_transform(Xy.iloc[:, :-1][zones == zone_id])
        # y_train = scaler_y.fit_transform(Xy.iloc[:, -1:][zones == zone_id])

        X_train = X_train_[zones == zone_id]
        y_train = y_train_[zones == zone_id]

        print(f"zone_id: {zone_id}, data rows: {len(X_train)}", "\n", "*" * 70)

        # Baseline model, y_mean and linear regression

        models_baseline = {"MLR": LinearRegression()}

        for model_name_, model in models_baseline.items():
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=10,
                scoring="neg_root_mean_squared_error",
            )
            print(f"{model_name_} rmse:\t{-np.mean(scores):.2f}")
            model_dict[f"rmse_{model_name_}"] = -np.mean(scores)
            # if model_name_ == "MLR":
            #     model_dict[f"estimator_{model_name_}"] = model

        # RandomizedSearchCV to find best hyperparameter combination
        param_distributions = {
            "n_estimators": range(100, 300, 20),
            "max_depth": range(1, 6),
            "min_child_weight": np.arange(0.01, 0.4, 0.01),
            "learning_rate": np.logspace(-3, -1),
            "subsample": np.arange(0.8, 1.02, 0.02),
            "colsample_bytree": np.arange(0.8, 1.02, 0.02),
            "gamma": range(0, 10),
        }

        RandCV = RandomizedSearchCV(
            estimator=XGB(
                tree_method="gpu_hist", objective="reg:squarederror", random_state=42
            ),
            param_distributions=param_distributions,
            n_iter=60,
            scoring="neg_root_mean_squared_error",
            cv=10,
            refit=True,
            random_state=42,
            verbose=2,
        )

        RandCV.fit(X=X_train, y=y_train)

        # save all the results
        model_dict["zone_id"] = zone_id
        model_dict["zone_model"] = KMeans_model
        model_dict["model_name"] = model_name
        model_dict["best_estimator"] = RandCV.best_estimator_
        model_dict["best_params"] = RandCV.best_params_
        model_dict["target_mnemonics"] = target_mnemonics
        model_dict["scaler_x"] = scaler_x
        model_dict["scaler_y"] = scaler_y
        model_dict["rmse_CV"] = -RandCV.best_score_

        model_dict_zones[f"zone_{zone_id}"] = model_dict

        print(
            f"\nCompleted training and saved model in {path} in {time.time()-time0:.1f} seconds!"
        )

        # first, get the best_estimator
        best_estimator = model_dict["best_estimator"]

        try:
            # get the feature_importance data
            plot_feature_important(
                best_estimator=best_estimator,
                features=target_mnemonics,
                plot_save_name=f"feature_importance_{model_name}_zone_{zone_id}",
                path=path,
            )
        except:
            print("No feature importance plotted!")

        # calculate y_pred, plot crossplot pred vs true
        # X_train already scaled! Not need to scale again!
        y_predict = best_estimator.predict(X_train).reshape(-1, 1)

        y_true = scaler_y.inverse_transform(y_train).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_predict).reshape(-1, 1)

        try:
            plot_crossplot(
                y_actual=y_true,
                y_predict=y_pred,
                text=None,
                axis_range=12,
                include_diagnal_line=True,
                plot_show=False,
                plot_return=False,
                plot_save_file_name=f"cross_plot_XGB_{model_name}_zone_{zone_id}",
                plot_save_path=path,
                plot_save_format=["png"],
            )
        except:
            print("No crossplot!")

    # save all models to pickle file during each iteration, for later prediction
    to_pkl(model_dict_zones, f"{path}/model_xgb_{model_name}.pickle")

# for val in model_dict_zones.values():
#     print(
#         f"zone_{val['zone_id']} MLR, XGB: {val['rmse_MLR']:.4f} - {val['rmse_CV']:.4f}"
#     )
# model_dict_zones["zone_1"]
# read_pkl(f"{path}/model_xgb_6_2_zone_1.pickle")
# #%% plot log in columns
# for zone_id in zone_ids:
#     print('zone_id': zone_id)
#     df = Xy[zones == zone_id]
#     df.reset_index(inplace=True, drop=True)

#     model_dict = model_dict_zones[f"zone_{zone_id}"]
#     model = model_dict["best_estimator"]

#     y_predict = model.predict(
#         model_dict["scaler_x"].transform(df.iloc[:, :-1])
#     ).reshape(-1, 1)
#     y_predict = model_dict["scaler_y"].inverse_transform(y_predict)

#     df_ypred = pd.DataFrame(
#         np.c_[df.index.values.reshape(-1, 1), y_predict.reshape(-1, 1)],
#         columns=["Depth", "DTSM_Pred"],
#     )

#     try:
#         plot_logs_columns(
#             df=df,
#             DTSM_pred=df_ypred,
#             well_name=f"Zone_{zone_id}",
#             alias_dict=alias_dict,
#             plot_show=False,
#             plot_return=False,
#             plot_save_file_name=f"Xy_{model_name}-{zone_id}",
#             plot_save_path=path,
#             plot_save_format=["png"],
#         )
#     except:
#         print("No logs columns plotted!")

#%% check models
# import glob

# path = f"predictions/tuning3/n_iter=100cv=10"

# params = []
# for f in glob.glob(f"{path}/*.pickle"):
#     model = read_pkl(f)
#     params.append(
#         [
#             model["zone_9"]["model_name"],
#             model["zone_9"]["target_mnemonics"],
#             f"{model['zone_9']['rmse_CV']:.4f}",
#             # model["zone_9"]["best_params"],
#         ]
#     )

# params = pd.DataFrame(params, columns=["model_name", "target_mnemonics", "rmse_CV"])
# params.to_csv(f"{path}/rmse_CV.csv")
