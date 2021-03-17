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
    las_data_DTSM_QC,
    las_data_DTSM_QC2,
    las_data_DTSM_felix,
    las_lat_lon,
    alias_dict,
    las_depth,
    las_lat_lon,
    test_list,
    TEST_neighbors_20,
    model_xgb_rock_class_6_2,
    rock_info,
)

# load customized functions and requried dataset
from util import (
    to_pkl,
    read_pkl,
    add_gradient_features,
    get_sample_weight,
    get_sample_weight2,
    process_las,
    get_nearest_neighbors,
    assign_rock_type,
)

pio.renderers.default = "browser"
rng = np.random.RandomState(42)
#%% LOOCV


def LOOCV_evaluate(
    target_mnemonics=None,
    models=None,
    add_features=False,
    add_rock_type=False,
    path=None,
    las_data_DTSM=None,
    plot_all=False,
):

    if not os.path.exists(path):
        os.mkdir(path)

    # evaluate models with Leave One Out Cross Validation (LOOCV)
    # setup recording rmse_las for each model_dict
    rmse_all_las = []
    rmse_all_las_temp = []

    # get training and test (las_name) data
    try:
        las_dict = read_pkl(f"{path}/las_dict_.pickle")
    except:
        las_dict = process_las().get_compiled_df_from_las_dict(
            las_data_dict=las_data_DTSM,
            target_mnemonics=target_mnemonics,
            log_mnemonics=["RT"],
            strict_input_output=True,
            outliers_contamination=None,
            alias_dict=alias_dict,
            return_dict=True,
            drop_na=True,
        )
        to_pkl(las_dict, f"{path}/las_dict.pickle")
    print("las_dict length:", len(las_dict))

    # target_mnemonics_rock = (
    # target_mnemonics  # model_xgb_rock_classification["target_mnemonics"]
    # )
    # evaluate the model
    # for _, las_name in test_list.itertuples():
    for las_name in ["109-84967b1f42e0_TGS"]:  # TEST_neighbors_20:

        try:
            Xy_train = read_pkl(f"{path}/Xy_train_.pickle")
        except:
            # create training dataset
            Xy_train = []
            for k, v in las_dict.items():
                # print(v.shape)
                # if k in list_109:
                if k != las_name:
                    if add_features:
                        # add the difference/gradients except DEPTH and DTSM as features to only X
                        target_mnemonics_ = target_mnemonics.copy()
                        target_mnemonics_.remove("DTSM")
                        v1 = add_gradient_features(v[target_mnemonics_])
                        v = pd.concat([v1, v[["DTSM"]]], axis=1).iloc[1:-1, :]

                        # target_mnemonics should include DTSM, but not 'rock_type'
                        target_mnemonics = v.columns.tolist()[:-1]

                        # print(v.shape)
                        # print(k, v.columns)  # , v.isnull().sum())
                    # if k in [
                    #     "015-0f7a4609731a_TGS",
                    #     "020-146b023afbcf_TGS",
                    #     "137-9a136bc9d56d_TGS",
                    # ]:

                    v = assign_rock_type(
                        df=v,
                        las_name=k,
                        info=rock_info,
                    )

                    Xy_train.append(v)
            Xy_train = pd.concat(Xy_train, axis=0)
            to_pkl(Xy_train, f"{path}/Xy_train.pickle")
        # print("Xy_train:", Xy_train.sample(10))

        las_dict = read_pkl(f"{path}/las_dict.pickle")
        Xy_test = las_dict[las_name]

        # print(Xy_train.shape, Xy_test.shape)
        # print(Xy_train.sample(3), Xy_test.sample(3))

        # reset rmse_las for each model_dict
        rmse_las = []
        y_predict_models = []

        for model_name, model in models.items():

            time0 = time.time()

            # clone the mdoel without the parameters
            model = clone(model)
            # print("model:", model)

            X_train_rock = Xy_train[["rock_type"]]
            X_train = Xy_train[target_mnemonics].iloc[:, :-1]
            y_train = Xy_train[target_mnemonics].iloc[:, -1:]

            model_rock_class = clone(
                XGBC(
                    **model_xgb_rock_class_6_2["best_params"],
                    eval_metric="logloss",
                )
            )
            model_rock_class.fit(
                X_train[[i for i in target_mnemonics if i not in ["DEPTH", "DTSM"]]],
                X_train_rock,
            )

            # print("X_train:", X_train.sample(3))
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            y_train = scaler_y.fit_transform(y_train)

            # add the rock type after scaling
            if add_rock_type:
                X_train = np.c_[X_train, X_train_rock.values.reshape(-1, 1)]
                print(
                    "X_train sum of rock_type:",
                    sum(X_train[:, -1:]),
                    len(X_train),
                    sum(X_train[:, -1:]) / len(X_train),
                )
            model.fit(X_train, y_train)

            # scale test data and predict, and scale back prediction
            X_test = Xy_test.iloc[:, :-1]
            y_test = Xy_test.iloc[:, -1:]

            # print("X_test:", X_test.sample(3))
            X_test_rock = model_rock_class.predict(
                X_test[[i for i in target_mnemonics if i not in ["DEPTH", "DTSM"]]]
            )  # np.ones((len(X_test), 1))  #

            # check the rock type for X_test
            print(
                "X_test_rock:",
                sum(X_test_rock),
                len(X_test_rock),
                sum(X_test_rock) / len(X_test_rock),
            )

            # find the mode of predicted rock type
            # X_test_rock = pd.DataFrame(X_test_rock, columns=["rock"])
            # X_test_rock["rock"] = X_test_rock["rock"].mode()[0]
            # X_test_rock = X_test_rock["rock"].values
            if sum(X_test_rock) / len(X_test_rock) > 0.8:
                X_test_rock = np.ones((len(X_test), 1))
            else:
                X_test_rock = np.zeros((len(X_test), 1))

            # add the difference/gradients except DEPTH as features to only X
            if add_features:
                X_test = add_gradient_features(X_test)
            # target_mnemonics_feature_importance = X_test.columns

            # print("X_test:", X_test.shape, X_test)

            X_test = scaler_x.transform(X_test)
            if add_rock_type:
                X_test = np.c_[X_test, X_test_rock.reshape(-1, 1)]
                print(
                    "X_test sum of rock_type:",
                    sum(X_test[:, -1:]),
                    len(X_test),
                    sum(X_test[:, -1:]) / len(X_test),
                )
            y_predict = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))

            y_predict_models.append(y_predict)

        y_predict_models = np.stack(y_predict_models, axis=1)
        y_predict = np.mean(y_predict_models, axis=1)

        # calculate rmse_las
        rmse_las = mean_squared_error(y_test, y_predict, squared=False)
        # df_ypred with proper column names as pd.DataFrame is required for proper plotting
        df_ypred = pd.DataFrame(
            np.c_[Xy_test.index.values.reshape(-1, 1), y_predict.reshape(-1, 1)],
            columns=["Depth", "DTSM_Pred"],
        )

        print(f"{las_name} rmse: {rmse_las:.4f} \trun in {time.time()-time0:.1f} s")

        if plot_all:
            # plot crossplot to compare y_predict vs y_actual
            plot_crossplot(
                y_actual=y_test.values,
                y_predict=y_predict,
                include_diagnal_line=True,
                text=None,
                plot_show=False,
                plot_return=False,
                plot_save_file_name=f"{model_name}-{las_name}-Prediction-Crossplot",
                plot_save_path=path,
                plot_save_format=["png"],
            )

            # plot pred vs true DTSM
            plot_logs_columns(
                df=Xy_test,
                DTSM_pred=df_ypred,
                well_name=las_name,
                alias_dict=alias_dict,
                plot_show=False,
                plot_return=False,
                plot_save_file_name=f"{model_name}-{las_name}-Prediction-Depth",
                plot_save_path=path,
                plot_save_format=["png"],
            )

            # get the feature_importance data
            # plot_feature_important(
            #     best_estimator=model,
            #     features=target_mnemonics_feature_importance,
            #     plot_save_name=f"feature_importance_{model_name}",
            #     path=path,
            # )

        # saving rmse for each las prediction
        rmse_all_las_temp.append(rmse_las)
        rmse_all_las.append([las_name, len(df_ypred), rmse_las])
        rmse_mean = np.mean(rmse_all_las_temp)
        print(f"{model_name} model_dict with mean rmse so far: {rmse_mean:.4f}")

    rmse_all_las = pd.DataFrame(rmse_all_las, columns=["las_name", "rows", "rmse"])

    try:
        rmse_all_las.to_csv(f"{path}/rmse_all_las_{model_name}.csv")
    except:
        rmse_all_las.to_csv(
            f"{path}/rmse_all_las_{model_name}_{str(np.random.rand())[2:]}.csv"
        )
        print("*" * 75)
        print(f"Permission denied: {path}/rmse_all_las_{model_name}.csv")
        print("*" * 75)

    rmse_corrected = (
        sum((rmse_all_las["rows"] * rmse_all_las["rmse"] ** 2))
        / sum(rmse_all_las["rows"])
    ) ** 0.5
    print(f"{model_name} model_dict with corrected rmse : {rmse_corrected:.4f}")

    return [rmse_mean, rmse_corrected]


#%%  LOOCV_evaluate

# list_109 = pd.read_csv("data/DTCO_DTSM_corr.csv")
# list_109 = list_109[list_109["intercept"] < -0]["WellName"].tolist()
# list_109.remove("109-84967b1f42e0_TGS")
# # 6_1 is the best with rmse=12.7749

model_name = "7"
algorithm_name = "xgb"

path = f"predictions/evaluate3/{model_name}_{algorithm_name}"

if not os.path.exists(path):
    os.mkdir(path)

# import model best_params
model_path = f"models/models_TEST"

for f in glob.glob(f"{model_path}/*.pickle"):
    model = read_pkl(f)
    if model["zone_9"]["model_name"] == model_name:
        print(
            model["zone_9"]["model_name"],
            model["zone_9"]["target_mnemonics"],
            f"{model['zone_9']['rmse_CV']:.3f}",
            "\n",
        )

        target_mnemonics = model["zone_9"]["target_mnemonics"][0]
        params_xgb = model["zone_9"]["best_params"]

model = XGB(
    **params_xgb,
    objective="reg:squarederror",
    tree_method="gpu_hist",  # "exact",  #
    deterministic_histogram=True,
    random_state=rng,
)

model_dicts = {
    f"{model_name}_{algorithm_name}": {
        "models": {f"{model_name}_{algorithm_name}": model},
        "target_mnemonics": target_mnemonics,
    }
}

# target_mnemonics.remove("DEPTH")

for model_name, model_dict in model_dicts.items():

    print(model_name, model_dict["target_mnemonics"])
    print("\n")

    time0 = time.time()

    rmse_LOOCV = LOOCV_evaluate(
        target_mnemonics=model_dict["target_mnemonics"],
        models=model_dict["models"],
        add_features=False,
        add_rock_type=True,
        path=path,
        las_data_DTSM=las_data_DTSM_QC2,
        plot_all=True,
    )

    print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")
    print(f"Prediction results are saved at: {path}")
