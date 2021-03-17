# -*- coding: utf-8 -*-
"""
Created on Wed Feb 02 18:54:26 2021

@author: Pritesh Bhoumick
"""

# libraries
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lasio
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


# Specify input parameters for training and testing
model_fn = "model.pkl"  # Trained model pickle file name

missing_value = -9999.25  # Missing value representation
missingness_thresh = 0.2  # Threshold for column missingness in pu

# List of mnemonics to be used for modelling including response log
vars_to_use = ["RESD", "RESM", "DTCO", "DTSM", "NPHI", "RHOB", "GR"]

response_var = "DTSM"  # Response log mnemonic

# Predictor logs
pred_vars = [x for x in vars_to_use if x != response_var]

# Logs that cannot have negative values
nonneg_vars = ["RESD", "RESM", "DTCO", "DTSM", "RHOB", "GR"]

# Lagging window length in ft.
res_lags = [4, 6, 8]
gr_lags = [1, 2, 3, 4, 6]
nphi_lags = [1, 3, 5]
rhob_lags = [2, 3, 5]
dtco_lags = [3, 4, 5, 6, 7]

# Rolling mean window length in ft.
res_win = [4, 6]
gr_win = [3, 5]
nphi_win = [4, 6]
dtco_win = [4, 6, 8]
rhob_win = [3, 5]

# dataframe to store all mnemonic details
mnemonics_df = pd.DataFrame(
    columns=[
        "FILE",
        "LOG",
        "UNIT",
        "DESC",
        "COUNT",
        "MEAN",
        "STD",
        "MIN",
        "25%",
        "50%",
        "75%",
        "MAX",
        "MISSINGNESS",
    ]
)


# Identify the best log in a dict
def find_best_logs(df, log_dict, response_var):
    for k, v in log_dict.items():
        if len(v) > 1:
            corr = df[v].corrwith(df[response_var])
            best_log = corr.idxmax()
            log_dict[k] = [best_log]

    return log_dict


# Identify which logs have multiple mappings in a dict
def identify_duplicate_log(log_dict):
    ret_dict = collections.defaultdict(list)
    for k, v in log_dict.items():
        ret_dict[v[0]].append(k)

    return ret_dict


# Rename and shortlist logs
def log_renaming_shortlisting(df, log_map, response_var):
    col_maps = {}
    try:
        for col in df.columns:
            new_col = log_map.loc[log_map["LOG"] == col, "CATEGORY"].values
            if pd.notna(new_col):
                col_maps[col] = new_col

        col_maps = identify_duplicate_log(col_maps)
        col_maps = find_best_logs(df, col_maps, response_var)
        col_maps = {v[0]: k for k, v in col_maps.items()}
        df = df[df.columns.intersection(list(col_maps.keys()))]
        df = df.rename(columns=col_maps)

    except Exception as e:
        print(e)

    return df


# Impute missing data for missingness below a threshold
def impute_missing_data(df, thresh):
    imputer = IterativeImputer()
    low_missing_cols = df.columns[df.isnull().mean() < thresh]
    high_missing_cols = df.columns[df.isnull().mean() > thresh]
    for cols in low_missing_cols:
        df[cols] = imputer.fit_transform(df[[cols]])

    return df, high_missing_cols


# Remove negative values for specified logs
def remove_negatives(df, cols):
    df[df[cols] < 0] = 0

    return df


# Create lagged features and rolling mean window for logs
def create_lag_features(df, param, lags=None, wins=None):
    # Create lagged features
    lag_cols = [param + "_lag_" + str(lag) for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[param].shift(2 * lag)

    # Create rolling window features
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            df[param + "_rmean_" + str(lag) + "_" + str(win)] = df[lag_col].transform(
                lambda x: x.rolling(2 * win).mean()
            )

    return df


# Convert resistivitiy log to log scale
def convert_res_to_log(df):
    df["RESM"] = np.log10(df["RESM"])
    df["RESD"] = np.log10(df["RESD"])

    return df


# Apply minmax scaler
def apply_minmaxscaler(df, cols):
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(df)
    scaler_transformed = scaler.transform(df)
    scaler_transformed = pd.DataFrame(scaler_transformed, columns=cols)

    return scaler_transformed, scaler_fit


# Normalize test data
def normalize_test(df, scaler):
    cols = df.columns
    normalized_data = scaler.transform(df)
    normalized_data = pd.DataFrame(normalized_data, columns=cols)

    return normalized_data


# Power transform columns (Not used in the code)
def powertransform_cols(df):
    cols = df.columns
    scaler = PowerTransformer(method="yeo-johnson")
    ct = ColumnTransformer([("transform", scaler, cols)], remainder="passthrough")
    PowerTransformer_data = ct.fit_transform(df)
    PowerTransformer_data = pd.DataFrame(PowerTransformer_data, columns=cols)

    return PowerTransformer_data, scaler


# Inverse transform normalized columns
def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


# Remove outliers from dataset
def outlier_detection(df):
    # Standard Deviation Method
    well_train_std = df[np.abs(df - df.mean()) <= (3 * df.std())]

    # Remove all rows with NaNs
    well_train_std = well_train_std.dropna()

    # Isolation Forest
    iso = IsolationForest(contamination=0.5)
    yhat = iso.fit_predict(df)
    mask = yhat != -1
    well_train_iso = df[mask]

    # Minimum Covariance Determinant
    ee = EllipticEnvelope(contamination=0.1)
    yhat = ee.fit_predict(df)
    mask = yhat != -1
    well_train_ee = df[mask]

    # Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.3)
    yhat = lof.fit_predict(df)
    mask = yhat != -1
    well_train_lof = df[mask]

    # One-class SVM
    svm = OneClassSVM(nu=0.1)
    yhat = svm.fit_predict(df)
    mask = yhat != -1
    well_train_svm = df[mask]

    print("Number of points before outliers removed                     :", len(df))
    print(
        "Number of points after outliers removed with Standard Deviation:",
        len(well_train_std),
    )
    print(
        "Number of points after outliers removed with Isolation Forest  :",
        len(well_train_iso),
    )
    print(
        "Number of points after outliers removed with Min. Covariance   :",
        len(well_train_ee),
    )
    print(
        "Number of points after outliers removed with Outlier Factor    :",
        len(well_train_lof),
    )
    print(
        "Number of points after outliers removed with One-class SVM     :",
        len(well_train_svm),
    )

    # Commented plotting code; Uncomment if required to be plotted
    # plt.figure(figsize=(13,10))

    # plt.subplot(3,2,1)
    # df.boxplot()
    # plt.title('Before Outlier Removal', size=15)

    # plt.subplot(3,2,2)
    # well_train_std.boxplot()
    # plt.title('After Outlier Removal with Standard Deviation Filter', size=15)

    # plt.subplot(3,2,3)
    # well_train_iso.boxplot()
    # plt.title('After Outlier Removal with Isolation Forest', size=15)

    # plt.subplot(3,2,4)
    # well_train_ee.boxplot()
    # plt.title('After Outlier Removal with Min. Covariance', size=15)

    # plt.subplot(3,2,5)
    # well_train_lof.boxplot()
    # plt.title('After Outlier Removal with Local Outlier Factor', size=15)

    # plt.subplot(3,2,6)
    # well_train_svm[vars_to_use].boxplot()
    # plt.title('After Outlier Removal with One-class SVM', size=15)

    # plt.tight_layout(1.7)
    # plt.show()

    return well_train_svm  # Return SVM output as it was found to be most effective


# Decaying learning rate generator
def learning_rate_decay_power(current_iter, learn_rate, decay_rate):
    base_learning_rate = learn_rate
    lr = base_learning_rate * np.power(decay_rate, current_iter)
    return lr if lr > 1e-3 else 1e-3


# Plot well logs
def log_plot(logs, well):
    logs = logs.sort_values(by="DEPTH")
    top = logs["DEPTH"].min()
    bot = logs["DEPTH"].max()

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 8))
    ax[0].plot(logs.GR, logs["DEPTH"], color="green")
    ax[1].plot(logs.RESD, logs["DEPTH"], color="red")
    ax[2].plot(logs.NPHI, logs["DEPTH"], color="black")
    ax[3].plot(logs.RHOB, logs["DEPTH"], color="c")
    ax[4].plot(logs.DTCO, logs["DEPTH"], color="blue")
    ax[5].plot(logs.DTSM, logs["DEPTH"], color="m")

    for i in range(len(ax)):
        ax[i].set_ylim(top, bot)
        ax[i].invert_yaxis()
        ax[i].grid()

    ax[0].set_xlabel("GR")
    ax[0].set_xlim(0, logs.GR.max())
    ax[0].set_ylabel("Depth(ft)")
    ax[1].set_xlabel("RESD")
    ax[1].set_xlim(0, 4)
    ax[2].set_xlabel("NPHI")
    ax[2].set_xlim(-0.4, 0.6)
    ax[3].set_xlabel("RHOB")
    ax[3].set_xlim(logs.RHOB.min(), logs.RHOB.max())
    ax[4].set_xlabel("DTCO")
    ax[4].set_xlim(logs.DTCO.min(), logs.DTCO.max())
    ax[5].set_xlabel("DTSM")
    ax[5].set_xlim(logs.DTSM.min(), logs.DTSM.max())

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])

    f.suptitle("Well: " + well, fontsize=14, y=0.94)


# TRAIN SCRIPT

# Read file names for training data
file_list = []
file_list += [
    file for file in os.listdir(os.curdir + "/train_data") if file.endswith(".las")
]


# Initialize empty var
train_df = pd.DataFrame()
test_df = pd.DataFrame()
val_df = pd.DataFrame()
inputlas = {}

# List out all mnemonics along with the each log's statistical descriptions
for file in file_list:
    inputlas_ = lasio.read("./train_data/" + file)  # Read file
    df = inputlas_.df()  # Convert data to dataframe
    df = df.rename_axis("DEPT").reset_index()  # Create depth axis and reset index
    df = df.replace(missing_value, "")  # Convert missing value validation to null
    df = df.dropna(subset=[response_var])  # Drop rows with no response logs
    des = pd.DataFrame(df.describe())  # Get data stats

    for curves in inputlas_.curves:  # Loop through each curve in the dataset
        curv_desc = [file, curves.mnemonic, curves.unit, curves.descr]
        curv_stats = list(des.loc[:, curves.mnemonic].values)
        missingness = (
            100 * df[curves.mnemonic].isnull().mean()
        )  # Calculate missingness in each column
        curv_desc.extend(curv_stats)
        curv_desc.extend([missingness])
        temp_df = pd.DataFrame(
            [curv_desc],
            columns=[
                "FILE",
                "LOG",
                "UNIT",
                "DESC",
                "COUNT",
                "MEAN",
                "STD",
                "MIN",
                "25%",
                "50%",
                "75%",
                "MAX",
                "MISSINGNESS",
            ],
        )
        temp_df = temp_df[
            temp_df["COUNT"] > 0
        ]  # Only take columns that dont have zero rows
        mnemonics_df = mnemonics_df.append(temp_df)

# Export input mnemonics and corresponding log stats
mnemonics_df.to_excel(
    os.path.dirname(os.path.realpath(__file__)) + "Log_mapping.xlsx", index=False
)

# Read Manually mapped log mnemonics (output from above lines of code)
log_mapping = pd.read_excel("Logs_Mapping.xlsx", sheet_name="Distinct mnemonics")


# Read files and get log stats in dataframe (Data preparation)
for file in file_list:
    inputlas[file] = lasio.read("./train_data/" + file)  # Read file
    print(f"Reading {file}")

    df = inputlas[file].df()  # Convert data to dataframe
    df = df.rename_axis("DEPT").reset_index()  # Create depth axis and reset index
    df = df.replace(missing_value, "")  # Convert missing value validation to null
    df = df.dropna(subset=[response_var])  # Drop rows with no response logs
    des = pd.DataFrame(df.describe())  # Get data stats

    df = df.dropna(axis=1, how="all")  # Drop rows with all NAs

    # Rename log mnemonics to a consistent nomenclature
    df = log_renaming_shortlisting(df, log_mapping, response_var)

    # If we have all columns to be used for modelling
    if all(x in df.columns for x in vars_to_use):
        df = df[vars_to_use]  # Filter logs to be used for modelling

        # Impute missing rows for columns with < 20% missingness
        df, high_missing_cols = impute_missing_data(df, missingness_thresh)

        # Check if not all columns are dropped during missingness check
        if len(high_missing_cols) > 0:
            df = df.dropna(axis=1, how="any")

        if len(df.columns) == len(vars_to_use):  # if the well has all well logs
            df[df["RESD"] <= 0] = 0.01  # Change negative resistivities to low value
            df[df["RESM"] <= 0] = 0.01
            df[df[nonneg_vars] < 0] = 0  # remove negative values

            df = convert_res_to_log(df)  # Convert resistivitiy to log scale

            # Split dataset into test and train
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

            # Create lagged features and rolling mean window for specified logs
            # for test data
            df_test = create_lag_features(df_test, "RESD", lags=res_lags, wins=res_win)
            df_test = create_lag_features(df_test, "RESM", lags=res_lags, wins=res_win)
            df_test = create_lag_features(df_test, "GR", lags=gr_lags, wins=gr_win)
            df_test = create_lag_features(
                df_test, "NPHI", lags=nphi_lags, wins=nphi_win
            )
            df_test = create_lag_features(
                df_test, "RHOB", lags=rhob_lags, wins=rhob_win
            )
            df_test = create_lag_features(
                df_test, "DTCO", lags=dtco_lags, wins=dtco_win
            )

            # Split train into train and validation logs
            df_train, df_val = train_test_split(
                df_train, test_size=0.1, random_state=11
            )

            # Create lagged features and rolling mean window for specified logs
            # for validation data
            df_val = create_lag_features(df_val, "RESD", lags=res_lags, wins=res_win)
            df_val = create_lag_features(df_val, "RESM", lags=res_lags, wins=res_win)
            df_val = create_lag_features(df_val, "GR", lags=gr_lags, wins=gr_win)
            df_val = create_lag_features(df_val, "NPHI", lags=nphi_lags, wins=nphi_win)
            df_val = create_lag_features(df_val, "RHOB", lags=rhob_lags, wins=rhob_win)
            df_val = create_lag_features(df_val, "DTCO", lags=dtco_lags, wins=dtco_win)

            # Remove outliers from train dataset
            df_train = outlier_detection(df_train)

            # Create lagged features and rolling mean window for specified logs
            # for training data
            df_train = create_lag_features(
                df_train, "RESD", lags=res_lags, wins=res_win
            )
            df_train = create_lag_features(
                df_train, "RESM", lags=res_lags, wins=res_win
            )
            df_train = create_lag_features(df_train, "GR", lags=gr_lags, wins=gr_win)
            df_train = create_lag_features(
                df_train, "NPHI", lags=nphi_lags, wins=nphi_win
            )
            df_train = create_lag_features(
                df_train, "RHOB", lags=rhob_lags, wins=rhob_win
            )
            df_train = create_lag_features(
                df_train, "DTCO", lags=dtco_lags, wins=dtco_win
            )

            print(f"Appending {file} to main df")

            # Append prepared data from each well
            train_df = train_df.append(df_train)
            test_df = test_df.append(df_test)
            val_df = val_df.append(df_val)

# Pairplot between all input logs
sns.pairplot(
    train_df,
    vars=vars_to_use,
    diag_kind="kde",
    plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "k"},
)

# Save prepared data into pickle files for easier access
train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")
val_df.to_pickle("val_df.pkl")

# Read datasets from pickle files
train_df = pd.read_pickle(r"train_df.pkl")
val_df = pd.read_pickle(r"val_df.pkl")
test_df = pd.read_pickle(r"test_df.pkl")

# Normalize train data for predictor logs
train_x, scalar_x = apply_minmaxscaler(
    train_df.drop([response_var], axis=1), train_df.drop([response_var], axis=1).columns
)

# Normalize train data for response logs
train_y, scalar_y = apply_minmaxscaler(train_df[[response_var]], [response_var])

# Dump scalar object from normalization
joblib.dump(scalar_x, "scaler_x.pkl")
joblib.dump(scalar_y, "scaler_y.pkl")

# Remove outlier from validation and test data on response variable only
svm = OneClassSVM(nu=0.1)
yhat = svm.fit_predict(pd.DataFrame(val_df[response_var]))
mask = yhat != -1
val_df = val_df[mask]

svm = OneClassSVM(nu=0.1)
yhat = svm.fit_predict(pd.DataFrame(test_df[response_var]))
mask = yhat != -1
val_df = test_df[mask]

# Normalize train data for predictor and response logs
test_x = normalize_test(test_df.drop([response_var], axis=1), scalar_x)
test_y = normalize_test(test_df[[response_var]], scalar_y)

val_x = normalize_test(val_df.drop([response_var], axis=1), scalar_x)
val_y = normalize_test(val_df[[response_var]], scalar_y)


# Hyper param optimization Lightgbm


# Fixed hyperparameters
fit_params = {
    "early_stopping_rounds": 30,
    "eval_metric": "rmse",
    "eval_set": [(val_x, val_y)],  # Validation dataset
    "verbose": 100,
}

# Tuning hyperparameters
param_test = {
    "num_leaves": sp_randint(6, 50),
    "min_child_samples": sp_randint(100, 500),
    "min_child_weight": [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    "subsample": sp_uniform(loc=0.2, scale=0.8),
    "colsample_bytree": sp_uniform(loc=0.4, scale=0.6),
    "reg_alpha": [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    "reg_lambda": [0, 1e-1, 1, 5, 10, 20, 50, 100],
}

# LGBM Regressor for Randomized Search using CV
clf = lgb.LGBMRegressor(
    max_depth=-1,
    random_state=11,
    silent=True,
    metric="rmse",
    n_jobs=-1,
    n_estimators=5000,
)

# Randomized Search using CV
gs = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_test,
    scoring="neg_root_mean_squared_error",
    cv=5,
    refit=True,
    random_state=11,
    verbose=True,
)

# Fit input data into Randomized Search
gs.fit(train_x.reset_index(drop=True), train_y.reset_index(drop=True), **fit_params)

# Print best params after Randomized Search
print(
    "Best score achieved: {} with params: {} ".format(gs.best_score_, gs.best_params_)
)

opt_parameters = gs.best_params_  # Save best hyperparameters

# LGBM regressor for Grid Search using CV
clf_sw = lgb.LGBMRegressor(**clf.get_params())

# set optimal parameters
clf_sw.set_params(**opt_parameters)

# Grid Search using CV
gs_sample_weight = GridSearchCV(
    estimator=clf_sw,
    param_grid={"scale_pos_weight": [1, 2, 6, 12]},
    scoring="neg_root_mean_squared_error",
    cv=5,
    refit=True,
    verbose=True,
)

# Fit input data into Grid Search
gs_sample_weight.fit(
    train_x.reset_index(drop=True), train_y.reset_index(drop=True), **fit_params
)

# Print best hyperparameters after Grid Search
print(
    "Best score achieved: {} with params: {} ".format(
        gs_sample_weight.best_score_, gs_sample_weight.best_params_
    )
)

# LGBM Regressor using best parameters from Randomized Grid Search
clf_final = lgb.LGBMRegressor(**clf.get_params())

# set optimal parameters
clf_final.set_params(**opt_parameters)

# Train the final model with learning rate decay
clf_final.fit(
    train_x,
    train_y,
    **fit_params,
    callbacks=[
        lgb.reset_parameter(
            learning_rate=learning_rate_decay_power(learn_rate=0.01, decay_rate=0.995)
        )
    ],
)

joblib.dump(clf_final, model_fn)  # Save final trained model object

# Plot feature importance
feat_imp = pd.Series(clf_final.feature_importances_, index=train_x.columns)
feat_imp.nlargest(20).plot(kind="barh", figsize=(8, 10))

# Predict output on test data
pred_gbm = clf_final.predict(test_x, n_jobs=-1)

# Inverse transfoem test data
test_y_ = invTransform(scalar_y, test_y, "DTSM", train_df.columns)

# Inverse transform predicted predicted data
pred_gbm = invTransform(scalar_y, pred_gbm, "DTSM", train_df.columns)

# Calculate RMSE of prediced data
rmse = np.sqrt(MSE(test_y_, pred_gbm))
print(f"RMSE from LightGBM model: {rmse}")


# Bidirectional RNN

LR = 0.00005  # Fixed Learning rate
ACTIVATION_FUNC = "relu"  # Activation function to be used
EPOCHS = 25  # Max no. of iterations
BATCH_SIZE = 64  # Batches of data to be used
METRIC_CHOICE = [tf.keras.metrics.RootMeanSquaredError()]  # Metric choice
LOSS = "mse"  # Loss function
OPTIMIZER = Adam(lr=LR)  # Optimizer function

# Define train, validation and test data for bidirectional RNN modelling
train_x_rnn = np.asarray(train_x.replace(np.nan, missing_value)).reshape(
    -1, 1, train_x.shape[1]
)
train_y_rnn = np.asarray(train_x.replace(np.nan, missing_value))
val_x_rnn = np.asarray(val_x.replace(np.nan, missing_value)).reshape(
    -1, 1, val_x.shape[1]
)
val_y_rnn = np.asarray(val_y.replace(np.nan, missing_value))
test_x_rnn = np.asarray(test_x.replace(np.nan, missing_value)).reshape(
    -1, 1, val_x.shape[1]
)
test_y_rnn = test_y.replace(np.nan, missing_value)

# Callback function to stop iteration once validation loss do not reduce further
# and save the best model checkpoint object
callbacks = [
    EarlyStopping(monitor="val_loss", mode=min, patience=3),
    ModelCheckpoint(
        os.listdir(os.curdir + "../rnn_model.h5"),
        save_best_only=True,
        save_weights_only=False,
        monitor="val_loss",
        verbose=1,
    ),
]

# Define the bidirectional RNN layer structure
model_pipeline_input = layers.Input(shape=(1, train_x.shape[1]), dtype="float32")

# Add masking layer to handle NaNs in the dataset
model_pipeline_masking = layers.Masking(mask_value=missing_value)(model_pipeline_input)

model_pipeline = layers.Bidirectional(
    layers.GRU(16, dropout=0.15, return_sequences=True, recurrent_dropout=0.5)
)(model_pipeline_masking)

model_pipeline = layers.Dense(1, activation=ACTIVATION_FUNC)(model_pipeline)
model = Model(inputs=model_pipeline_input, outputs=model_pipeline)

# Compile model with loss and optimizer
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRIC_CHOICE)
model.fit(
    train_x_rnn,
    train_y_rnn,
    validation_data=(val_x_rnn, val_y_rnn),  # Error check model with validation data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Evaluate model performance on validation data
model.evaluate(val_x_rnn, val_y_rnn)

# Predict model with test data
pred_rnn = model.predict(test_x_rnn)

# inverse transofrm prediction
pred_rnn = invTransform(
    scalar_y, pred_rnn.ravel().reshape(-1, 1), response_var, train_df.columns
)

# Inverse transofrm test data
test_y_ = invTransform(scalar_y, test_y, response_var, train_df.columns)

# Calculate RMSE on predicted DTSM
rmse = np.sqrt(MSE(test_y_, pred_rnn))
print(f"RMSE from RNN model: {rmse}")


# TEST SCRIPT

# Read file names for test data
file_list = []
file_list += [
    file for file in os.listdir(os.curdir + "/final_test_data") if file.endswith(".las")
]


# Read normalized scalar objects
scalar_x = joblib.load("scaler_x.pkl")
scalar_y = joblib.load("scaler_y.pkl")


# Read test files
for file in file_list:
    inputlas = lasio.read("./final_test_data/" + file)  # Read file
    print(file)

    df = inputlas.df()  # Convert data to dataframe
    df_length = len(df)  # Dataframe length

    df = df.replace(missing_value, "")  # Convert missing value validation to null

    # Rename log mnemonics to a consistent nomenclature
    df = log_renaming_shortlisting(df, log_mapping, "DTCO")

    # If we have all columns to be used for modelling
    if not all(x in df.columns for x in pred_vars):
        for col in pred_vars:
            if col not in df.columns:
                df[col] = np.nan

    df = df[pred_vars]  # Filter logs to be used for modelling

    # Impute missing rows for columns with < 20% missingness
    df, high_missing_cols = impute_missing_data(df, missingness_thresh)

    df[df[nonneg_vars] < 0] = 0  # set negative values to zero

    # For resistivity logs
    if all(x in df.columns for x in ["RESD", "RESM"]):
        df[df["RESD"] <= 0] = 0.01  # Change negative resistivities to low value
        df[df["RESM"] <= 0] = 0.01

        df = convert_res_to_log(df)  # Convert resistivitiy to log scale

        # Create lagged features and rolling mean window for resistivity logs
        # for test data
        df = create_lag_features(df, "RESD", lags=res_lags, wins=res_win)
        df = create_lag_features(df, "RESM", lags=res_lags, wins=res_win)

    # Create lagged features and rolling mean window for specified logs
    # for test data
    df = create_lag_features(df, "GR", lags=gr_lags, wins=gr_win)
    df = create_lag_features(df, "NPHI", lags=nphi_lags, wins=nphi_win)
    df = create_lag_features(df, "RHOB", lags=rhob_lags, wins=rhob_win)
    df = create_lag_features(df, "DTCO", lags=dtco_lags, wins=dtco_win)

    test_df_norm = normalize_test(df, scalar_x)  # Normalize logs

    test_x = np.asarray(test_df_norm)  # Get test data as array

    m_lgb = joblib.load(model_fn)  # Load trained model object

    print(f"Predicting data for {file}")
    pred = m_lgb.predict(test_x)  # Predict response log

    print(f"Predicted data for {file}. Now inverse transforming")
    # Inverse transform predicted predicted data
    pred = list(invTransform(scalar_y, pred, "DTSM", ["DTSM"]))
    pred_len = len(pred)  # Get predicted data length

    if df_length == pred_len:  # Check if test data and predicted data length match
        print(f"{file} output count passed")

    # Plot logs with predicted output
    df["DTSM"] = pred
    df["DEPTH"] = df.index
    log_plot(df, file.split(".")[0])

    # Save output to xlsx files based on specified format
    well_data = pd.DataFrame({"Depth": df.index, "DTSM": pred})
    well_data.to_excel(
        os.path.dirname(os.path.realpath(__file__))
        + "\\output_files\\"
        + file.split(".")[0]
        + ".xlsx",
        index=False,
    )
