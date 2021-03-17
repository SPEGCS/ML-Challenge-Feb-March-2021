#%% import lib
import pathlib
import pickle
import re

import lasio
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.signal import medfilt
from scipy.ndimage import median_filter
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.covariance import EllipticEnvelope


def to_pkl(data, path):
    """
    'path' should be a path with file name
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return None


def read_pkl(path):
    """
    'path' should be a path with file name
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# given a mnemonic, find all of its alias
def get_alias(mnemonic, alias_dict=None):
    assert alias_dict is not None, "alias_dict is None, assign alias_dict!"
    alias_dict = alias_dict or {}
    return [k for k, v in alias_dict.items() if mnemonic in v]


# given a alias, find its corresponding one and only mnemonic
# return a mnemonis if found, or else ''
def get_mnemonic(alias=None, alias_dict=None):
    assert alias_dict is not None, "alias_dict is None, assign alias_dict!"
    alias_dict = alias_dict or {}
    try:
        return [v for k, v in alias_dict.items() if alias == k][0]
    except:
        return ""


def add_gradient_features(df=None):
    assert isinstance(df, pd.DataFrame)
    assert "DTSM" not in df.columns

    df_original = df.copy()
    df = df.copy()
    cols = [col for col in ["RHOB", "NPHI", "GR", "DTCO"] if col in df.columns]
    # cols = [col for col in ["RHOB", "NPHI", "GR", "DTCO"] if col in df.columns]
    df = df[cols]
    df_gradient = np.gradient(df.values, axis=0)
    df_gradient = pd.DataFrame(df_gradient, columns=df.columns + "_grad")
    df_gradient.index = df.index
    return pd.concat([df_original, df_gradient], axis=1)


def assign_rock_type(df=None, las_name=None, info=None):
    """
    return df['rock_type']=0 if las_name not in info.index
    """

    if las_name not in info.index:
        print(f"{las_name} not in rock_info, assigned rock_type=0!")
        df["rock_type"] = 0
        return df

    df = df.copy()

    if pd.isnull(info.loc[las_name, "rock_median"]):
        assert not pd.isnull(info.loc[las_name, "rock_type"])
        df["rock_type"] = info.loc[las_name, "rock_type"]
    else:
        assert all(
            [
                not pd.isnull(info.loc[las_name, "rock_type"]),
                not pd.isnull(info.loc[las_name, "rock_type2"]),
            ]
        )
        df["rock_type"] = np.nan
        df["rock_type"][df.index < info.loc[las_name, "rock_median"]] = info.loc[
            las_name, "rock_type"
        ]
        df["rock_type"][df.index >= info.loc[las_name, "rock_median"]] = info.loc[
            las_name, "rock_type2"
        ]

    return df


def predict_zones(df=None, cluster_model=None):
    df = df.copy()

    if cluster_model is not None:
        target_mnemonics = ["DTCO", "NPHI", "GR"]
        assert all([i in df.columns for i in target_mnemonics])
        df["labels"] = cluster_model["model"].predict(
            cluster_model["scaler_x"].transform(df[target_mnemonics])
        )
    else:
        df["labels"] = 9

    # if "RHOB" in df.columns and cluster_model is not None:
    #     std_ = df[target_mnemonics].std().values.reshape(-1, 1)
    #     df["labels"] = cluster_model.predict(std_)[0]
    # else:
    #     df["labels"] = 9

    # return a np.array with numbers for zones
    return df["labels"].values


def fit_X_zone(model=None, cluster_model=None, Xy=None, sample_weight=None):

    # cluster_mode can be provided as np.ndarry directly

    # scale the data before zoning
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train_ = scaler_x.fit_transform(Xy.iloc[:, :-1])
    y_train_ = scaler_y.fit_transform(Xy.iloc[:, -1:])

    if sample_weight is None:
        sample_weight = np.ones((len(Xy), 1))

    model_dict = dict()
    # predict zones and build models for each zones
    if cluster_model is not None:
        if isinstance(cluster_model, np.ndarray):
            zones = cluster_model
        else:
            zones = predict_zones(df=Xy, cluster_model=cluster_model)
        zone_ids = np.unique(zones)

        # create individual model for each zone
        for zone_id in zone_ids:
            print(f"zone_id: {zone_id}")
            X_train = X_train_[zones == zone_id]
            y_train = y_train_[zones == zone_id]
            sw_train = sample_weight[zones == zone_id]

            # reset model for each zone_id
            model_ = clone(model)
            model_.fit(X_train, y_train, sw_train)

            # save the fitted model
            model_dict[zone_id] = model_

    # crate model using all data for all zones
    model_ = clone(model)
    model_.fit(X_train_, y_train_, sample_weight)
    model_dict["9"] = model_

    return model_dict, scaler_x, scaler_y


def pred_y_zone(models=None, scalers=None, zone_model=None, X_test=None):
    """
    Sample input format:
    models={'xgb': {'0'=XGB(), '1'=XGB() ...}
            'mlp': {'0'=MLP(), '1'=MLP() ...}
            }
    scalers={'x': scaler_x, 'y': scaler_y'}
    zone_model={'model': KMeans(), 'scaler_x': RobusScaler(), 'target_mnemonics': ['DTCO', 'NPHI', 'GR']}
    X_test=pd.DataFrame()

    Return: y_predict (np.ndarray)
    """

    X_test = X_test.copy()
    y_predict_models = []

    # if data has required curves for zone classification
    if (
        all([i in X_test.columns for i in ["DTCO", "NPHI", "GR"]])
        and zone_model is not None
    ):
        zones = predict_zones(X_test, cluster_model=zone_model)
        zone_ids = np.unique(zones)

        for model_zones in models.values():

            # reset y_predict for each model
            y_predict = X_test.iloc[:, 0:1].copy()

            for zone_id in zone_ids:

                # get the model for zone_id
                model = model_zones[zone_id]

                # get data for zone_id
                X_test_ = X_test.values[zones == zone_id]

                X_test_ = scalers["x"].transform(X_test_)
                y_predict[zones == zone_id] = scalers["y"].inverse_transform(
                    model.predict(X_test_).reshape(-1, 1)
                )

            # append y_predict for each model
            y_predict_models.append(y_predict)

    else:

        X_test_ = scalers["x"].transform(X_test)

        for model_zones in models.values():
            model = model_zones["9"]  # '9' is the overall model
            y_predict = scalers["y"].inverse_transform(
                model.predict(X_test_).reshape(-1, 1)
            )

            # append y_predict for each model
            y_predict_models.append(y_predict)

    y_predict_models = np.stack(y_predict_models, axis=1)
    y_predict = np.mean(y_predict_models, axis=1)

    return y_predict


#%% read las, return curves and data etc.


class read_las:
    def __init__(self, las):
        self.las = lasio.read(las)
        return None

    def df(self):
        return self.las.df()

    def df_curvedata(self):
        l = []
        for i in self.las.curves:
            l.append([i.mnemonic, i.unit, i.descr])
        return pd.DataFrame(l, columns=["mnemonic", "unit", "description"])

    def get_mnemonic_unit(self):
        return self.df_curvedata()[["mnemonic", "unit"]]

    def get_curvedata(self, data_names=[]):
        df = self.df_curvedata()
        return df

    def df_welldata(self, valid_value_only=True):
        l = []
        for i in self.las.well:
            if valid_value_only and (i.value != ""):
                l.append([i.mnemonic, i.unit, i.value, i.descr])
        return pd.DataFrame(l, columns=["mnemonic", "unit", "value", "description"])

    def get_welldata(self, data_names=[]):
        df = self.df_welldata()
        return [df[df["mnemonic"] == i]["value"].values[0] for i in data_names]

    def get_lat_lon(self, data_names=["SLAT", "SLON"]):
        return self.get_welldata(data_names=data_names)

    def get_start_stop(self, data_names=["STRT", "STOP"]):
        return self.get_welldata(data_names=data_names)


#% TEST: read_las()
# convert las.curves info to df for better reading

if __name__ == "__main__":
    las = "data/las/0052442d0162_TGS.las"
    las = read_las(las)
    df = las.df()
    print(df)
    print(las.df_curvedata())
    print(las.df_welldata())
    print(las.get_mnemonic_unit())
    print("lat and lon:", las.get_lat_lon())
    print("start-stop depth:", las.get_start_stop())

#%%
def sample_weight_calc(length=1, decay=0.999):

    assert all([decay > 0, decay <= 1])
    assert all([length >= 1, type(length) is int])
    return decay ** np.arange(length, 0, step=-1)


def get_distance(a, b):
    """
    a and b are lists: [lat, lon] of two locations
    """
    assert isinstance(a, list)
    assert isinstance(b, list)
    assert all([len(a) == 2, len(b) == 2])

    return np.sum(np.square(np.array(a) - np.array(b))) ** 0.5


def get_sample_weight(las_name=None, las_dict=None, las_lat_lon=None):
    """
    sample weight based on horizontal distance between wells
    """
    assert (las_name, str)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    sample_weight = []
    for k in sorted(las_dict.keys()):
        if k not in [las_name]:
            sample_weight = sample_weight + [
                1 / get_distance(las_lat_lon[las_name], las_lat_lon[k])
            ] * len(las_dict[k])
    sample_weight = np.array(sample_weight).reshape(-1, 1)
    sample_weight = MinMaxScaler().fit_transform(sample_weight)

    return sample_weight


def get_distance_weight(las_name=None, las_dict=None, las_lat_lon=None):
    """
    sample weight based on horizontal distance between wells
    """
    assert (las_name, str)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    distance_weight = dict()

    for k in sorted(las_dict.keys()):
        if k not in [las_name]:
            distance_weight[k] = (
                1 / get_distance(las_lat_lon[las_name], las_lat_lon[k]) * 10
            )
            if distance_weight[k] > 10:
                distance_weight[k] = 10

    distance_weight[las_name] = max(distance_weight.values()) * 2

    return distance_weight


def get_sample_weight2(
    las_name=None,
    las_dict=None,
    vertical_anisotropy=0.01,
    las_lat_lon=None,
):
    """
    sample weight based on horizontal and vertical distance between wells
    """
    assert (las_name, str)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    sample_weight1 = []
    sample_weight2 = []
    for k in sorted(las_dict.keys()):
        if k not in [las_name]:
            sample_weight1 = sample_weight1 + [
                1 / get_distance(las_lat_lon[las_name], las_lat_lon[k])
            ] * len(las_dict[k])

            avg = np.mean(las_dict[las_name].index)
            vertical_distance = list(abs(np.array(las_dict[k].index) - avg))
            sample_weight2 = sample_weight2 + vertical_distance

    sample_weight1 = np.array(sample_weight1).reshape(-1, 1)
    sample_weight1 = MinMaxScaler().fit_transform(sample_weight1)

    sample_weight2 = np.array(sample_weight2).reshape(-1, 1)
    sample_weight2 = MinMaxScaler().fit_transform(sample_weight2)

    sample_weight = np.sqrt(
        sample_weight1 ** 2 + (1 / vertical_anisotropy * sample_weight2) ** 2
    )

    return sample_weight


def get_sample_weight3(df=None, depth_range=None, decay=1):
    assert isinstance(depth_range, list)

    df = df.copy()
    depth_range = [7000, 8000]
    depth_min = min(depth_range)
    depth_max = max(depth_range)
    df["sample_weight"] = 1

    df["sample_weight"].loc[df.index < depth_min] = abs(
        df.index[df.index < depth_min] - depth_min
    )
    df["sample_weight"].loc[df.index > depth_max] = abs(
        df.index[df.index > depth_max] - depth_max
    )
    df["sample_weight"] = np.power(decay, df["sample_weight"])

    return df["sample_weight"].values


def get_nearest_neighbors(
    depth_TEST=None,
    lat_lon_TEST=None,
    las_depth=None,
    las_lat_lon=None,
    num_of_neighbors=20,
    vertical_anisotropy=1,
    depth_range_weight=0.1,
):
    assert isinstance(
        depth_TEST, list
    ), "depth_TEST should be a list with [min_depth, max_depth]"
    assert isinstance(las_depth, dict)
    assert isinstance(
        lat_lon_TEST, list
    ), "las_lon_TEST should be a list with lat and lon"
    num_of_neighbors = int(num_of_neighbors)

    depth_rank = []
    for key, val in las_depth.items():
        d1 = abs(np.mean(depth_TEST) - np.mean(val))
        d2 = max(depth_TEST) - min(depth_TEST)
        if lat_lon_TEST is not None:
            d3 = get_distance(lat_lon_TEST, las_lat_lon[key])
        else:
            d3 = 0

        depth_rank.append([key, d1, d2, d3])

    depth_rank = pd.DataFrame(depth_rank, columns=["WellName", "d1", "d2", "d3"])
    for col in depth_rank.columns[1:]:
        scaler = MinMaxScaler()
        depth_rank[col] = scaler.fit_transform(depth_rank[[col]])

    depth_rank["d"] = (
        depth_rank["d1"] - (depth_rank["d2"] * depth_range_weight)
    ) + depth_rank["d3"] * vertical_anisotropy

    if num_of_neighbors == 0:
        nn = depth_rank.iloc[:, 0:1].values
    else:
        nn = (
            depth_rank.sort_values(by="d", axis=0, ascending=True)
            .iloc[1 : num_of_neighbors + 1, 0:1]
            .values
        )
    return nn


def get_nearest_neighbors_2(
    lat_lon_TEST=None,
    las_lat_lon=None,
    num_of_neighbors=20,
):

    assert isinstance(
        lat_lon_TEST, list
    ), "las_lon_TEST should be a list with lat and lon"
    num_of_neighbors = int(num_of_neighbors)

    depth_rank = []
    for key in las_lat_lon.keys():
        d = get_distance(lat_lon_TEST, las_lat_lon[key])

        depth_rank.append([key, d])

    depth_rank = pd.DataFrame(depth_rank, columns=["WellName", "d"])

    if num_of_neighbors == 0:
        nn = depth_rank.iloc[1:, 0:1].values
    else:
        nn = (
            depth_rank.sort_values(by="d", axis=0, ascending=True)
            .iloc[1:num_of_neighbors, 0:1]
            .values
        )
    return nn


def get_sample_weight2_TEST(
    lat_lon_TEST=None,
    mid_depth_TEST=None,
    las_dict=None,
    vertical_anisotropy=0.01,
    las_lat_lon=None,
):
    """
    sample weight based on horizontal and vertical distance between wells
    """
    assert (lat_lon_TEST, list)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    sample_weight1 = []
    sample_weight2 = []
    for k in sorted(las_dict.keys()):

        sample_weight1 = sample_weight1 + [
            1 / get_distance(lat_lon_TEST, las_lat_lon[k])
        ] * len(las_dict[k])

        vertical_distance = list(abs(np.array(las_dict[k].index) - mid_depth_TEST))
        sample_weight2 = sample_weight2 + vertical_distance

    sample_weight1 = np.array(sample_weight1).reshape(-1, 1)
    sample_weight1 = MinMaxScaler().fit_transform(sample_weight1)

    sample_weight2 = np.array(sample_weight2).reshape(-1, 1)
    sample_weight2 = MinMaxScaler().fit_transform(sample_weight2)

    sample_weight = np.sqrt(
        sample_weight1 ** 2 + (1 / vertical_anisotropy * sample_weight2) ** 2
    )

    return sample_weight


class CustomCrossValidation:
    def __init__(self, n_splits=5, las_index=None):
        assert las_index is not None, "please provide las_index"
        self.n_splits = n_splits
        self.las_index = las_index

    def split(self, X, y, groups=None):
        assert all([len(X) == len(y), len(X) == len(self.las_index)])
        self.X = X.copy()
        self.y = y.copy()

        self.X["las_index"] = self.las_index
        las_index_unique = np.unique(self.las_index)
        self.test_index_ = np.random.choice(
            las_index_unique, size=len(las_index_unique) // self.n_splits, replace=False
        )


def CV_weighted(model, X, y, weights=None, cv=10):
    """
    model : a sci-kit learn estimator
    X : a numpy array of shape (n_samples, n_features)
    y : numpy array of shape (n_samples,)
    weights : sample weights
    cv : TYPE, optional, The default is 10.
    metrics : TYPE, optional, The default is [mean_squared_error].
    Returns: scores
    """

    if weights is None:
        weights = np.ones(len(X))

    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        model_clone = clone(model)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]

        try:
            model_clone.fit(X_train, y_train, sample_weight=weights_train)
        except:
            # KNN, MLP does not accept sample_weight
            model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        score = mean_squared_error(
            y_test, y_pred, sample_weight=weights_test, squared=False
        )

        scores.append(score)

    return np.mean(scores)


#%% process las data


class process_las:
    def __init__(self):
        return None

    def _renumber_columns(self, c):
        c = list(c.values)
        assert isinstance(c, list)
        c = c.copy()
        for i in c:
            if c.count(i) > 1:
                ix = [j for j, val in enumerate(c) if val == i]
                for m, n in enumerate(ix):
                    c[n] = c[n] + "_" + str(m)
        return c

    def despike(self, df=None, cols=None, size=11):
        """
        df should be a dataframe, will despike all or selected columns
        """
        assert isinstance(df, pd.DataFrame)
        assert size % 2 == 1, "Median filter window size must be odd."

        if cols is not None:
            assert isinstance(cols, list), "cols should be a list columns names of df"
            assert all(
                [i in df.columns for i in cols]
            ), "cols should be a list columns names of df"
        else:
            cols = df.columns

        for col in cols:
            df[col] = median_filter(df[col].values, size=size, mode="nearest")

        return df

    def keep_valid_DTSM_only(self, df=None):
        """
        return a df with rows of valid DTSM data
        """

        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

        # make sure only one 'DTSM' exist
        if list(df.columns.values).count("DTSM") > 1:
            print("More than one 'DTSM' curve exist!")

        if "DTSM" not in df.columns:
            print("DTSM not present in this las file!")

        # drop all empty rows in 'DTSM' column if there is a 'DTSM' column
        # if 'DTSM' in df.columns, dropp all rows with nan in DTSM column
        df = df.dropna(subset=["DTSM"])

        # drop other columns with all na
        df = df.dropna(axis=1, how="all")

        return df

    def load_test_No_DTSM(self, df=None, alias_dict=None):
        """
        return a df with selected mnemonics/alias
        """

        assert alias_dict is not None, "alias_dict is None, assign alias_dict!"

        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

        # drop other columns with all na
        df = df.dropna(axis=1, how="all")

        return df

    def detect_outliers(self, df=None, contamination=0.01):
        assert all([i in df.columns for i in ["DTCO", "DTSM"]])

        outlier_detector = EllipticEnvelope(contamination=contamination)
        try:
            labels = outlier_detector.fit_predict(df[["DTCO", "DTSM"]])
        except:
            labels = outlier_detector.fit_predict(df)

        return labels

    def get_df_by_mnemonics(
        self,
        df=None,
        target_mnemonics=None,
        log_mnemonics=[],
        strict_input_output=True,
        outliers_contamination=None,
        alias_dict=None,
        drop_na=True,
        despike_=False,
    ):
        """
        useage: get a cleaned dataframe by given mnemonics,
        target_mnemonics: a list of legal mnemonics,
        strict_input_output: if true, only output a df only if all target mnemonics are found in las, output None otherwise
            if false, will output a df with all possible mnemonis found in las
        """
        assert alias_dict is not None, "alias_dict is None, assign alias_dict!"

        possible_new_mnemonics = ["DEPTH"]

        df = df.copy()
        # check required parameters
        if target_mnemonics is None:
            print(
                "Target mnemonics are required as 'get_df_by_mnemonics' function input!"
            )
            return None
        else:
            assert isinstance(target_mnemonics, list)
            assert len(target_mnemonics) >= 1
            assert all(
                [
                    i in alias_dict.values() or possible_new_mnemonics
                    for i in target_mnemonics
                ]
            ), f"Mnemonics should be in the list of {np.unique(list(alias_dict.values()))}"

        # find alias in df for target mnemonics
        alias = dict()
        for m in target_mnemonics:
            alias[m] = []
            for col in df.columns:
                n = get_mnemonic(col, alias_dict=alias_dict)
                if m == n:
                    alias[m].append(col)

        df_ = None
        df_cols = []

        # key: target mnenomics; value: alias in df, average the curves if more than 1 exist!
        for key, value in alias.items():
            if len(value) == 1:
                temp = df[value].values.reshape(-1, 1)
                df_cols.append(key)
            elif len(value) > 1:
                temp = df[value].mean(axis=1).values.reshape(-1, 1)
                df_cols.append(key)
                # print(f'{key} has more than 1 curves:{value}, averaged!')
            elif len(value) == 0:  # return index if no such column
                # print(f'\tNo corresponding alias for {key}!')
                continue

            if df_ is None:
                df_ = temp
            else:
                df_ = np.c_[df_, temp]

        if df_ is None:
            print("No data found for any requested mnemonics!")
            return None
        else:
            df_ = pd.DataFrame(df_, columns=df_cols)
            df_.index = df.index

        # # # shift NPHI
        # if "NPHI" in df_.columns:
        #     if any(df_["NPHI"] < 0):
        #         adj = df_[df_["NPHI"] < 0]["NPHI"].quantile(q=0.001)
        #         df_["NPHI"] = df_["NPHI"] + abs(adj)

        # remove data that's "abnormal", not for TEST data which does not have 'DTSM'
        if "DTSM" in target_mnemonics:
            mnemonic_range = {
                "DTSM": [60, 270],
                "DTCO": [40, 120],
                "GR": [-10, 250],
                "NPHI": [-0.06, 0.6],
                "PEFZ": [1, 11],
                "RHOB": [1.5, 3.2],
                "CALI": [4, 25],
            }
            for key, value in mnemonic_range.items():
                if key in df_.columns:
                    df_[key] = df_[key][(df_[key] >= value[0]) & (df_[key] <= value[1])]

            if all([i in df_.columns for i in ["DTCO", "DTSM"]]):
                df_ = df_[~((df_["DTCO"] > 95) & (df_["DTSM"] < 80))]

            if all([i in df_.columns for i in ["RHOB", "DTSM"]]):
                df_ = df_[~((df_["RHOB"] < 2.2) & (df_["DTSM"] < 80))]

        # take log of 'RT' etc.
        for col in log_mnemonics:
            if col in df_.columns:
                df_[col] = abs(df_[col]) + 1e-4
                df_[col] = np.log(df_[col])

        # add 'DEPTH' col if requested
        if ("DEPTH" in target_mnemonics) and ("DEPTH" not in df_.columns):
            df_["DEPTH"] = df_.index

        # drop na from the resulting dataframe, do not do it when it's TEST dataset
        if drop_na:
            df_ = df_.dropna(axis=0)  # subset=["DTSM"],

        # smooth/despike the logs
        if despike_:
            df_ = self.despike(df_)

        # remove 'outliers', where DTCO and DTSM does not correlate well
        # outliers == -1, inliners==1, if have problem, then do not remove as it's not as a big deal
        try:
            if outliers_contamination is not None and len(df_) > 1 and df_ is not None:
                df_ = df_[
                    self.detect_outliers(df=df_, contamination=outliers_contamination)
                    == 1
                ]
        except:
            pass

        if strict_input_output and all([i in df_.columns for i in target_mnemonics]):
            # print(f"\tAll target mnemonics are found in df, returned COMPLETE dataframe!")
            # rearrange mnemonics sequence, required
            return df_[target_mnemonics]

        elif not strict_input_output:
            return df_
        else:
            return None

    def get_compiled_df_from_las_dict(
        self,
        las_data_dict=None,
        target_mnemonics=None,
        log_mnemonics=[],
        strict_input_output=True,
        outliers_contamination=False,
        alias_dict=None,
        drop_na=True,
        return_dict=False,
    ):

        target_las_dict = dict()
        # get the data that corresponds to terget mnemonics
        for key in las_data_dict.keys():
            print(f"Loading {key}")
            df = las_data_dict[key]

            df = self.despike(df)

            df = self.get_df_by_mnemonics(
                df=df,
                target_mnemonics=target_mnemonics,
                log_mnemonics=log_mnemonics,
                strict_input_output=strict_input_output,
                outliers_contamination=outliers_contamination,
                alias_dict=alias_dict,
                drop_na=drop_na,
            )

            if (df is not None) and len(df > 1):
                target_las_dict[key] = df

        print(
            f"Total {len(target_las_dict.keys())} las files loaded and total {sum([len(i) for i in target_las_dict.values()])} rows of data!"
        )

        df_ = pd.concat([target_las_dict[k] for k in target_las_dict.keys()], axis=0)

        if return_dict:
            return target_las_dict
        else:
            return df_


class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        return None

    def fit(self, X_train=None, y_train=None):
        # the prediction will always be the mean of y
        # assert isinstance(X_train, pd.DataFrame) or isinstance(X_train, np.ndarray)
        assert isinstance(y_train, pd.DataFrame) or isinstance(y_train, np.ndarray)
        assert len(y_train) == len(X_train)
        assert y_train.shape[1] == 1

        try:
            self.y_bar_ = np.mean(y_train.values, axis=0)
        except:
            self.y_bar_ = np.mean(y_train, axis=0)
        self.y = y_train
        return self.y_bar_

    def predict(self, X_test):
        # give back the mean of y, in the same length as input X
        return np.ones(len(X_test)) * self.y_bar_


# #%% Test data
# las_name_test = "001-00a60e5cc262_TGS"
# las_test = "data/las/00a60e5cc262_TGS.las"
# df_test = read_las(las_test).df()
