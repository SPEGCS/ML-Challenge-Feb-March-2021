#%% load necessary data for main.py
import pickle
import pandas as pd
import numpy as np
from util import read_pkl

#%% get the alias_dict, required
try:
    alias_dict = read_pkl(f"data/alias_dict.pickle")
except:
    print(" No 'alias_dict.pickle' loaded as it is NOT available!")
    alias_dict = None

# load from training data
try:
    las_data_DTSM_QC = read_pkl(f"data/las_data_DTSM_QC.pickle")
except:
    print(" No 'las_data_DTSM_QC.pickle' loaded as it is NOT available!")
    las_data_DTSM_QC = None

# get las_lat_lon
try:
    las_lat_lon = read_pkl(f"data/las_lat_lon.pickle")
except:
    print(" No 'las_lat_lon.pickle' in data loaded as it is NOT available!")
    las_lat_lon = None

# get las_depth
try:
    las_depth = read_pkl(f"data/las_depth.pickle")
except:
    print(" No 'las_depth.pickle' loaded as it is NOT available!")
    las_depth = None

# ----------------------------------------------------------------------------------------
# get TEST data

try:
    las_data_TEST = read_pkl(f"data/leaderboard_final/las_data_TEST.pickle")
except:
    print(" No 'las_data_TEST.pickle' loaded as it is NOT available!")
    las_data_TEST = None

try:
    las_lat_lon_TEST = read_pkl(f"data/leaderboard_final/las_lat_lon_TEST.pickle")
except:
    print(
        " No 'las_lat_lon_TEST.pickle' in leaderboard_final loaded as it is NOT available!"
    )
    lat_lon_TEST = None

try:
    las_depth_TEST = read_pkl(f"data/leaderboard_final/las_depth_TEST.pickle")
except:
    print("No 'las_depth_TEST' loaded as it is NOT available!")
    las_depth_TEST = None

# ----------------------------------------------------------------------------------------

try:
    test_list = pd.read_csv("data/test_list.csv", index_col=False)

except:
    print("No test_list loaded!")
    test_list = None

try:
    KMeans_model_3clusters = read_pkl(
        f"models/KMeans_model_[DTCO,NPHI,GR]_3_clusters.pickle"
    )

    KMeans_model_5clusters = read_pkl(
        f"models/KMeans_model_[DTCO,NPHI,GR]_5_clusters.pickle"
    )
    KMeans_model_RHOB_2clusters = read_pkl(
        f"models/KMeans_model_RHOB_std_2_clusters.pickle"
    )

except:
    print("No KMeans_model loaded!")


try:
    neighbors_TEST = read_pkl("data/leaderboard_final/neighbors_TEST.pickle")

    # TEST_neighbors_7features = []
    TEST_neighbors_20 = []
    # TEST_neighbors = []

    for val in neighbors_TEST.values():

        val_bool = [i[0] for i in val if i in test_list["Test LAS"].tolist()]

        # get 1 well for TEEST well, make sure it has 7 features
        if len(val_bool) >= 1:
            if val_bool[0] not in TEST_neighbors_20:
                TEST_neighbors_20.append(val_bool[0])

        # for i in val:

        #     if i not in TEST_neighbors:
        #         TEST_neighbors.append(i[0])
        #     if i not in TEST_neighbors_7features and i in test_list.values:
        #         TEST_neighbors_7features.append(i[0])
    TEST_neighbors_20 = sorted(TEST_neighbors_20)
    TEST_neighbors_20.remove("048-3ca90f59eddb_TGS")
except:
    print("No neighbors_TEST loaded!")
    neighbors_TEST = None

try:
    las_data_DTSM_felix = read_pkl("data/felix_data/las_data_DTSM_felix.pickle")
except:
    print("No las_data_DTSM_felix loaded!")

feature_model = {
    "7": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    "6_1": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "DTSM"],
    "6_2": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    "6_3": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "PEFZ", "DTSM"],
    # "5_0": ["DEPTH", "RHOB", "NPHI", "GR", "RT", "CALI", "DTSM"],
    "5_1": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "PEFZ", "DTSM"],
    "5_2": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "DTSM"],
    # "5_3": ["DEPTH", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    "5_4": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "CALI", "DTSM"],
    # "4_1": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "DTSM"],
    "4_2": ["DEPTH", "DTCO", "GR", "RT", "CALI", "DTSM"],
    # "4_3": ["DEPTH", "DTCO", "RHOB", "NPHI", "PEFZ", "DTSM"],
    "4_4": ["DEPTH", "DTCO", "NPHI", "GR", "RT", "DTSM"],
    # "3_1": ["DEPTH", "DTCO", "NPHI", "GR", "DTSM"],
    "3_2": ["DEPTH", "DTCO", "GR", "RT", "DTSM"],
    # "3_3": ["DEPTH", "DTCO", "RHOB", "NPHI", "DTSM"],
    "3_4": ["DEPTH", "DTCO", "NPHI", "GR", "DTSM"],
    # "3_5": ["DTCO", "NPHI", "GR", "DTSM"],
    # "2_1": ["DEPTH", "DTCO", "NPHI", "DTSM"],
    # "2_2": ["DEPTH", "RHOB", "NPHI", "DTSM"],
    "2_3": ["DEPTH", "DTCO", "GR", "DTSM"],
    # "1_1": [ "DTCO", "DTSM"],
    "1_2": ["DEPTH", "DTCO", "DTSM"],  # for clustering study
    "0_4_DTCO": ["DEPTH", "RHOB", "NPHI", "GR", "RT", "DTCO"],
    # "0_4_PEFZ": ["DEPTH", "RHOB", "NPHI", "GR", "RT", "PEFZ"],
    # "0_6_PEFZ": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "CALI", "RT", "PEFZ"],
    # "0_4_RHOB": ["DEPTH", "DTCO", "NPHI", "GR", "RT", "RHOB"],
}

# model_xgb_rock_clas_7 = read_pkl("models/model_xgb_rock_clas_7.pickle")
# model_xgb_rock_class_6_2 = read_pkl("models/model_xgb_rock_class_6_2.pickle")

las_data_DTSM_QC2 = read_pkl("data/las_data_DTSM_QC2.pickle")
rock_info = read_pkl("data/rock_info.pickle")

las_data_TEST_renamed = read_pkl(
    "data/leaderboard_final/final_data_plots/las_data_TEST_renamed.pickle"
)
