#%% this file load the each las data to dataframe and save them
# the resulting files are in /data/ folder.

import glob
import pickle
import time
import numpy as np
import pandas as pd
import re
import plotly.express as px
from plot import plot_logs_columns, plot_wells_3D
from util import (
    read_las,
    process_las,
    get_mnemonic,
    to_pkl,
    read_pkl,
    get_nearest_neighbors,
    get_nearest_neighbors_2,
)
from load_pickle import alias_dict, las_depth, las_lat_lon

path = f"data/leaderboard_final"

#%% read all las files to df, keep all and valid DTSM only, and store to pickle file format


las_data = dict()  # raw df in las files
las_lat_lon_TEST = dict()  # lat and lon of test wells
las_depth_TEST = dict()  # STRT and STOP depth of test wells
las_list_TEST = []  # name of the test files

time_start = time.time()
count_ = 1
for f in glob.glob(f"{path}/*.las"):

    # get the file name w/o extension
    WellName = re.split("[\\\/.]", f)[-2]
    WellName = f"T{count_:0>2}-{WellName}"
    print(f"Loading {count_:0>3}th las file: \t{WellName}.las")
    count_ += 1

    # save the WellName for future use
    las_list_TEST.append(WellName)

    # get raw df and save to a dict
    las = read_las(f)
    df = las.df()
    las_data[WellName] = df.copy()

    # get lat-lon and save to a dict
    las_lat_lon_TEST[WellName] = las.get_lat_lon()

    # get depth and save to a dict
    las_depth_TEST[WellName] = [df.index.min(), df.index.max()]

    # plot the raw logs for visual inspection
    plot_logs_columns(
        df,
        well_name=f"{WellName}-raw-data",
        plot_show=False,
        plot_save_file_name=WellName,
        plot_save_path=path,
        plot_save_format=["png"],
        alias_dict=alias_dict,
    )

# write file names
las_list_TEST = pd.DataFrame(las_list_TEST, columns=["WellName"])
las_list_TEST.to_csv(f"{path}/las_list_TEST.csv")


# special case for Well_02
df = las_data["T02-0a7822c59487_TGS"]
df["NPHI"] = pd.concat(
    [df["SPHI_LS"][df.index > 7900], df["NPOR_LS"][df.index <= 7900]]
)
df["GRD"] = pd.concat([df["GRS"][df.index > 7900], df["GRD"][df.index <= 7900]])

# special case for Well_04
df = las_data["T04-1684cc35f399_TGS"]
df["CALI"] = pd.concat([df["CALR"][df.index < 11000], df["CALD"][df.index >= 11000]])

# write las_data
to_pkl(las_data, f"{path}/las_data.pickle")

# write las_lat_lon_TEST
to_pkl(las_lat_lon_TEST, f"{path}/las_lat_lon_TEST.pickle")

# write las_depth_TEST
to_pkl(las_depth_TEST, f"{path}/las_depth_TEST.pickle")

print(f"\nSuccessfully loaded total {len(las_data)} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")

#%% pick the curves from test data, requires log_QC_input.csv

log_QC_input = pd.read_csv(f"{path}/TEST_log_QC_input.csv")

# read las_data
las_data = read_pkl(f"{path}/las_data.pickle")
las_data_TEST = dict()  # cleaned df from las_data

# remove the undesired curves
temp = log_QC_input[["WellName", "Curves to remove"]]
for ix, WellName, curves_to_remove in temp.itertuples():

    # if ix == 1:
    # print("curves_to_remove:", curves_to_remove)

    curves_to_remove = [i.strip() for i in str(curves_to_remove).split(",")]
    print(WellName, "removing", curves_to_remove[:2], "...")

    # check if all 'curves_to_remove' are in columns names, if YES then drop these curves
    if all([i in las_data[WellName].columns for i in curves_to_remove]):
        las_data_TEST[WellName] = las_data[WellName][
            las_data[WellName].columns.difference(curves_to_remove)
        ]

        # remaining_mnemonics = [
        #     get_mnemonic(i, alias_dict=alias_dict)
        #     for i in las_data_TEST[WellName].columns
        # ]

        # # make sure not to remove curves by accident
        # for i in curves_to_remove:
        #     mapped_mnemonic = get_mnemonic(i, alias_dict=alias_dict)
        #     if mapped_mnemonic not in remaining_mnemonics:
        #         print(
        #             f"\tWarning: removing {i} from data, while {remaining_mnemonics} does not have  {mapped_mnemonic}!"
        #         )

    # if not all curves in columns, then no curves removed! Recheck!
    else:
        las_data_TEST[WellName] = las_data[WellName]
        if curves_to_remove != ["nan"]:
            print(
                f"\tNot all {curves_to_remove} are in {WellName} columns. No curves are removed!"
            )

las_data["T02-0a7822c59487_TGS"]
las_data_TEST["T02-0a7822c59487_TGS"]
las_data_TEST.keys()

# write las_data
to_pkl(las_data_TEST, f"{path}/las_data_TEST.pickle")

print("*" * 90)
print("Congratulations! Loaded data successfully!")
#%% plot the QC'd logs for 20 TEST wells
for key in las_data_TEST.keys():
    plot_logs_columns(
        df=las_data_TEST[key],
        well_name=f"{key}-test-data",
        plot_show=False,
        plot_save_file_name=f"{key}-test-data",
        plot_save_path=path,
        plot_save_format=["png", "html"],
        alias_dict=alias_dict,
    )


#%% plot relative locations in 3D

las_depth_TEST = read_pkl(f"{path}/las_depth_TEST.pickle")
las_lat_lon_TEST = read_pkl(f"{path}/las_lat_lon_TEST.pickle")

for key in las_depth_TEST.keys():
    plot_wells_3D(
        las_name_test=key,
        las_depth=dict(**las_depth, **las_depth_TEST),
        las_lat_lon=dict(**las_lat_lon, **las_lat_lon_TEST),
        num_of_neighbors=1,
        vertical_anisotropy=0.1,
        depth_range_weight=0,
        title=key,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{key}-3D-location",
        plot_save_path=f"{path}/3D_location",
        plot_save_format=["png"],
    )

#%%  #%% plot relative locations with N neighbors in 3D
for key in las_depth_TEST.keys():
    plot_wells_3D(
        las_name_test=key,
        las_depth=dict(**las_depth, **las_depth_TEST),
        las_lat_lon=dict(**las_lat_lon, **las_lat_lon_TEST),
        num_of_neighbors=10,
        vertical_anisotropy=0.1,
        depth_range_weight=0,
        title=key,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"{key}-3D-location-5neighbors_va=0.01",
        plot_save_path=f"{path}/3D_location",
        plot_save_format=["png", "html"],
    )

#%% get N nearest neighbors for 20 test wells

neighbors_TEST = dict()

for key in las_lat_lon_TEST.keys():
    las_depth_ = dict(**las_depth, **dict(key=las_depth_TEST[key]))
    las_lat_lon_ = dict(**las_lat_lon, **dict(key=las_lat_lon_TEST[key]))
    neighbors = get_nearest_neighbors(
        depth_TEST=las_depth_TEST[key],
        lat_lon_TEST=las_lat_lon_TEST[key],
        las_depth=las_depth_,
        las_lat_lon=las_lat_lon_,
        num_of_neighbors=10,
        vertical_anisotropy=0.1,
        depth_range_weight=0,
    )
    neighbors_TEST[key] = neighbors

    print(key, neighbors, "\n")
to_pkl(neighbors_TEST, f"{path}/neighbors_TEST.pickle")


key = "T20-ff7845ea074d_TGS"
get_nearest_neighbors(
    depth_TEST=las_depth_TEST[key],
    lat_lon_TEST=las_lat_lon_TEST[key],
    las_depth=las_depth_,
    las_lat_lon=las_lat_lon_,
    num_of_neighbors=10,
    vertical_anisotropy=0.1,
    depth_range_weight=0,
)

#%% plot the QC'd and renamed logs for 20 TEST wells
from load_pickle import feature_model, las_data_DTSM_QC2

path = "data/leaderboard_final/final_data_plots"
las_data_TEST = read_pkl(f"{path}/las_data_TEST.pickle")

model_name = "7"
target_mnemonics = feature_model[model_name].copy()
target_mnemonics.remove("DTSM")
las_data_TEST_renamed = dict()

for key, df in las_data_TEST.items():
    print(key)

    # if key in ["109-84967b1f42e0_TGS", "204-d6aa464fab0e_TGS"]:
    # plot QC'd logs
    plot_logs_columns(
        df=df,
        well_name=f"{key}-selected-data",
        plot_show=False,
        plot_save_file_name=f"{key}-selected-data",
        plot_save_path=path,
        plot_save_format=["png", "html"],
        alias_dict=alias_dict,
    )

    df = process_las().get_df_by_mnemonics(
        df=df,
        target_mnemonics=target_mnemonics,
        log_mnemonics=["RT"],
        strict_input_output=False,
        alias_dict=alias_dict,
        drop_na=False,
    )
    print(df.sample(5))

    for i in target_mnemonics:
        if i not in df.columns:
            df[i] = np.nan
            print(f"fill target mnemonics: {i}")

    df = df[target_mnemonics]
    print(df.sample(5))

    las_data_TEST_renamed[key] = df
    to_pkl(las_data_TEST_renamed, f"{path}/las_data_TEST_renamed.pickle")

    # plot renamed logs
    plot_logs_columns(
        df=df,
        well_name=f"{key}-renamed-data",
        plot_show=False,
        plot_save_file_name=f"{key}-renamed-data",
        plot_save_path=path,
        plot_save_format=["png", "html"],
        alias_dict=alias_dict,
    )

    fig = px.histogram(df["DTCO"])
    fig.write_image(f"{path}/{key}_DTCO_histogram.png")

print("Completed plotting!")
