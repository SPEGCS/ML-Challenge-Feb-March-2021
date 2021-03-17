#%% plot module, import lib
import os
import pickle
import random
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score

from util import (
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,
    get_distance_weight,
    get_nearest_neighbors,
)

from load_pickle import las_data_DTSM_QC, las_lat_lon, alias_dict

pio.renderers.default = "browser"

#%% plot feature importance


def plot_feature_important(
    best_estimator=None, features=None, plot_save_name=None, path=None
):
    xgb_feature_importance = [round(i, 3) for i in best_estimator.feature_importances_]
    # print(f"Feature importance:\n{features} \n{xgb_feature_importance}")

    xgb_feature_importance_df = pd.DataFrame(
        np.c_[
            features[: len(xgb_feature_importance)],
            np.array(xgb_feature_importance).reshape(-1, 1),
        ],
        columns=["feature", "importance"],
    )

    # plot feature_importance bar
    fig = px.bar(
        xgb_feature_importance_df,
        x="feature",
        y="importance",
        width=1600,
        height=900,
    )
    if path is not None:
        if plot_save_name is None:
            fig.write_image(f"{path}/xgb_feature_importance.png")
        else:
            fig.write_image(f"{path}/xgb_feature_importance_{plot_save_name}.png")


#%% 3D plot of wels


def plot_wells_3D(
    las_name_test=None,
    las_depth=None,
    las_lat_lon=None,
    num_of_neighbors=5,
    vertical_anisotropy=0.1,
    depth_range_weight=0.1,
    title=None,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,
):

    assert isinstance(las_name_test, str)
    assert isinstance(las_depth, dict)
    assert all([las_name_test in las_depth.keys()])

    if title is None:
        title = f"Wellbore Visualization | Test well shown as the center well | {num_of_neighbors} neighbor wells shown"
    else:
        title = (
            f"Wellbore Visualization {title} | {num_of_neighbors} neighbor wells shown"
        )

    fig = go.Figure()

    neighbors = get_nearest_neighbors(
        depth_TEST=las_depth[las_name_test],
        lat_lon_TEST=las_lat_lon[las_name_test],
        las_depth=las_depth,
        las_lat_lon=las_lat_lon,
        num_of_neighbors=num_of_neighbors,
        vertical_anisotropy=vertical_anisotropy,
        depth_range_weight=depth_range_weight,
    )

    # add line connections from all wells to test well
    connect_dict = dict()
    connect_dict[las_name_test] = pd.DataFrame(
        [np.mean(las_depth[las_name_test])], columns=["Depth"]
    )
    connect_dict[las_name_test][["Lat", "Lon"]] = las_lat_lon[las_name_test]
    connect_dict[las_name_test]["Las_Name"] = las_name_test

    # create data for each wellbore and plot it
    depth_dict = dict()
    for key, val in las_depth.items():

        depth_dict[key] = pd.DataFrame(val, columns=["Depth"])
        depth_dict[key][["Lat", "Lon"]] = las_lat_lon[key]
        depth_dict[key]["Las_Name"] = key

        # create connection line data
        if key != las_name_test:
            connect_dict[key] = pd.concat(
                [
                    connect_dict[las_name_test],
                    depth_dict[key],
                    connect_dict[las_name_test],
                ],
                axis=0,
            )

        # add enighbor or non-neighbor wells
        if key in neighbors:
            fig.add_traces(
                go.Scatter3d(
                    x=depth_dict[key]["Lat"],
                    y=depth_dict[key]["Lon"],
                    z=depth_dict[key]["Depth"],
                    showlegend=False,
                    name=key,
                    mode="lines",
                    line=dict(width=15),
                    # hoverinfo='skip',
                    hovertemplate="<br><b>Depth<b>: %{z:.0f}",
                )
            )
        else:
            fig.add_traces(
                go.Scatter3d(
                    x=depth_dict[key]["Lat"],
                    y=depth_dict[key]["Lon"],
                    z=depth_dict[key]["Depth"],
                    showlegend=False,
                    name=key,
                    mode="lines",
                    line=dict(width=1),
                    # hoverinfo='skip',
                    hovertemplate="<br><b>Depth<b>: %{z:.0f}",
                )
            )

        # add connecting lines
        fig.add_traces(
            go.Scatter3d(
                x=connect_dict[key]["Lat"],
                y=connect_dict[key]["Lon"],
                z=connect_dict[key]["Depth"],
                showlegend=False,
                mode="lines",
                line=dict(width=0.1),
                hoverinfo="skip",
            )
        )

    # emphasize center TEST wells
    fig.add_traces(
        go.Scatter3d(
            x=depth_dict[las_name_test]["Lat"],
            y=depth_dict[las_name_test]["Lon"],
            z=depth_dict[las_name_test]["Depth"],
            showlegend=False,
            name=key,
            mode="lines",
            line=dict(width=30),
            # hoverinfo='skip',
            hovertemplate="<br><b>Depth<b>: %{z:.0f}",
        )
    )

    fig.update_layout(
        scene_camera=dict(eye=dict(x=2, y=0, z=0.0)),
        template="plotly_dark",
        width=2500,
        height=1200,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=dict(
            text=title, x=0.5, xanchor="center", font=dict(color="Lime", size=20)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.06,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_scenes(
        xaxis=dict(
            title="",
            showgrid=False,
            showline=False,
            showbackground=False,
            showticklabels=False,
            # range=[ ]
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            showline=False,
            showbackground=False,
            showticklabels=False,
            # range = [ ]
        ),
        zaxis=dict(
            title="",
            showgrid=False,
            showline=False,
            showbackground=False,
            showticklabels=False,
            range=(25000, 0),
        ),
    ),

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"wellbore_3D_{num_of_neighbors}_neighbors"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)
            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            # print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            pass
            # print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


#%% simple plot of all curves in one column, good for quick plotting


def plot_logs(
    df,
    well_name="well",
    DTSM_only=True,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"]
):

    """
    simple plot of all curves in one column, good for quick plotting
    """

    df = df.copy()

    # drop part of logs where no DTSM data
    if DTSM_only:
        df.dropna(subset=["DTSM"], inplace=True)

    fig = go.Figure()

    for col in df.columns:

        # do not plots in the below list:
        if col not in ["DEPT", "TEND", "TENR"]:
            fig.add_trace(go.Scatter(x=df[col], y=df.index, name=col))

    fig.update_layout(
        showlegend=True,
        title=dict(text=well_name),
        yaxis=dict(autorange="reversed", title="Depth, ft"),
        font=dict(size=18),
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)
            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            # print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            pass
            # print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


#%% complex plot that plots curves in multiple columns


def plot_logs_columns(
    df,
    DTSM_pred=None,
    well_name="",
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"]
    alias_dict=None,
):
    """
    complex plot that plots curves in multiple columns, good for detailed analysis of curves
    """

    df = df.copy()

    # determine how many columns for grouped curves
    columns = df.columns.map(alias_dict)
    tot_cols = [
        ["DTCO", "DTSM"],  #  row=1, col=1
        ["RHOB"],  #  row=1, col=2
        # ['DPHI'],                           #  row=1, col=3
        ["NPHI"],  #  row=1, col=4
        ["GR"],  #  row=1, col=5
        ["RT"],  #  row=1, col=6
        ["CALI"],  #  row=1, col=7
        ["PEFZ"],
    ]  #  row=1, col=8

    num_of_cols = 1
    tot_cols_new = []  # update the tot_cols if some curves are missing
    tot_cols_old = []

    for cols in tot_cols:
        if any([(i in columns) for i in cols]):
            tot_cols_new.append(cols)
            num_of_cols += 1

        # get the old columns as subplot titles
        temp = []
        for i in df.columns:

            if get_mnemonic(i, alias_dict=alias_dict) in cols:
                temp.append(i)
        if len(temp) > 0:
            tot_cols_old.append(temp)

    # make subplots (flexible with input)
    fig = make_subplots(
        rows=1,
        cols=num_of_cols,
        subplot_titles=[",".join(j) for j in tot_cols_old],
        shared_yaxes=True,
    )

    for col_old in df.columns:

        # find the mnemonic for alias
        col_new = get_mnemonic(col_old, alias_dict=alias_dict)
        try:
            # find the index for which column to plot the curve
            col_id = [i + 1 for i, v in enumerate(tot_cols_new) if col_new in v][0]
        except:
            col_id = num_of_cols

        # print(f'col_old: {col_old}, col_new: {col_new}, col_id: {col_id}')
        # if 'TENS' not in col_new:
        if col_id != num_of_cols:
            fig.add_trace(
                go.Scatter(x=df[col_old], y=df.index, name=col_old), row=1, col=col_id
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[col_old],
                    y=df.index,
                    name=col_old,
                    mode="markers",
                    marker=dict(size=1),
                ),
                row=1,
                col=col_id,
            )

    # add predicted DTSM if not None
    if DTSM_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=DTSM_pred["DTSM_Pred"],
                y=DTSM_pred["Depth"],
                mode="lines",  # +markers
                line_color="rgba(255, 0, 0, .7)",
                name="DTSM_Pred",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        showlegend=True,
        title=dict(text=well_name, font=dict(size=12)),
        yaxis=dict(autorange="reversed", title="Depth, ft"),
        font=dict(size=18),
        legend=dict(
            orientation="h",
            y=1.07,
            yanchor="middle",
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        template="plotly",
        width=3000,
        height=1200,
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:

            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            # print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            pass
            # print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


#%% plot predicted and actual in a crossplot
def plot_crossplot(
    y_actual=None,
    y_predict=None,
    text=None,
    axis_range=300,
    include_diagnal_line=False,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"])
):

    assert len(y_actual) == len(y_predict)
    rmse_test = mean_squared_error(y_actual, y_predict, squared=False)
    r2_test = r2_score(y_actual, y_predict)

    y_pred_act = pd.DataFrame(
        np.c_[y_actual.reshape(-1, 1), y_predict.reshape(-1, 1)],
        columns=["Actual", "Predict"],
    )
    abline = pd.DataFrame(
        np.c_[
            np.arange(axis_range).reshape(-1, 1), np.arange(axis_range).reshape(-1, 1)
        ],
        columns=["Actual", "Predict"],
    )

    if text is not None:
        title_text = f"{text}, rmse_test:{rmse_test:.2f}, r2_score: {r2_test:.2f}"
    else:
        title_text = f"rmse_test:{rmse_test:.2f}, r2_score: {r2_test:.2f}"

    fig = px.scatter(y_pred_act, x="Actual", y="Predict")

    if include_diagnal_line:
        fig.add_traces(px.line(abline, x="Actual", y="Predict").data[0])
        fig.update_layout(xaxis_range=[0, axis_range], yaxis_range=[0, axis_range])

    fig.update_layout(
        title=dict(text=title_text),
        width=1200,
        height=1200,
        # xaxis_range=[0,axis_range],
        # yaxis_range=[0,axis_range]
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            # print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            pass
            # print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


def plot_cords(
    cords=None,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"]
):

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(x=cords["Lon"], y=cords["Lat"], mode="markers"),
        hoverinfo="text",
        hovertext=cords["Well"],
    )
    fig.update_layout(
        xaxis=dict(title="Longitude"),
        yaxis=dict(title="Latitude"),
        # title = dict(text='Size: Stop Depth'),
        font=dict(size=18),
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            # print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            pass
            # print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


def plot_outliers(
    Xy=None,
    Xy_out=None,
    abline=None,
    text=None,
    axis_range=300,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,
):

    if text is not None:
        title_text = f"{text}"
    else:
        title_text = "Outliers"

    if Xy is not None:
        fig = px.scatter(Xy, x="DTCO", y="DTSM")
    else:
        fig = go.Figure()

    if Xy_out is not None:
        fig.add_traces(
            px.scatter(
                Xy_out, x="DTCO", y="DTSM", color_discrete_sequence=["red"]
            ).data[0],
        )

    if abline is not None:
        fig.add_traces(px.line(abline, x="DTCO", y="DTSM").data[0])

    fig.update_layout(
        title=dict(text=title_text),
        width=1200,
        height=1200,
        xaxis_range=[40, 120],  # "DTCO": [40, 120],
        yaxis_range=[50, 250],  # "DTSM": [60, 270],
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            # print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            pass
            # print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig