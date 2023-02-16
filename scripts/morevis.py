from utils import create_connections, plot_summary, colormap2D
from projections import projection_selector
from optimization import optim_handler
from metrics import compute_intersection_metric, stress_measure_shapes, crossings
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_height_constant(df, method="area_max"):
    """
    Compute the height constant for the height of the objects.
    The constant is the ratio between bouding box of 1D objects and  bouding box of 2D objects.

    Inputs:
        df: DataFrame with columns ["y", "points"]
        method: name of the method to be used

    Outputs:
        area_scaler: Function that transform area

    """

    if method == "area_max":
        old_range = df.groupby("timestep").agg({"area": "sum"}).area.max()
        area_scaler = lambda x: x / old_range
    elif method.find("area_max") != -1:
        scaling = method.split("_")[-1]
        old_range = df.groupby("timestep").agg({"area": "sum"}).area.max()
        area_scaler = lambda x: x / old_range * float(scaling)
    else:
        raise ValueError("Method not recognized")

    return area_scaler


def update_connections(df):
    """
    Check if there is spurious crossings and mark them if there is.

    Inputs:
        df - DataFrame with result of MoReVis

    Outputs:
        df - Updated DataFrame
    """

    objects_list = df.object.unique()
    objects_list.sort()
    problematic_pairs = {"timestep": [], "object_i": [], "object_j": []}

    # compute the spurious intersections between links
    # that only occur when two objects change order
    for i, object_i in enumerate(objects_list):
        for j, object_j in enumerate(objects_list):
            if j <= i:
                continue

            df_object_i = df[df.object == object_i]
            df_object_j = df[df.object == object_j]

            timesteps_intersection = np.intersect1d(
                df_object_i.timestep.values,
                df_object_j.timestep.values,
                assume_unique=True,
            )

            df_object_i = df_object_i[
                df_object_i.timestep.isin(timesteps_intersection)
            ].sort_values("timestep")
            df_object_j = df_object_j[
                df_object_j.timestep.isin(timesteps_intersection)
            ].sort_values("timestep")

            # compute intersections for all timesteps
            I = []
            W = []
            for it, t in enumerate(timesteps_intersection):
                shape_i = Polygon(df_object_i["shape"].values[it])
                shape_j = Polygon(df_object_j["shape"].values[it])
                I.append(shape_i.intersection(shape_j).area)
                region_i = Polygon(df_object_i["points"].values[it]).convex_hull
                region_j = Polygon(df_object_j["points"].values[it]).convex_hull
                W.append(region_i.intersection(region_j).area)

            for it, t in enumerate(timesteps_intersection):
                # if is a timestep of a rect, skip it
                if df_object_i["shape_type"].values[it] == "rect":
                    continue

                shape_i = df_object_i["shape"].values[it]
                shape_j = df_object_j["shape"].values[it]

                y_left_i = (shape_i[0][1] + shape_i[3][1]) / 2
                y_left_j = (shape_j[0][1] + shape_j[3][1]) / 2
                y_right_i = (shape_i[1][1] + shape_i[2][1]) / 2
                y_right_j = (shape_j[1][1] + shape_j[2][1]) / 2

                # check if two lines don't crosses
                shapes_cross = (y_right_i - y_right_j) * (y_left_i - y_left_j) < 0
                objects_intersection = W[it + 1] > 0 or W[it - 1] > 0
                spurios_intersection = shapes_cross and ~objects_intersection

                # if after these checks, it still a spurious intersection, update the shape
                if spurios_intersection:
                    problematic_pairs["timestep"].append(t)
                    problematic_pairs["object_i"].append(object_i)
                    problematic_pairs["object_j"].append(object_j)

    problematic_pairs = pd.DataFrame(problematic_pairs)
    # process by timestep
    timesteps = problematic_pairs.timestep.unique()

    for t in timesteps:
        problematic_pairs_t = problematic_pairs[problematic_pairs.timestep == t]

        for i, row in problematic_pairs_t.iterrows():
            index = np.where(
                (df.timestep == t)
                & ((df.object == row.object_i) | (df.object == row.object_j))
            )[0]
            if len(index) == 2:
                df.at[index[0], "style"] = "dashed"
                df.at[index[1], "style"] = "dashed"
            else:
                df.at[index, "style"] = "dashed"

    return df


def compute_intersections_info(df):
    """
    Create a DataFrame with information of intersections for each timestep.

    Inputs:
        df - DataFrame with columns ["timestep", "y", "height", "points"]

    Outputs:
        DataFrame with metric detailed values

    """

    results = {
        "timestep": [],
        "object1": [],
        "object2": [],
        "y_bottom": [],
        "y_center": [],
        "y_top": [],
        "area_1d": [],
        "area_2d": [],
        "intersection_1d": [],
        "intersection_2d": [],
        "spurious_intersection": [],
        "shape_type": [],
    }

    objects_list = df.object.unique()
    objects_list.sort()

    # first, compute spurious intersections between rects
    df_rect = df[df["shape_type"] == "rect"]
    for i, o1 in enumerate(objects_list):
        for j, o2 in enumerate(objects_list):
            if j <= i:
                continue

            df_o1 = df_rect[df_rect.object == o1]
            df_o2 = df_rect[df_rect.object == o2]
            df_o1_timesteps = df_o1.timestep.values
            df_o2_timesteps = df_o2.timestep.values
            timesteps_intersection = np.intersect1d(
                df_o1_timesteps, df_o2_timesteps, assume_unique=True
            )

            df_o1 = df_o1[df_o1.timestep.isin(timesteps_intersection)].sort_values(
                "timestep"
            )
            df_o2 = df_o2[df_o2.timestep.isin(timesteps_intersection)].sort_values(
                "timestep"
            )

            for it, t in enumerate(timesteps_intersection):
                shape_i = Polygon(df_o1["shape"].values[it])
                shape_j = Polygon(df_o2["shape"].values[it])
                I = shape_i.intersection(shape_j).area

                region_i = Polygon(df_o1["points"].values[it]).convex_hull
                region_j = Polygon(df_o2["points"].values[it]).convex_hull
                W = region_i.intersection(region_j).area

                # if there is no intersection both in 1D and 2D, continue to next iteration
                if I == 0 and W == 0:
                    continue

                results["timestep"].append(t)
                results["object1"].append(o1)
                results["object2"].append(o2)
                y_bottom = min(
                    df_o1.y.values[it] - df_o1.height.values[it] / 2,
                    df_o2.y.values[it] - df_o2.height.values[it] / 2,
                )
                y_top = max(
                    df_o1.y.values[it] + df_o1.height.values[it] / 2,
                    df_o2.y.values[it] + df_o2.height.values[it] / 2,
                )
                results["y_bottom"].append(y_bottom)
                results["y_top"].append(y_top)
                results["y_center"].append(0)
                results["area_1d"].append(I)
                results["area_2d"].append(W)
                results["intersection_1d"].append(I > 0)
                results["intersection_2d"].append(W > 0)
                results["spurious_intersection"].append(I > 0 and W == 0)
                results["shape_type"].append("rect")

    results = pd.DataFrame(results).sort_values("timestep")
    return results


def MoReVis(
    input_data,
    projection_name="pca",
    projection_params={},
    height_method="area_max",
    optimization_params={},
    color_column_name="object",
    colormap_name="Paired",
    return_area_scaler=False,
    plot_metrics=False,
    plot=True,
    ax=None,
):

    data = input_data.copy()
    area_scaler = compute_height_constant(data, height_method)
    data["height"] = data["area"].apply(area_scaler)
    data = data.sort_values(["timestep", "object"])
    data = projection_selector(data, projection_name, projection_params)

    # morevis optimization
    data = optim_handler(data, area_scaler=area_scaler, **optimization_params)
    # data["y"] = data["proj"]
    colormap = plt.get_cmap(colormap_name)
    # if color_column_name is dtype float
    if data[color_column_name].dtype == "float":
        color_min = data[color_column_name].min()
        color_max = data[color_column_name].max()
        data["color"] = [
            colormap((c - color_min) / (color_max - color_min))
            for c in data[color_column_name]
        ]
    else:
        color_index = data[color_column_name].unique()
        color_index = np.sort(color_index)
        color_index = dict([(c, i) for i, c in enumerate(color_index)])
        data["color"] = [
            colormap(color_index[c] % len(colormap.colors))
            for c in data[color_column_name]
        ]

    for colormap in ["bremm", "schumann_urban", "steiger", "teuling", "ziegler"]:
        data["color_" + colormap] = colormap2D(
            data[["xcenter", "ycenter"]].to_numpy(copy=True), colormap
        )

    data = create_connections(data)
    data = update_connections(data)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        plot_summary(data, ax=ax)

    if plot_metrics:
        _, spurious_intersections, _, _, _ = compute_intersection_metric(data)
        crossings_mean, _, _ = crossings(data[data.shape_type == "rect"])
        stress_measure = stress_measure_shapes(data[data.shape_type == "rect"])
        ax.annotate(
            f"Spurious: {int(100*spurious_intersections)}%",
            xy=(0.02, -0.1),
            xycoords="axes fraction",
            fontsize=12,
        )
        # ax.annotate(f"Crossings: {crossings_mean:.2f}", xy = (0.01, 0.8), xycoords = "axes fraction", fontsize = 10)
        ax.annotate(
            f"Stress: {stress_measure:.2f}",
            xy=(0.2, -0.1),
            xycoords="axes fraction",
            fontsize=12,
        )

    if return_area_scaler:
        return data, area_scaler
    return data
