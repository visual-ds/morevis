import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def stress_measure_shapes(df):
    """
    Compute the stress measure (MDS loss function) between the distances of polygons in the 2D space
    (smallest segment linking two polygons) and the distances of rects in the 1D space.

    Inputs:
        df - DataFrame with columns ["points", "y", "height"]

    Outputs:
        stress1 - metric error value
    """

    n = df.shape[0]
    polys = df.points.apply(lambda x: Polygon(x).convex_hull)
    old_d = np.zeros((n, n))
    new_d = np.zeros((n, n))

    y = df.y.values
    height = df.height.values

    for i in range(n):
        for j in range(i + 1, n):
            old_d[i, j] = polys[i].distance(polys[j])
            old_d[j, i] = old_d[i, j]

            new_d[i, j] = max(abs(y[i] - y[j]) - height[i] / 2 - height[j] / 2, 0)
            new_d[j, i] = new_d[i, j]

    old_d /= old_d.max()
    new_d /= new_d.max()

    stress = np.power(new_d - old_d, 2).sum()
    stress1 = np.sqrt(stress / (np.power(old_d, 2).sum()))
    return stress1


def crossings(df):
    """
    For each subsequents timesteps, count the number of crossings between two curves.

    Inputs:
        df - DataFrame with columns ["timestep", "object", "y"]

    Outputs:
        mean_crossings - mean value of crossings metrics (per timestep)
    """
    df = df.copy().sort_values(["timestep", "object"])
    timesteps = df.timestep.unique()
    timesteps.sort()
    t = len(timesteps)
    crossings = []

    df_prev = df[df.timestep == timesteps[0]]
    for t_ in range(1, t):
        df_cur = df[df.timestep == timesteps[t_]]

        # get objects on both timesteps
        objects_prev = df_prev.object.unique()
        objects_cur = df_cur.object.unique()
        objects = list(set(objects_prev) & set(objects_cur))
        objects.sort()
        n = len(objects)

        # get position of objects
        y_prev = df_prev[df_prev.object.isin(objects)].y.values
        y_cur = df_cur[df_cur.object.isin(objects)].y.values

        # count the number of crossings
        c = 0
        for i in range(n):
            for j in range(i + 1, n):
                if y_prev[i] > y_prev[j] and y_cur[i] < y_cur[j]:
                    c += 1
                if y_prev[i] < y_prev[j] and y_cur[i] > y_cur[j]:
                    c += 1

        crossings.append(c)
        df_prev = df_cur

    mean_crossings = np.mean(crossings)
    return mean_crossings


def jump_distance(df):
    """
    For each subsequents timesteps, track the change in the ordering index of a curve.

    Inputs:
        df - DataFrame with columns ["timestep", "object", "y"]

    Outputs:
        mean_jump - mean value of jump distance metrics (per timestep)
    """

    df = df.copy().sort_values(["timestep", "object"])
    timesteps = df.timestep.unique()
    timesteps.sort()
    t = len(timesteps)
    jump_distances = []

    df_prev = df[df.timestep == timesteps[0]]
    for t_ in range(1, t):
        df_cur = df[df.timestep == timesteps[t_]]

        # get objects on both timestep
        objects_prev = df_prev.object.unique()
        objects_cur = df_cur.object.unique()
        objects = list(set(objects_prev) & set(objects_cur))
        objects.sort()

        # get ordering of position of objects
        y_prev = df_prev[df_prev.object.isin(objects)].y.values
        y_cur = df_cur[df_cur.object.isin(objects)].y.values

        # get rank of objects
        y_prev_rank = np.argsort(np.argsort(y_prev))
        y_cur_rank = np.argsort(np.argsort(y_cur))

        # calculate kendall tau
        jump_distances.append(np.abs(y_cur_rank - y_prev_rank).sum())
        df_prev = df_cur

    mean_jump = np.mean(jump_distances)
    return mean_jump


def check_intervals_intersect(xmin, xmax, ymin, ymax):
    """Check if there is a intersection between two intervals."""
    if xmax < ymin or xmin > ymax:
        return False
    else:
        return True


def check_intervals_order(xmin, xmax, ymin, ymax):
    """Check the order between two intervals."""
    if check_intervals_intersect(xmin, xmax, ymin, ymax):
        return 0
    if xmax < ymin:
        return 1
    else:
        return -1


def compute_intersection_metric(df, area_scaler=None):
    """
    Compute the diferent intersection metrics for each timestep.

    Inputs:
        df - DataFrame with columns ["timestep", "y", "height", "points"]

    Outputs:
        missing_intersections - fraction of missing intersections
        spurious_intersections - fraction of extra intersctions
        spurious_crossings - number of extra crossings
        intersection_area_ratio_mean - mean of ration between 1D and 2D intersection area

    """

    results = {
        "timestep": [],
        "area_1d": [],
        "area_2d": [],
        "intersection_1d": [],
        "intersection_2d": [],
        "spurious_intersection": [],
        "missing_intersection": [],
        "spurious_crossing": [],
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
                y_i = df_o1["y"].values[it]
                y_j = df_o2["y"].values[it]
                height_i = df_o1["height"].values[it]
                height_j = df_o2["height"].values[it]
                I = max(
                    0,
                    min(y_i + height_i / 2, y_j + height_j / 2)
                    - max(y_i - height_i / 2, y_j - height_j / 2),
                )

                region_i = Polygon(df_o1["points"].values[it]).convex_hull
                region_j = Polygon(df_o2["points"].values[it]).convex_hull
                W = region_i.intersection(region_j).area

                # if there is no intersection both in 1D and 2D, continue to next iteration
                if I == 0 and W == 0:
                    continue

                results["timestep"].append(t)
                results["area_1d"].append(I)
                results["area_2d"].append(W)
                results["intersection_1d"].append(I > 0)
                results["intersection_2d"].append(W > 0)
                results["missing_intersection"].append(I == 0 and W > 0)
                results["spurious_intersection"].append(I > 0 and W == 0)
                results["spurious_crossing"].append(False)
                results["shape_type"].append("rect")

    # than, compute intersections between edges
    for i, object_i in enumerate(objects_list):
        for j, object_j in enumerate(objects_list):
            if j <= i:
                continue

            df_object_i = df[df.object == object_i]
            df_object_j = df[df.object == object_j]

            # timestep with both objects
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
                y_i = df_object_i["y"].values[it]
                y_j = df_object_j["y"].values[it]
                height_i = df_object_i["height"].values[it]
                height_j = df_object_j["height"].values[it]
                I.append(
                    max(
                        0,
                        min(y_i + height_i / 2, y_j + height_j / 2)
                        - max(y_i - height_i / 2, y_j - height_j / 2),
                    )
                )
                region_i = Polygon(df_object_i["points"].values[it]).convex_hull
                region_j = Polygon(df_object_j["points"].values[it]).convex_hull
                W.append(region_i.intersection(region_j).area)

            for it, t in enumerate(timesteps_intersection):
                # if is a timestep of a rect, skip it
                if df_object_i["shape_type"].values[it] == "rect":
                    continue

                shape_i = df_object_i["shape"].values[it]
                shape_j = df_object_j["shape"].values[it]

                ymin_left_i = shape_i[0][1]
                ymin_left_j = shape_j[0][1]
                ymax_left_i = shape_i[3][1]
                ymax_left_j = shape_j[3][1]
                ymin_right_i = shape_i[1][1]
                ymin_right_j = shape_j[1][1]
                ymax_right_i = shape_i[2][1]
                ymax_right_j = shape_j[2][1]

                # check if the links cross and if there is any intersection between objects
                links_left_order = check_intervals_order(
                    ymin_left_i, ymax_left_i, ymin_left_j, ymax_left_j
                )
                links_right_order = check_intervals_order(
                    ymin_right_i, ymax_right_i, ymin_right_j, ymax_right_j
                )
                shapes_cross = links_left_order * links_right_order > 0
                spurious_intersection = W[it - 1] == 0 and W[it + 1] == 0

                results["timestep"].append(t)
                results["area_1d"].append(I[it])
                results["area_2d"].append(W[it])
                results["intersection_1d"].append(I[it] > 0)
                results["intersection_2d"].append(W[it] > 0)
                results["missing_intersection"].append(I[it] == 0 and W[it] > 0)
                results["spurious_intersection"].append(spurious_intersection)
                results["spurious_crossing"].append(
                    spurious_intersection and shapes_cross
                )
                results["shape_type"].append("link")

    results = pd.DataFrame(results).sort_values("timestep")
    error_rects = results[results.shape_type == "rect"]
    total_2d_intersection = (
        1
        if error_rects.intersection_2d.sum() == 0
        else error_rects.intersection_2d.sum()
    )
    total_1d_intersection = (
        1
        if error_rects.intersection_1d.sum() == 0
        else error_rects.intersection_1d.sum()
    )
    missing_intersections = (
        error_rects.missing_intersection.sum() / total_2d_intersection
    )
    spurious_intersections = (
        error_rects.spurious_intersection.sum() / total_1d_intersection
    )
    error_links = results[results.shape_type == "link"]
    spurious_crossings = error_links.spurious_crossing.sum()

    if area_scaler is None:
        area_scaler = lambda x: x

    error_rects_2d_intersection = error_rects[error_rects.intersection_2d]
    intersection_area_ratio = (
        error_rects_2d_intersection.area_1d
        / error_rects_2d_intersection.area_2d.apply(area_scaler)
    ).values
    intersection_area_ratio_mean = intersection_area_ratio.mean()

    return (
        missing_intersections,
        spurious_intersections,
        spurious_crossings,
        intersection_area_ratio_mean,
    )


def compute_metrics(df, area_scaler=None):
    """
    Compute metrics for the input dataframe (result of a curve positioning).
    Computed metrics are: missing intersections, spurious intersections, crossings,
        jump distance, stress measure.

    Inputs:
        df - DataFrame with columns ["points", "y", "height", "timestep"]

    Outputs:
        metric - Dictionary with metric errors
    """
    df = df.copy()
    df_rects = df[df.shape_type == "rect"]
    metrics = {}
    (
        missing_intersections,
        spurious_intersections,
        spurious_crossings,
        intersection_area_ratio_mean,
    ) = compute_intersection_metric(df, area_scaler)
    metrics["missing_intersections"] = missing_intersections
    metrics["spurious_intersections"] = spurious_intersections
    metrics["spurious_crossings"] = spurious_crossings
    metrics["intersection_area_ratio_mean"] = intersection_area_ratio_mean
    metrics["crossings_mean"] = crossings(df_rects)
    metrics["jump_distance_mean"] = jump_distance(df_rects)
    metrics["stress_measure"] = stress_measure_shapes(df_rects)

    return metrics
