import pandas as pd
import numpy as np
from ast import literal_eval
from shapely.geometry import Polygon


def groupby_objects_timestep(df, agg_dict={}):
    """
    Function that for each object, group all the rows that are from the same time interval (timestep),
    calculating the mean position and a new area value with convex hulls.

    Inputs:
        df - dataframe with timstep and object information, and also columns for position and points
        agg_dict - dictionary with extra operations for aggregation

    Output:
        df - grouped dataframe
    """
    sum_lambda = lambda t: sum(t, [])
    agg_lambda = lambda t: list(Polygon(sum(t, [])).convex_hull.exterior.coords)
    agg_dict = dict(
        agg_dict,
        **{
            "xcenter": "mean",
            "ycenter": "mean",
            "points": sum_lambda,
            "points_coords": agg_lambda,
        },
    )

    df = df.groupby(["object", "timestep"]).agg(agg_dict).reset_index()

    def convex_hull_area(t):
        if len(t) >= 3:
            return Polygon(t).convex_hull.area
        else:
            return 1

    df["area"] = df.points.apply(convex_hull_area)
    return df


def load_dataset(dataset):
    """Load the selected dataset."""
    if dataset == "motivating":
        df = pd.read_csv("../data/processed/motivating.csv")
        df.points = df.points.apply(literal_eval)

    elif dataset == "wildtrack_use_case":
        df = pd.read_csv(f"../data/processed/wildtrack.csv")
        df.points = df.points.apply(literal_eval)

        # use-case filtering
        use_case_objects = [
            358,
            406,
            378,
            375,
            592,
            224,
            617,
            80,
            83,
            86,
            1108,
            91,
            92,
            93,
        ]
        df = df[df.object.isin(use_case_objects)]

    elif dataset == "wildtrack_use_case_2":
        df = pd.read_csv(f"../data/processed/wildtrack.csv")
        df.points = df.points.apply(literal_eval)

        # use-case filtering
        # get objects with more than 200 rows
        use_case_objects = df.object.value_counts()[
            df.object.value_counts() > 200
        ].index
        df = df[df.object.isin(use_case_objects)]

    elif dataset == "hurdat_use_case":
        df = pd.read_csv("../data/processed/hurdat.csv")
        df["points"] = df.points.apply(literal_eval)
        df["points_coords"] = df.points_coords.apply(literal_eval)

        agg_dict = {
            "wind": "max",
            "pressure": "max",
            "longitude": "mean",
            "latitude": "mean",
            "longitude_start": "first",
            "latitude_start": "first",
            "name": "first",
        }

        # Filtering by start_lon and start_lat
        df = df[((df.longitude_start <= -20) & (df.longitude_start >= -50))]
        df = df[(df.latitude_start <= 20) & (df.latitude_start >= 10)]

        # Group events by time interval
        # 1 day
        time_interval = 86400 * 2
        df["timestep"] = (df.time.astype(float) / time_interval).apply(np.floor)
        df = groupby_objects_timestep(df, agg_dict)

    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return df
