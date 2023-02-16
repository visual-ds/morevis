import matplotlib.pyplot as plt
import numpy as np
from projections import projection_selector
from utils import colormap2D, create_connections, plot_summary
from morevis import compute_height_constant
from metrics import compute_intersection_metric


def set_color(df, color_column_name, colormap_name):
    """
    Set the color of rectangles based on a column.

    Inputs:
        df - DataFrame with columns [color_column_name]
        color_column_name - column to define the color
        colormap_name - name of color map/scale

    Outputs:
        df - DataFrame with new column "color"
    """
    color_min = df[color_column_name].min()
    color_max = df[color_column_name].max()
    colormap = plt.get_cmap(colormap_name)
    if df[color_column_name].dtype == "float":
        color_min = df[color_column_name].min()
        color_max = df[color_column_name].max()
        df["color"] = [
            colormap((c - color_min) / (color_max - color_min))
            for c in df[color_column_name]
        ]
    else:
        color_index = df[color_column_name].unique()
        color_index = np.sort(color_index)
        color_index = dict(
            [(c, i % len(colormap.colors)) for i, c in enumerate(color_index)]
        )
        df["color"] = [colormap(color_index[c]) for c in df[color_column_name]]
    return df


def MotionRugs(
    input_df,
    projection_name="hilbert_timesteps",
    projection_params={"level": 5, "order": True},
    proportional_height=False,
    height_method="area_max",
    color_column_name="object",
    colormap_name="bwr",
    ax=None,
):
    """
    MotionRugs visualization. Recieves a dataframe with columns ["xcenter", "ycenter", "timestep", "object"],
    each object is projected to 1D with the selected projection method.
    After it is defined the height of each object, and the color.

    Inputs:
        - input_df: dataframe with columns ["xcenter", "ycenter", "timestep", "object"]
        - projection_name: name of the projection to use
        - projection_params: parameters for the projection
        - proportional_height: if the height of cells will be proportional to the area
        - height_method: method to compute the height of each cell
        - color_column_name: name of the column with the color of each object
        - colormap_name: name of the colormap to use
        - ax: axis to plot the rug plot

    Outputs:
        - df: dataframe with columns ["xcenter", "ycenter", "timestep", "object", "y", "height", "color"]
    """

    df = input_df.copy()
    df = projection_selector(df, projection_name, projection_params)
    df["y"] = df["proj"]

    if proportional_height:
        area_scaler = compute_height_constant(df, height_method)
        df["height"] = df["area"].apply(area_scaler)
        df["width"] = df["height"]
    else:
        df["height"] = 1
        df["width"] = 1

    if not "color" in df.columns:
        df = set_color(df, color_column_name, colormap_name)

    plot_summary(df, ax=ax)
    return df


def SpatialRugs(
    input_df,
    projection_name="hilbert_timesteps",
    projection_params={"level": 5, "order": True},
    proportional_height=False,
    height_method="area_max",
    colormap_name="steiger",
    ax=None,
):
    """
    SpatialRugs visualization. Recieves a dataframe with columns ["xcenter", "ycenter", "timestep", "object"],
    and create a visualization similar to the MotionRugs function, but uses
    a 2D colormap to represent space.

    Inputs:
        - input_df: dataframe with columns ["xcenter", "ycenter", "timestep", "object"]
        - projection_name: name of the projection to use
        - projection_params: parameters for the projection
        - proportional_height: if the height of cells will be proportional to the area
        - height_method: method to compute the height of each cell
        - colormap_name: name of the 2D colormap to use
        - ax: axis to plot the rug plot

    Outputs:
        - df: dataframe with columns ["xcenter", "ycenter", "timestep", "object", "y", "height", "color"]

    """

    df = input_df.copy()
    df = projection_selector(df, projection_name, projection_params)
    df["y"] = df["proj"]

    if proportional_height:
        area_scaler = compute_height_constant(df, height_method)
        df["height"] = df["area"].apply(area_scaler)
        df["width"] = df["height"]
    else:
        df["height"] = 1
        df["width"] = 1

    df["color"] = colormap2D(df[["xcenter", "ycenter"]].values, colormap_name)
    df["color"] = df["color"].apply(lambda x: [c / 255 for c in x])

    plot_summary(df, ax=ax)
    return df


def MotionLines(
    input_df,
    projection_name="spc",
    projection_params={"sigma": 0.6},
    proportional_height=False,
    height_method="area_max",
    colormap_name="steiger",
    color_column_name=None,
    plot=True,
    plot_metrics=False,
    ax=None,
):
    """
    MotionLines visualization. Recieves a dataframe with columns ["xcenter", "ycenter", "timestep", "object"],
    and create a visualization similar to the MotionRugs function, but uses
    a 2D colormap to represent space.

    Inputs:
        - input_df: dataframe with columns ["xcenter", "ycenter", "timestep", "object"]
        - projection_name: name of the projection to use
        - projection_params: parameters for the projection
        - proportional_height: if the height of cells will be proportional to the area
        - height_method: method to compute the height of each cell
        - colormap_name: name of the 2D colormap to use
        - color_column_name: name of the column with the color of each object
        - plot: if the visualization will be plotted
        - plot_metrics: if the metrics of the visualization will be plotted
        - ax: axis to plot the rug plot

    Outputs:
        - df: dataframe with columns ["xcenter", "ycenter", "timestep", "object", "y", "height", "color", "shape"]

    """

    df = input_df.copy()
    df = projection_selector(df, projection_name, projection_params)
    df["y"] = df["proj"]

    if proportional_height:
        area_scaler = compute_height_constant(df, height_method)
        df["height"] = df["area"].apply(area_scaler)
        df["width"] = df["height"]
    else:
        df["height"] = 0.075
        df["width"] = 0.075

    if color_column_name != None:
        df = set_color(df, color_column_name, colormap_name)
    else:
        df["color"] = colormap2D(df[["xcenter", "ycenter"]].values, colormap_name)
        df["color"] = df["color"].apply(lambda x: [c / 255 for c in x])

    df = create_connections(df)
    if plot:
        plot_summary(df, ax=ax)

    if plot_metrics:
        (
            missing_intersections,
            spurious_intersections,
            _,
            _,
        ) = compute_intersection_metric(df)
        ax.annotate(
            f"Missing: {int(100*missing_intersections)}%",
            (0.02, 0.8),
            xycoords="axes fraction",
            fontsize=16,
        )
        ax.annotate(
            f"Spurious: {int(100*spurious_intersections)}%",
            (0.02, 0.9),
            xycoords="axes fraction",
            fontsize=16,
        )

    return df
