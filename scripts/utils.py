import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def create_connections(df, padding=0.2):
    """
    Create the links the link between the rectangles of the same object.
    The links are simply a quadrilateral connecting the edges from the previous timestep
    to the next timestep. The link is saved as a list of 4 points.

    Inputs:
        df: DataFrame with columns ["object", "timestep", "y", "height"]
        padding: padding to decrease the size of the rectangles

    Outputs:
        df: DataFrame with the column "shape" containing the links

    """
    df = df.sort_values(by=["object", "timestep"])
    columns = df.columns
    df_shapes = {"shape": [], "shape_type": []}

    for col in columns:
        df_shapes[col] = []

    object_ids = df.object.unique()
    object_ids.sort()

    # create first the rects
    for obj in object_ids:
        object_data = df[df.object == obj]
        if object_data.shape[0] == 1:
            t1 = object_data.timestep.values[0]
            t2 = t1 + padding
            t3 = t1 + 1 - padding
            t4 = t1 + 1
            y1 = object_data.y.values[0]
            h1 = object_data.height.values[0]
            df_shapes["shape"].append(
                [
                    [t1, y1],
                    [t2, y1 + h1 / 2],
                    [t3, y1 + h1 / 2],
                    [t4, y1],
                    [t3, y1 - h1 / 2],
                    [t2, y1 - h1 / 2],
                    [t1, y1],
                ]
            )
            df_shapes["shape_type"].append("rect")

            for col in columns:
                df_shapes[col].append(object_data[col].values[0])

        else:

            # add the first rect
            t1 = object_data.timestep.values[0]
            t2 = t1 + padding
            t3 = t1 + 1 - padding
            y1 = object_data.y.values[0]
            h1 = object_data.height.values[0]

            df_shapes["shape"].append(
                [
                    [t1, y1],
                    [t2, y1 + h1 / 2],
                    [t3, y1 + h1 / 2],
                    [t3, y1 - h1 / 2],
                    [t2, y1 - h1 / 2],
                    [t1, y1],
                ]
            )

            df_shapes["shape_type"].append("rect")

            for col in columns:
                df_shapes[col].append(object_data[col].values[0])

            # add the last rect
            t1 = object_data.timestep.values[-1] + padding
            t2 = object_data.timestep.values[-1] + 1 - padding
            t3 = object_data.timestep.values[-1] + 1
            y1 = object_data.y.values[-1]
            h1 = object_data.height.values[-1]

            df_shapes["shape"].append(
                [
                    [t1, y1 + h1 / 2],
                    [t2, y1 + h1 / 2],
                    [t3, y1],
                    [t2, y1 - h1 / 2],
                    [t1, y1 - h1 / 2],
                    [t1, y1 + h1 / 2],
                ]
            )

            df_shapes["shape_type"].append("rect")

            for col in columns:
                df_shapes[col].append(object_data[col].values[-1])

            # intermediary rects
            for i in range(1, object_data.shape[0] - 1):
                t1 = object_data.timestep.values[i]
                t2 = object_data.timestep.values[i] + padding
                t3 = object_data.timestep.values[i] + 1 - padding
                t4 = object_data.timestep.values[i] + 1
                y2 = object_data.y.values[i]
                h2 = object_data.height.values[i]

                df_shapes["shape"].append(
                    [
                        [t2, y2 + h2 / 2],
                        [t3, y2 + h2 / 2],
                        [t3, y2 - h2 / 2],
                        [t2, y2 - h2 / 2],
                        [t2, y2 + h2 / 2],
                    ]
                )
                df_shapes["shape_type"].append("rect")

                for col in columns:
                    df_shapes[col].append(object_data[col].values[i])

    # create link between rectangles
    for obj in object_ids:
        object_data = df[df.object == obj]
        if object_data.shape[0] == 1:
            continue

        # connections between rects
        for i in range(1, object_data.shape[0]):
            t1 = object_data.timestep.values[i] - padding
            t2 = object_data.timestep.values[i] + padding
            y1 = object_data.y.values[i - 1]
            y2 = object_data.y.values[i]
            h1 = object_data.height.values[i - 1]
            h2 = object_data.height.values[i]

            df_shapes["shape"].append(
                [
                    [t1, y1 - h1 / 2],
                    [t2, y2 - h2 / 2],
                    [t2, y2 + h2 / 2],
                    [t1, y1 + h1 / 2],
                    [t1, y1 - h1 / 2],
                ]
            )
            df_shapes["shape_type"].append("link")

            for col in columns:
                df_shapes[col].append(object_data[col].values[i])

            df_shapes["timestep"][-1] -= 0.5

    df_shapes = pd.DataFrame(df_shapes)
    df_shapes["style"] = "solid"
    return df_shapes


def plot_summary(df, ax=None):
    """
    Plot the summary of the data, for each timestep a slice with the positioned rectangles.

    Inputs:
        df: DataFrame with objects information
        ax: matplotlib axis

    """
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(18, 4))

    if "shape" in df.columns:
        for _, row in df.iterrows():
            kwargs = {"color": row.color, "ec": row.color, "lw": 2}

            if "ec_color" in row:
                kwargs["ec"] = row.ec_color

            if "line_width" in row:
                kwargs["lw"] = row.line_width

            if "alpha" in row:
                kwargs["alpha"] = row.alpha

            if "style" in row:
                if row.style == "dashed":
                    kwargs["hatch"] = "//"
                    kwargs["fill"] = False
                elif row.style == "empty":
                    kwargs["fill"] = False

            poly = patches.Polygon(row["shape"], **kwargs)
            ax.add_patch(poly)

        ymin = df["shape"].apply(lambda x: np.array(x)[:, 1].min()).min()
        ymax = df["shape"].apply(lambda x: np.array(x)[:, 1].max()).max()
        ax.set_ylim(ymin - 0.1, ymax + 0.1)

    else:
        for _, row in df.iterrows():
            rect = patches.Rectangle(
                (row.timestep + 0.5 - row.width / 2, row.y - 0.5 - row.height / 2),
                row.width,
                row.height,
                color=row.color,
            )
            ax.add_patch(rect)
        ax.set_ylim(0, df.object.unique().shape[0])

    # compute xticks
    t_min = df.timestep.min()
    t_max = df.timestep.max()
    if t_max - t_min > 20:
        step_size = int((t_max - t_min) / 40)
    else:
        step_size = 1
    xticks = np.arange(t_min, t_max + step_size, step_size)
    ax.set_xticks(xticks)

    ax.set_xlim(df.timestep.min(), df.timestep.max() + 1)
    ax.set_xlabel("Time", fontsize=20, labelpad=-8)
    ax.set_ylabel("Space", fontsize=20, labelpad=-8)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.25)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)


def colormap2D(data, colormap="ziegler"):
    """
    Similar to the work of SpatialRugs, will identify the position of each object (centroid)
    in a 2D space [0, 511]² and get the color of the pixel at this position in the colormap.

    Inputs:
        data: array of shape (n, 2) with the coordinates of the objects
        colormap: name of the colormap to use, must be ["bremm", "schumann_urban", "steiger", "teuling", "ziegler"]

    Outputs:
        colors: array of shape (n, 3) with the RGB colors of the objects
    """
    for i in range(2):
        data[:, i] = (
            (data[:, i] - data[:, i].min())
            / (data[:, i].max() - data[:, i].min())
            * 511
        )
    data = np.floor(data)
    data = data.astype(np.int64)
    colormap = np.array(Image.open(f"../data/colormap/{colormap}.png"))
    colors = colormap[(511 - data[:, 1]), data[:, 0], :].tolist()
    return colors
