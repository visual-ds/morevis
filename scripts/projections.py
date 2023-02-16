from scipy.stats import rankdata
import numpy as np
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
from hilbertcurve.hilbertcurve import HilbertCurve
import pymorton as pm


def projection_selector(df, projection_name, projection_params):
    """
    Call the respective projection method from selections.
    There are projections applied with all the data and projections applied in each timestep.
    Inputs:
        df - dataframe with columns to be used on projections, use columns ["xcenter", "ycenter", "timestep", "object"]
        projection_name - name of projection function to be called
        projection_params - dictonary of parameters to be passed to the projection function

    Output:
        proj - numpy array [n]
    """
    ## PCA projection
    if projection_name == "pca":
        df = pca_proj(df, **projection_params)
    elif projection_name == "pca_timesteps":
        df = pca_proj_timesteps(df, **projection_params)
    ## SPC projection
    elif projection_name == "spc":
        df = StablePrincipalComponent(df, **projection_params)
    ## MDS projection
    elif projection_name == "mds":
        df = mds_proj(df, **projection_params)
    elif projection_name == "mds_timesteps":
        df = mds_proj_timesteps(df, **projection_params)
    ## t-SNE projection
    elif projection_name == "tsne":
        df = tsne_proj(df, **projection_params)
    elif projection_name == "tsne_timesteps":
        df = tsne_proj_timesteps(df, **projection_params)
    ## UMAP projection
    elif projection_name == "umap":
        df = umap_proj(df, **projection_params)
    elif projection_name == "umap_timesteps":
        df = umap_proj_timesteps(df, **projection_params)
    ## Hilbert projection
    elif projection_name == "hilbert":
        df = hilbert_proj(df, **projection_params)
    elif projection_name == "hilbert_timesteps":
        df = hilbert_proj_timesteps(df, **projection_params)
    ## Morton projection
    elif projection_name == "morton":
        df = morton_proj(df, **projection_params)
    elif projection_name == "morton_timesteps":
        df = morton_proj_timesteps(df, **projection_params)
    ## Force directed layout
    elif projection_name == "force":
        df = force_proj(df, **projection_params)
    elif projection_name == "force_timesteps":
        df = force_proj_timesteps(df, **projection_params)
    else:
        raise Exception("Invalid projection.")
    return df


def build_2d_distance_matrix(df):
    """
    Build matrix of distance between polygons.

    Inputs:
        df - DataFrame with column "points"

    Outputs:
        dist_matrix - array with distance values
    """
    n = df.shape[0]
    polys = df.points.apply(lambda x: Polygon(x).convex_hull).values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = polys[i].distance(polys[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


def normalize(y):
    """Normalize a vector."""
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y


def pca_proj(df, order=False):
    """
    Project 2D data to 1D using PCA.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        order - boolean if should return original transformation or return relative orders

    Output:
        df - dataframe with new column ["proj"]

    """
    pca = PCA(n_components=1)
    df["proj"] = pca.fit_transform(df[["xcenter", "ycenter"]].values)
    if order:
        df.proj = rankdata(df.proj, "dense")
    else:
        df.proj = normalize(df.proj)
    return df


def pca_proj_timesteps(df, order=False):
    """
    Project 2D data to 1D using PCA with a projection for each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "timestep"]
        order - boolean if should return original transformation or return relative orders

    Output:
        df - dataframe with new column ["proj"]
    """
    df = df.groupby("timestep").apply(pca_proj, order)
    return df


def mds_proj(df, metric=True, poly_distance=False, order=False):
    """
    Project 2D data to 1D using MDS.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        metric - boolen if should use metric or non-metric MDS
        poly_distance - boolean if should use polygonal distances
        order - boolean if should return original transformation or return relative orders

    Outputs:
        df - dataframe with new column ["proj"]
    """

    if poly_distance:
        mds = MDS(
            metric=metric,
            dissimilarity="precomputed",
            n_components=1,
            eps=1e-16,
            n_init=25,
        )
        df["proj"] = mds.fit_transform(build_2d_distance_matrix(df))
    else:
        mds = MDS(n_components=1, metric=metric, eps=1e-16, n_init=25)
        df["proj"] = mds.fit_transform(df[["xcenter", "ycenter"]].values)

    if order:
        df.proj = rankdata(df.proj, "dense")
    else:
        df.proj = normalize(df.proj)
    return df


def mds_proj_timesteps(df, metric=True, poly_distance=False, order=False):
    """
    Project 2D data to 1D using MDS fitting a projection in each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "timestep"]
        metric - boolen if should use metric or non-metric MDS
        poly_distance - boolean if should use polygonal distances
        order - boolean if should return original transformation or apply ordering

    Outputs:
        df - dataframe with new column ["proj"]

    """
    df = df.groupby("timestep").apply(mds_proj, metric, poly_distance, order)
    return df


def tsne_proj(
    df,
    perplexity=30,
    learning_rate=500,
    iterations=1000,
    poly_distance=False,
    order=False,
):
    """
    Project 2D data to 1D using t-SNE.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        perplexit - int, perplexity of t-SNE
        learning_rate - int, learning rate of t-SNE
        iterations - int, iterations of t-SNE
        poly_distances - boolean if should consider distances between centroids or polygons
        order - boolean if should return original transformation or apply ordering

    Output:
        df - dataframe with new column ["proj]

    """
    if poly_distance:
        metric = "precomputed"
        init = "random"
    else:
        metric = "euclidean"
        init = "pca"

    tsne = TSNE(
        n_components=1,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=iterations,
        metric=metric,
        init=init,
    )

    if poly_distance:
        dist_matrix = build_2d_distance_matrix(df)
        proj = tsne.fit_transform(dist_matrix)
    else:
        data = df[["xcenter", "ycenter"]].to_numpy(copy=True)
        proj = tsne.fit_transform(data)

    if order:
        df["proj"] = rankdata(proj, "dense")
    else:
        df["proj"] = normalize(proj)
    return df


def tsne_proj_timesteps(
    df,
    perplexity=30,
    learning_rate=500,
    iterations=1000,
    poly_distance=False,
    order=False,
):
    """
    Project 2D data to 1D using t-SNE fitting a projection in each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        perplexit - int, perplexity of t-SNE
        learning_rate - int, learning rate of t-SNE
        iterations - int, iterations of t-SNE
        poly_distance - boolean if should consider distances between centroids or polygons
        order - boolean if should return original transformation or apply ordering

    Output:
        df - dataframe with new column ["proj"]

    """

    timesteps = df.timestep.unique()
    timesteps.sort()
    if poly_distance:
        metric = "precomputed"
        init = "random"
    else:
        metric = "euclidean"
        init = "pca"

    for t in timesteps:

        tsne = TSNE(
            n_components=1,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=iterations,
            metric=metric,
            init=init,
        )

        if poly_distance:
            data = df.loc[df.timestep == t].copy()
            dist_matrix = build_2d_distance_matrix(data)
            proj = tsne.fit_transform(dist_matrix)
        else:
            data = df.loc[df.timestep == t, ["xcenter", "ycenter"]].to_numpy(copy=True)
            proj = tsne.fit_transform(data)

        init = proj

        if order:
            proj = rankdata(proj, "dense")
        else:
            proj = normalize(proj)

        df.loc[df.timestep == t, "proj"] = proj
    return df


def umap_proj(
    df,
    n_neighbors=8,
    min_dist=0.1,
    spread=1.0,
    n_epochs=200,
    poly_distance=False,
    order=False,
):
    """
    Project 2D data to 1D using UMAP.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        order - boolean if should return original transformation or apply ordering
        n_neighbors - int, number of neighbors to use in UMAP
        min_dist - float, minimum distance to use in UMAP
        spread - float, spread of UMAP
        n_epochs - int, number of epochs to run UMAP

    Output:
        proj - numpy array [n]

    """
    if poly_distance:
        metric = "precomputed"
    else:
        metric = "euclidean"

    init = pca_proj(df)["proj"].values.reshape(-1, 1)

    if poly_distance:
        data = build_2d_distance_matrix(df)
    else:
        data = df[["xcenter", "ycenter"]].values
    reducer = UMAP(
        n_components=1,
        n_neighbors=min(n_neighbors, data.shape[0] - 1),
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        n_epochs=n_epochs,
        init=init,
    )
    reducer.fit(data)
    df["proj"] = reducer.transform(data)

    if order:
        df["proj"] = rankdata(df["proj"], "dense")
    else:
        df["proj"] = normalize(df["proj"])
    return df


def umap_proj_timesteps(
    df,
    n_neighbors=8,
    min_dist=0.1,
    spread=1.0,
    n_epochs=200,
    poly_distance=False,
    order=False,
):
    """
    Project 2D data to 1D using UMAP fitting a projection in each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        n_neighbors - int, number of neighbors to use in UMAP
        min_dist - float, minimum distance to use in UMAP
        spread - float, spread of UMAP
        poly_distance - boolean if should consider distances between centroids or polygons
        order - boolean if should return original transformation or apply ordering

    Output:
        df - dataframe with new column ["proj"]

    """

    timesteps = df.timestep.unique()
    timesteps.sort()
    if poly_distance:
        metric = "precomputed"
    else:
        metric = "euclidean"

    data = df.loc[df.timestep == timesteps[0], ["xcenter", "ycenter", "points"]]
    init = pca_proj(data)["proj"].values.reshape(-1, 1)
    for t in timesteps:
        data = df.loc[df.timestep == t, ["xcenter", "ycenter", "points"]]
        reducer = UMAP(
            n_components=1,
            n_neighbors=min(n_neighbors, data.shape[0] - 1),
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            metric=metric,
            init=init,
        )

        if poly_distance:
            dist_matrix = build_2d_distance_matrix(data)
            reducer.fit(dist_matrix)
            proj = reducer.transform(dist_matrix)
        else:
            data = data[["xcenter", "ycenter"]].values
            reducer.fit(data)
            proj = reducer.transform(data)

        init = proj

        if order:
            proj = rankdata(proj, "dense")
        else:
            proj = normalize(proj)
        df.loc[df.timestep == t, "proj"] = proj
    return df


def hilbert_proj(df, level=5, order=False):
    """
    Project 2D data to 1D using Hilbert space-filling curve.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        level - int order of curve
        order - boolean if should return original transformation or apply ordering

    Output:
        proj - numpy array [n]

    """
    curve = HilbertCurve(level, 2)
    data = df[["xcenter", "ycenter"]].to_numpy(copy=True)
    for i in range(2):
        data[:, i] = data[:, i] - data[:, i].min()
        if data[:, i].max() != 0:
            data[:, i] = data[:, i] / data[:, i].max()
        else:
            data[:, i] = 0
        data[:, i] = data[:, i] * (2 ** (level) - 1)
    data = data.astype(np.uint32)
    data = data.tolist()
    proj = curve.distances_from_points(data)
    if order:
        proj = rankdata(proj, "dense")
    else:
        proj = normalize(proj)
    df["proj"] = proj
    return df


def hilbert_proj_timesteps(df, level=5, order=False):
    """
    Project 2D data to 1D using Hilbert space-filling curve making a projection in each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "timestep"]
        level - int order of curve
        order - boolean if should return original transformation or apply ordering

    Output:
        proj - numpy array [n]

    """
    df = df.groupby("timestep").apply(hilbert_proj, level, order)
    return df


def morton_proj(df, level=5, order=False):
    """
    Project 2D data to 1D using Morton space-filling curve.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter"]
        level - int order of curve
        order - boolean if should return original transformation or apply ordering

    Output:
        proj - numpy array [n]

    """
    data = df[["xcenter", "ycenter"]].to_numpy(copy=True)
    for i in range(2):
        data[:, i] = data[:, i] - data[:, i].min()
        if data[:, i].max() != 0:
            data[:, i] = data[:, i] / data[:, i].max()
        else:
            data[:, i] = 0
        data[:, i] = data[:, i] * (2 ** (level) - 1)
    data = data.astype(np.uint32)

    proj = np.array([pm.interleave2(int(coord[0]), int(coord[1])) for coord in data])

    if order:
        proj = rankdata(proj, "dense")
    else:
        proj = normalize(proj)

    df["proj"] = proj
    return df


def morton_proj_timesteps(df, level=5, order=False):
    """
    Project 2D data to 1D using Morton space-filling curve fitting a projection in each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "timestep"]
        level - int order of curve
        order - boolean if should return original transformation or apply ordering

    Output:
        proj - numpy array [n]

    """
    df = df.groupby("timestep").apply(morton_proj, level, order)
    return df


def StablePrincipalComponent(df, sigma=0.5, order=False):
    """
    Project data from 2D to 1D using the Stable Principal Component method from the paper "Stable Visual Summaries"
    Makes a PCA projection in each timestep and preserve the stability between timesteps.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "timestep"]
        order - boolean if should return original transformation or apply ordering

    Output:
        df - dataframe with new column ["proj"]
    """

    def PCA_eigenvalues(data):
        """
        With the data, compute the eigenvalues and eigenvector of the covariance matrix.
        Return the two biggest eigenvalues (in order) and the first principal component.
        """
        data -= data.mean(axis=0)
        M = np.cov(data, rowvar=False)
        evals, evecs = np.linalg.eig(M)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        return evals[0], evals[1], evecs[:, 0]

    def angle_between_vectors(v, w):
        """Return the angle in radians between vectors 'v' and 'w'."""
        v1 = v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w))
        if v1 > 1:
            v1 = 1
        return np.arccos(v1)

    principal_components = []
    t0 = int(df.timestep.min())
    t_ = t0
    alpha = 0
    timestamps = int(df.timestep.max() + 1)

    ## First iteration of projection
    coords = df.loc[df.timestep == t_, ["xcenter", "ycenter"]].to_numpy(copy=True)
    if coords.shape[0] == 1:
        eval1, eval2, evec1 = np.max(coords), np.min(coords), np.array([1, 0])
    else:
        eval1, eval2, evec1 = PCA_eigenvalues(coords)
    principal_components.append(evec1)

    ## For each of the timestamps
    for t in range(t0 + 1, timestamps):
        coords = df.loc[df.timestep == t, ["xcenter", "ycenter"]].to_numpy(copy=True)

        # If there is no data, keep the previous eigenvector
        if coords.shape[0] <= 1:
            principal_components.append(principal_components[-1])
        else:
            eval1, eval2, evec1 = PCA_eigenvalues(coords)
            # Verify the direction of the new eigenvector
            if principal_components[-1].dot(evec1) < 0:
                evec1 *= -1

            principal_components.append(evec1)
            alpha += angle_between_vectors(
                principal_components[-1], principal_components[-2]
            )

            # If there is a big angle, change the previous eigenvectors to be a gradual change
            if eval2 / eval1 <= sigma or t == (timestamps - 1):
                for t_s in range(t_ + 1, t):
                    rotation_angle = alpha * (t_s - t_) / (t - t_)
                    rotation_matrix = np.array(
                        [
                            [np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)],
                        ]
                    )
                    principal_components[t_s - t0] = np.dot(
                        rotation_matrix, principal_components[t_ - t0]
                    )

                t_ = t
                alpha = 0

    proj = np.zeros(df.shape[0])
    k = 0
    # With the eigen vectors, project each timestep
    for t in range(t0, timestamps):
        M = df.loc[df.timestep == t, ["xcenter", "ycenter"]].to_numpy(copy=True)
        proj_t = np.dot(principal_components[k].T, M.T).T
        if order:
            proj_t = rankdata(proj_t, "ordinal")
        proj[df.timestep == t] = proj_t
        k += 1

    if not order:
        df["proj"] = normalize(proj.astype(np.float32))
        df["proj"] = 1 - df["proj"]  # to have the same orientation as PCA
    else:
        df["proj"] = proj
    return df


def interval_distance(xmin, xmax, ymin, ymax):
    """Compute the distance between two intervals."""
    # if there is no intersection between intervals
    if xmax < ymin or xmin > ymax:
        return min(abs(xmin - ymax), abs(xmax - ymin))
    else:
        return 0


def force_proj(df, max_iter=5, frac=0.05, tol=1e-4, order=False):
    """
    Project data from 2D to 1D using the Force-directed layout method.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "points"]
        max_iter - number of iterations
        frac - multiplier of the jump size
        tol - tolerante to verify if there was any update between iterations
        order - boolean if should return original transformation or apply ordering

    Output:
        proj - numpy array [n]

    """
    n = df.shape[0]
    dist_matrix = build_2d_distance_matrix(df)
    dmin = dist_matrix.min()
    dmax = dist_matrix.max()
    dist_matrix = (dist_matrix - dmin) / (dmax - dmin)

    y = np.zeros(n)
    h = df.height.values
    y_prev = y

    for _ in range(max_iter):
        for i in range(n):
            x_ = y[i]
            for j in range(n):
                if j == i:
                    continue
                q_ = y[j]

                if x_ + h[i] / 2 < q_ - h[j] / 2:
                    v = 1
                elif x_ - h[i] / 2 > q_ + h[j] / 2:
                    v = -1
                else:
                    v = 1

                dist_1d = interval_distance(
                    x_ - h[i] / 2, x_ + h[i] / 2, q_ - h[j] / 2, q_ + h[j] / 2
                )
                dist_2d = dist_matrix[i, j]
                delta = dist_2d - dist_1d
                move = v * delta * frac
                y[j] += move

        y = normalize(y)
        if np.abs(y - y_prev).max() < tol:
            break
        y_prev = y

    if order:
        df["proj"] = rankdata(y, "ordinal")
    else:
        df["proj"] = normalize(y)
    return df


def force_proj_timesteps(df, max_iter=5, frac=0.05, tol=1e-4, order=False):
    """
    Project data from 2D to 1D using the Force-directed layout method fittint a projection in each timestep.

    Inputs:
        df - dataframe with columns ["xcenter", "ycenter", "points"]
        max_iter - number of iterations
        frac - multiplier of the jump size
        tol - tolerante to verify if there was any update between iterations
        order - boolean if should return original transformation or apply ordering

    Output:
        proj - numpy array [n]

    """

    df = df.groupby("timestep").apply(
        force_proj, max_iter=max_iter, frac=frac, tol=tol, order=order
    )
    return df
