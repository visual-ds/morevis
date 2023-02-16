import numpy as np
from shapely.geometry import Polygon
import cvxpy as cp
from tqdm import tqdm

if "GUROBI" in cp.installed_solvers():
    solver = "GUROBI"
elif "MOSEK" in cp.installed_solvers():
    solver = "MOSEK"
else:
    raise Exception("You must have Gurobi or Mosek installed to run this script")


def compute_subsets(W):
    """
    Function that recieve a matrix of intersection values for n objects,
    and compute subsets of objects that intersect themselves, i.e.,
    two objects i and j are in a subset only if there is a sequence of objects
    i1, i2, ..., ik that w[i, i1] > 0, w[i1, i2] > 0, ... , w[ik, j] > 0

    Inputs:
        W - numpy array [n, n]

    Output:
        subsets - list of lists, each one is a subset
    """
    n = W.shape[0]
    already_in_subset = [0] * n
    subsets = []

    for i in range(n):
        if already_in_subset[i] == 0:
            already_in_subset[i] = 1
            cur_subset = [i]
            stack = []
            # Find every objects that has intersection bigger than 0 and add to stack
            for k in range(n):
                if k != i and already_in_subset[k] == 0:
                    intersec = W[i, k]
                    if intersec > 0:
                        stack.append(k)
                        cur_subset.append(k)
                        already_in_subset[k] = 1
            # For each group in stack find objects that has intersection bigger than 0
            while len(stack) > 0:
                new_i = stack.pop()
                for k in range(n):
                    if k != new_i and already_in_subset[k] == 0:
                        intersec = W[new_i, k]
                        if intersec > 0:
                            stack.append(k)
                            cur_subset.append(k)
                            already_in_subset[k] = 1
            subsets.append(cur_subset)
    return subsets


def miqp_optim(
    Y,
    L,
    W,
    minimize_distance=True,
    minimize_spurious=True,
    minimize_intersection_area=True,
    lamb1=1,
    lamb2=1,
):
    """
    Solve the optimization problem to adjust rectangles positions to represent intersections.

    Inputs:
        Y - projection values of rectangles (vertical center)
        L - height of rectangles
        W - array matrix of intersections between objects
        minimize_distance - boolean if should minimize the distance change
        minimize_spurious - boolean if should minimize the number of spurious intersections
        minimize_intersection_area - boolean if should minimize the extra intersection area
        lamb1 - weight factor of minimize intersection area
        lamb2 - weight factor of minimize spurious

    Outputs:
        Y_sol - array with solutions
    """

    n = len(Y)
    # if is a trivial problem, don't optimize
    if n <= 1:
        return Y

    f_distance = []
    f_minimize_spurious = []
    f_minimize_intersection_area = []
    constraints = []

    Y = np.array(Y)
    L = np.array(L)
    M = (np.max(Y + L / 2) - np.min(Y - L / 2)) * 100
    Y_var = [cp.Variable(1) for _ in range(n)]
    if minimize_distance:
        f_distance = [cp.square(Y[i] - Y_var[i]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                # If there is intersection in the 2D space
                # Constraints to ensure intersection representation
                upper_bound = -W[i, j] + L[i] / 2 + L[j] / 2

                # |X| <= U  ==>  X <= U  &&  X >= -U
                constraints.append(Y_var[i] - Y_var[j] <= upper_bound)
                constraints.append(Y_var[i] - Y_var[j] >= -upper_bound)

                if minimize_intersection_area:
                    # Constraints to minimize the area of represented intersections
                    k = cp.Variable(1)
                    # constraints.append(k >= 1)
                    f_minimize_intersection_area.append(k)
                    lower_bound = -W[i, j] * k + L[i] / 2 + L[j] / 2

                    # |X| >= U   ==>   X >= U  || X <= -U
                    b = cp.Variable(1, boolean=True)
                    constraints.append(Y_var[i] - Y_var[j] >= lower_bound + b * (-M))
                    constraints.append(
                        Y_var[j] - Y_var[i] >= lower_bound + (1 - b) * (-M)
                    )

            # If there isn't intersection and we want to minimize if
            elif W[i, j] == 0 and minimize_spurious:
                # Constraints to minimize the number of spurious intersections
                c = cp.Variable(1, boolean=True)
                f_minimize_spurious.append(c)
                right_side = (1 - c) * (L[i] / 2 + L[j] / 2)

                # |X| >= U   ==>   X >= U  || X <= -U
                b = cp.Variable(1, boolean=True)
                constraints.append(Y_var[i] - Y_var[j] >= right_side + b * (-M))
                constraints.append(Y_var[j] - Y_var[i] >= right_side + (1 - b) * (-M))

    f = (
        sum(f_distance)
        + lamb1 * sum(f_minimize_intersection_area)
        + lamb2 * sum(f_minimize_spurious)
    )

    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=solver)  # solver = cp.GUROBI)
    Y_sol = [y.value[0] for y in Y_var]

    return Y_sol


def compute_2d_intersection_matrix(df):
    """
    Compute the intersection matrix of the 2D objects.

    Inputs:
        df - DataFrame with column "points"

    Outputs:
        W - array with intersection values
    """
    n = df.shape[0]
    W = np.zeros((n, n))
    polys = df.points.apply(lambda x: Polygon(x).convex_hull).to_list()
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = polys[i].intersection(polys[j]).area
            W[j, i] = W[i, j]
    return W


def compute_variables_subset(subset, Y, L, W):
    """
    Function that compute the variables of the optimization problem when there is
    the separation by subsets.

    Inputs:
        subset - list of index of objects in the subset
        Y - list of the y-coordinates of the objects
        L - list of the heights of the objects
        W - 2d matrix of the intersection area of the objects

    Outputs:
        Y_subset - values of the variables in the subset
        L_subset - list of the heights of the objects in the subset
        W_subset - 2d matrix of the intersection area of the objects in the subset
    """
    Y_subset = [Y[s] for s in subset]
    L_subset = [L[s] for s in subset]
    W_subset = np.zeros((len(subset), len(subset)))
    for i, s_i in enumerate(subset):
        for j, s_j in enumerate(subset):
            W_subset[i, j] = W[s_i, s_j]

    return Y_subset, L_subset, W_subset


def optim_subsets(subsets, Y_sol, L, alpha=0.05):
    """
    After optimization of rectangles, apply an optimization to each of the objects subsets.

    Inputs:
        subsets - list of lists, each of them has the index of the objects of the subsets
        Y_sol - array with rectangles center position
        L - array with rectangles height
        alpha - factor to define a gap between rectangles

    Outputs:
        Y_sol - array updated with solution
    """
    subsets.sort(
        key=lambda subset: (
            min([Y_sol[s] - L[s] / 2 for s in subset])
            + max([Y_sol[s] + L[s] / 2 for s in subset])
        )
        / 2
    )
    subsets_min = [min([Y_sol[s] - L[s] / 2 for s in subset]) for subset in subsets]
    subsets_max = [max([Y_sol[s] + L[s] / 2 for s in subset]) for subset in subsets]
    subsets_height = [subsets_max[i] - subsets_min[i] for i in range(len(subsets))]
    sum_height = sum(subsets_height)
    alpha = (1 - sum_height) * 0.05
    subsets_center = [
        (subsets_min[i] + subsets_max[i]) / 2 for i in range(len(subsets))
    ]

    Y = [cp.Variable(1) for _ in range(len(subsets))]
    # minimze the quadratic distance from the previous position
    f = cp.Minimize(
        cp.sum([cp.square(Y[i] - subsets_center[i]) for i in range(len(subsets))])
    )
    constraints = []
    for i, subset in enumerate(subsets):
        if i > 0:
            constraints.append(
                Y[i] - subsets_height[i] / 2
                >= Y[i - 1] + subsets_height[i - 1] / 2 + alpha
            )

    prob = cp.Problem(f, constraints)
    prob.solve()
    subsets_new_center = [Y[i].value[0] for i in range(len(subsets))]

    for i, y_new in enumerate(subsets_new_center):
        dist = y_new - subsets_center[i]
        for j in subsets[i]:
            Y_sol[j] += dist

    return Y_sol


def optim_handler(
    df,
    area_scaler,
    minimize_distance=True,
    minimize_spurious=True,
    minimize_intersection_area=True,
    lamb1=1,
    lamb2=1,
):
    """
    Optimization procedure to represent intersections. Iterate over timesteps, identifying subsets
    of objects and optimizing them.

    Inputs:
        data - DataFrame with objects data ["timestep", "points", "proj"]
        area_scaler - function to scale the area
        minimize_distance - boolean if should minimize the distance change
        minimize_spurious - boolean if should minimize the number of spurious intersections
        minimize_intersection_area - boolean if should minimize the extra intersection area
        lamb1 - weight factor of minimize intersection area
        lamb2 - weight factor of minimize spurious

    Outputs:
        df - DataFrame with new column "y"

    """
    t_min = int(df.timestep.min())
    t_max = int(df.timestep.max())

    for t in range(t_min, t_max + 1):

        df_t = df.loc[df.timestep == t]
        if df_t.shape[0] == 0:
            continue

        n_objects = df_t.shape[0]
        L = df_t.height.values
        Y = df_t.proj.values
        Y_sol = np.zeros(n_objects)
        W = compute_2d_intersection_matrix(df_t)
        W = np.vectorize(area_scaler)(W)

        subsets = compute_subsets(W)

        for subset in subsets:
            Y_subset, L_subset, W_subset = compute_variables_subset(subset, Y, L, W)

            Y_subset_sol = miqp_optim(
                Y_subset,
                L_subset,
                W_subset,
                minimize_distance,
                minimize_spurious,
                minimize_intersection_area,
                lamb1,
                lamb2,
            )

            for i, s_i in enumerate(subset):
                Y_sol[s_i] = Y_subset_sol[i]

        Y_sol = optim_subsets(subsets, Y_sol, L)

        # save solution in dataframe update values of previous solution
        df.loc[df.timestep == t, "y"] = Y_sol

    return df
