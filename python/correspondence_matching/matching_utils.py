import numpy as np
from scipy import optimize
from scipy.spatial import distance


def greedy_assignment(x):
    """
    Attempt to find maximum assignment via greedy algorithm
    """

    assert x.ndim == 2
    x_copy = x.copy()
    row_ind, col_ind = [], []

    while (x_copy > 0).any():
        row, col = np.unravel_index(np.argmax(x_copy, axis=None), x_copy.shape)
        row_ind.append(row)
        col_ind.append(col)

        x_copy[row, :] = 0
        x_copy[:, col] = 0

    return np.array(row_ind), np.array(col_ind)


def assign_cluster_matches(matrix, algo="hungarian"):
    """
    """

    if algo == "hungarian":
        assignment_algo = optimize.linear_sum_assignment
        modifier = -1  # this algo minimizes the cost, so invert the signs
    elif algo == "greedy":
        assignment_algo = greedy_assignment
        modifier = 1
    else:
        raise NotImplementedError(f"Assignment algorithm '{algo}' not implemented.")

    # Perform cluster matching
    col_pct = matrix / matrix.sum(axis=0)
    row_pct = matrix / matrix.sum(axis=1)[:, None]

    col_pct = np.nan_to_num(col_pct)
    row_pct = np.nan_to_num(row_pct)

    match_strengths = (col_pct + row_pct) / 2
    row_ind, col_ind = assignment_algo(modifier * match_strengths)

    return match_strengths, row_ind, col_ind


def transform_distance(x, *args):
    """
    Find mean distance between matching points after rotation and translation.

    :param x: ndarray of shape (3,), translation_x, translation_y and rotation angle
    :param args: 2-tuple, (left_points, right_points) where each has shape (n, 2)
    """

    assert x.ndim == 1
    assert x.shape[0] == 3
    translation_x, translation_y, theta = x

    left_points, right_points = args
    assert left_points.ndim == 2
    assert left_points.shape[1] == 2
    assert right_points.ndim == 2
    assert right_points.shape[1] == 2

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed_points = np.matmul(rot_matrix, left_points.T).T
    transformed_points += [translation_x, translation_y]

    transform_dist = np.diagonal(distance.cdist(transformed_points, right_points))

    return np.mean(transform_dist)
