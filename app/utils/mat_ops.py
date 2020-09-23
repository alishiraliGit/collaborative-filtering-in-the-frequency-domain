import numpy as np


def vectorize_rows(rows_indices, mat):
    """
    :param rows_indices:
    :param mat:
    :return: [-1 x 1]
    """
    vec_rows = mat[rows_indices].reshape((-1,), order='C')
    return vec_rows[~np.isnan(vec_rows)].reshape((-1, 1))


def vectorize(mat, order):
    return mat.reshape((-1, ), order=order)


def unvectorize(vec, n_col, order):
    return vec.reshape((-1, n_col), order=order)
