import numpy as np
from scipy.special import binom


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


def get_ari(mat):
    a = np.sum(mat, axis=1)
    b = np.sum(mat, axis=0)

    t1 = 0
    for a_i in a:
        t1 += binom(a_i, 2)

    t2 = 0
    for b_i in b:
        t2 += binom(b_i, 2)

    t3 = t1*t2 / binom(np.sum(mat), 2)

    t = 0
    for n_i in mat.reshape((-1, )):
        t += binom(n_i, 2)

    ari = (t - t3)/((t1 + t2)/2 - t3)

    return ari
