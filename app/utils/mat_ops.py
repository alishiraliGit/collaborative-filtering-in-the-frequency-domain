import numpy as np
from scipy.special import binom


def vectorize_rows(rows_indices, mat, mask=None):
    """
    :param rows_indices:
    :param mat:
    :param mask:
    :return: [-1 x 1]
    """
    vec_rows = vectorize(mat[rows_indices], order='C')

    if mask is None:
        return vec_rows[~np.isnan(vec_rows)].reshape((-1, 1))
    else:
        vec_rows_mask = vectorize(mask[rows_indices], order='C')
        return vec_rows[vec_rows_mask].reshape((-1, 1))


def vectorize(mat, order):
    return mat.reshape((-1, ), order=order)


def unvectorize(vec, n_col, order):
    return vec.reshape((-1, n_col), order=order)


def e1(dim):
    vec = np.zeros((dim,))
    vec[0] = 1
    return vec


def eij(i, j, dim):
    vec = np.zeros((dim,))
    vec[i] = 1
    vec[j] = -1
    return vec


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
