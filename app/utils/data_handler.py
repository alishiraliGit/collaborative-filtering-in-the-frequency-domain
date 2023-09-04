import csv
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
import os
import pandas as pd


def load_dataset(loadpath, name, va_split=None, te_split=None, do_transpose=False, random_state=1, **kwargs):

    rng = default_rng(random_state)

    if name == 'ml-100k':
        edges_notmapped_tr_va = get_edge_list_from_file_ml100k(loadpath, 'u%d.base' % kwargs['part'])
        edges_notmapped_te = get_edge_list_from_file_ml100k(loadpath, 'u%d.test' % kwargs['part'])

        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped_tr_va + edges_notmapped_te)

        edges_tr_va = edges[:len(edges_notmapped_tr_va)]
        assert te_split == 0.2, 'test split is not consistent with predefined test sets.'
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split/(1 - te_split), random_state=random_state)
        edges_te = edges[len(edges_notmapped_tr_va):]

        rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
        rat_mat_va = get_rating_mat(edges_va, num_user, num_item)
        rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

        # From dataset spec
        min_val = 1
        max_val = 5

    elif name == 'ml-1m':
        edges_notmapped = get_edge_list_from_file_ml1m(loadpath, 'ratings.dat')
        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped)

        edges_tr_va, edges_te = train_test_split(edges, test_size=te_split, random_state=random_state)
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split/(1 - te_split), random_state=random_state)

        rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
        rat_mat_va = get_rating_mat(edges_va, num_user, num_item)
        rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

        # From dataset spec
        min_val = 1
        max_val = 5

    elif name == 'jester':
        df_1 = pd.read_excel(os.path.join(loadpath, 'jester-data-1.xls'))
        df_2 = pd.read_excel(os.path.join(loadpath, 'jester-data-2.xls'))

        rat_mat_1 = df_1.to_numpy()[:, 1:]
        rat_mat_2 = df_2.to_numpy()[:, 1:]

        # Concat two datasets
        rat_mat = np.concatenate((rat_mat_1, rat_mat_2), axis=0)

        # ToDo
        n_sub_user = 2000
        rat_mat = rat_mat[:n_sub_user, :]

        # Fill unrated cells with NaNs
        rat_mat[rat_mat == 99] = np.NaN

        # Train-test split
        rated_indices = np.where(~np.isnan(rat_mat))

        tr_va_idx, te_idx = \
            train_test_split(range(len(rated_indices[0])), test_size=te_split, random_state=random_state)
        tr_idx, va_idx = train_test_split(tr_va_idx, test_size=va_split/(1 - te_split), random_state=random_state)

        tr_rated_indices = (rated_indices[0][tr_idx], rated_indices[1][tr_idx])
        va_rated_indices = (rated_indices[0][va_idx], rated_indices[1][va_idx])
        te_rated_indices = (rated_indices[0][te_idx], rated_indices[1][te_idx])

        # Init. rating matrices
        rat_mat_tr = np.empty(rat_mat.shape)
        rat_mat_tr[:] = np.NaN

        rat_mat_va = rat_mat_tr.copy()
        rat_mat_te = rat_mat_tr.copy()

        # Fill rating matrices
        rat_mat_tr[tr_rated_indices] = rat_mat[tr_rated_indices]
        rat_mat_va[va_rated_indices] = rat_mat[va_rated_indices]
        rat_mat_te[te_rated_indices] = rat_mat[te_rated_indices]

        num_user, num_item = rat_mat.shape

        # From dataset spec
        min_val = -10
        max_val = 10

    elif name == 'monday_offers':
        df = pd.read_csv(os.path.join(loadpath, 'users_n_offering(binary).csv'))

        # Convert to numpy array
        rat_mat = df.to_numpy()[::5]

        # Train-test split
        rated_indices = np.where(~np.isnan(rat_mat))

        tr_va_idx, te_idx = \
            train_test_split(range(len(rated_indices[0])), test_size=te_split, random_state=random_state)
        tr_idx, va_idx = train_test_split(tr_va_idx, test_size=va_split, random_state=random_state)

        tr_rated_indices = (rated_indices[0][tr_idx], rated_indices[1][tr_idx])
        va_rated_indices = (rated_indices[0][va_idx], rated_indices[1][va_idx])
        te_rated_indices = (rated_indices[0][te_idx], rated_indices[1][te_idx])

        # Init. rating matrices
        rat_mat_tr = np.empty(rat_mat.shape)
        rat_mat_tr[:] = np.NaN

        rat_mat_va = rat_mat_tr.copy()
        rat_mat_te = rat_mat_tr.copy()

        # Fill rating matrices
        rat_mat_tr[tr_rated_indices] = rat_mat[tr_rated_indices]
        rat_mat_va[va_rated_indices] = rat_mat[va_rated_indices]
        rat_mat_te[te_rated_indices] = rat_mat[te_rated_indices]

        num_user, num_item = rat_mat.shape

        # From dataset spec
        min_val = 0
        max_val = 1

    elif name == 'coat':
        file_path_tr = os.path.join(loadpath, 'train.ascii')
        file_path_te = os.path.join(loadpath, 'test.ascii')

        rat_mat_tr_va = np.genfromtxt(fname=file_path_tr, delimiter=' ', dtype=np.float)
        rat_mat_te = np.genfromtxt(fname=file_path_te, delimiter=' ', dtype=np.float)

        rat_mat_tr_va[rat_mat_tr_va == 0] = np.nan
        rat_mat_te[rat_mat_te == 0] = np.nan

        num_user, num_item = rat_mat_tr_va.shape

        # Train-validation split
        mask_tr = np.ones((num_user, num_item), dtype=bool)
        mask_tr[rng.random(size=(num_user, num_item)) < va_split] = False

        rat_mat_tr = rat_mat_tr_va.copy()
        rat_mat_tr[~mask_tr] = np.nan

        rat_mat_va = rat_mat_tr_va.copy()
        rat_mat_va[mask_tr] = np.nan

        # From dataset spec
        min_val = 1
        max_val = 5

    elif name == 'yahoo-r3':
        edges_notmapped = get_edge_list_from_file_yahoo_r3(loadpath, 'ydata-ymusic-rating-study-v1_0-train.txt')
        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped)

        edges_tr_va, edges_te = train_test_split(edges, test_size=te_split, random_state=random_state)
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split/(1 - te_split), random_state=random_state)

        rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
        rat_mat_va = get_rating_mat(edges_va, num_user, num_item)
        rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

        # From dataset spec
        min_val = 1
        max_val = 5

    else:
        raise Exception('%s is not a valid dataset.' % name)

    if do_transpose:
        rat_mat_tr = rat_mat_tr.T
        rat_mat_va = rat_mat_va.T
        rat_mat_te = rat_mat_te.T
        num_user, num_item = num_item, num_user

    return rat_mat_tr, rat_mat_va, rat_mat_te, num_user, num_item, min_val, max_val


def get_edge_list_from_file_ml100k(file_path, file_name):
    edges = []
    with open(file_path + '/' + file_name) as data_file:
        data_reader = csv.reader(data_file, delimiter='\t')

        for row in data_reader:
            user, item, value = int(row[0]), int(row[1]), float(row[2])

            edges += [(user, item, value)]

    return edges


def get_edge_list_from_file_ml1m(file_path, file_name):
    edges = []
    with open(file_path + '/' + file_name) as data_file:
        data_reader = csv.reader(data_file, delimiter=':')

        for row in data_reader:
            user, item, value = int(row[0]), int(row[2]), float(row[4])

            edges += [(user, item, value)]

    return edges


def get_edge_list_from_file_yahoo_r3(file_path, file_name):
    edges = []
    with open(file_path + '/' + file_name) as data_file:
        data_reader = csv.reader(data_file, delimiter='\t')

        for row in data_reader:
            user, item, value = int(row[0]), int(row[1]), float(row[2])

            edges += [(user, item, value)]

    return edges


def map_ids(edges):
    # map to 0, 1, 2, ... range
    last_u = -1
    last_i = -1
    map_u = {}
    map_i = {}
    new_edges = []
    for user, item, val in edges:
        if user not in map_u:
            last_u += 1
            map_u[user] = last_u

        if item not in map_i:
            last_i += 1
            map_i[item] = last_i

        new_edges += [(map_u[user], map_i[item], val)]

    return new_edges, map_u, map_i, last_u + 1, last_i + 1


def get_rating_mat(edges, n_user, n_item):
    rating_mat = np.empty((n_user, n_item))
    rating_mat[:] = np.NaN

    for user, item, val in edges:
        rating_mat[user, item] = val

    return rating_mat


def load_propensity_scores(loadpath, name, do_transpose=False):
    if name == 'coat':
        file_path_prop = os.path.join(loadpath, 'propensities.ascii')

        propensity_mat = np.genfromtxt(fname=file_path_prop, delimiter=' ', dtype=np.float)
    else:
        raise Exception('Cannot find propensity scores for %s.' % name)

    if do_transpose:
        propensity_mat = propensity_mat.T

    return propensity_mat


if __name__ == '__main__':
    load_path = os.path.join('..', '..', 'data', 'yahoo-r3')

    rating_mat_tr, rating_mat_va, rating_mat_te, n_u, n_i, min_value, max_value = \
        load_dataset(load_path, 'yahoo-r3', va_split=0.05, te_split=0.1, random_state=1)
