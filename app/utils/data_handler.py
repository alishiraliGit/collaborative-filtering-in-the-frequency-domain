import csv
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd


def load_dataset(loadpath, name, va_split=None, te_split=None, part=1, do_transpose=False):

    if name == 'ml-100k':
        edges_notmapped_tr_va = get_edge_list_from_file_ml100k(loadpath, 'u%d.base' % part)
        edges_notmapped_te = get_edge_list_from_file_ml100k(loadpath, 'u%d.test' % part)

        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped_tr_va + edges_notmapped_te)

        edges_tr_va = edges[:len(edges_notmapped_tr_va)]
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split)
        edges_te = edges[len(edges_notmapped_tr_va):]

        rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
        rat_mat_va = get_rating_mat(edges_va, num_user, num_item)
        rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

    elif name == 'ml-1m':
        edges_notmapped = get_edge_list_from_file_ml1m(loadpath, 'ratings.dat')
        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped)

        edges_tr_va, edges_te = train_test_split(edges, test_size=te_split)
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split)

        rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
        rat_mat_va = get_rating_mat(edges_va, num_user, num_item)
        rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

    elif name == 'jester':
        df_1 = pd.read_excel(os.path.join(loadpath, 'jester-data-1.xls'))
        df_2 = pd.read_excel(os.path.join(loadpath, 'jester-data-2.xls'))

        # ToDo
        rat_mat_1 = df_1.to_numpy()[:, 1:]  # 2000
        rat_mat_2 = df_2.to_numpy()[:, 1:]

        # Concat two datasets
        rat_mat = np.concatenate((rat_mat_1, rat_mat_2), axis=0)

        # Fill unrated cells with NaNs
        rat_mat[rat_mat == 99] = np.NaN

        # Train-test split
        rated_indices = np.where(~np.isnan(rat_mat))

        tr_va_idx, te_idx = train_test_split(range(len(rated_indices[0])), test_size=te_split, random_state=1)
        tr_idx, va_idx = train_test_split(tr_va_idx, test_size=va_split, random_state=2)

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

    elif name == 'monday_offers':
        df = pd.read_csv(os.path.join(loadpath, 'users_n_offering(binary).csv'))

        # Convert to numpy array
        rat_mat = df.to_numpy()[::5]

        # ToDo
        rat_mat -= 0.5

        # Train-test split
        rated_indices = np.where(~np.isnan(rat_mat))

        tr_va_idx, te_idx = train_test_split(range(len(rated_indices[0])), test_size=te_split, random_state=1)
        tr_idx, va_idx = train_test_split(tr_va_idx, test_size=va_split, random_state=2)

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

    else:
        raise Exception('%s is not valid dataset.' % name)

    if do_transpose:
        rat_mat_tr = rat_mat_tr.T
        rat_mat_va = rat_mat_va.T
        rat_mat_te = rat_mat_te.T
        num_user, num_item = num_item, num_user

    return rat_mat_tr, rat_mat_va, rat_mat_te, num_user, num_item


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


if __name__ == '__main__':
    load_path = os.path.join('..', '..', 'data', 'monday_offers')

    rating_mat_tr, rating_mat_va, rating_mat_te, n_u, n_i = load_dataset(load_path, 'monday_offers', 0.2, 0.1)
