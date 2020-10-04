import csv
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_dataset(loadpath, name, va_split=None, te_split=None, part=1):
    edges_tr, edges_va, edges_te, num_user, num_item = (None,)*5

    if name == 'ml-100k':
        edges_notmapped_tr_va = get_edge_list_from_file_ml100k(loadpath, 'u%d.base' % part)
        edges_notmapped_te = get_edge_list_from_file_ml100k(loadpath, 'u%d.test' % part)

        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped_tr_va + edges_notmapped_te)

        edges_tr_va = edges[:len(edges_notmapped_tr_va)]
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split)
        edges_te = edges[len(edges_notmapped_tr_va):]

    elif name == 'ml-1m':
        edges_notmapped = get_edge_list_from_file_ml1m(loadpath, 'ratings.dat')
        edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped)

        edges_tr_va, edges_te = train_test_split(edges, test_size=te_split)
        edges_tr, edges_va = train_test_split(edges_tr_va, test_size=va_split)

    rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
    rat_mat_va = get_rating_mat(edges_va, num_user, num_item)
    rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

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
    load_path = os.path.join('..', '..', 'data', 'ml-1m')

    edg = get_edge_list_from_file_ml1m(load_path, 'ratings.dat')

