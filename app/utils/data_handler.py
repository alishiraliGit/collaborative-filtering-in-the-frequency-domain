import csv
import numpy as np


def get_edge_list_from_file(file_path, file_name):
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
