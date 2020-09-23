import numpy as np
import os

from app.utils.data_handler import get_edge_list_from_file, map_ids, get_rating_mat
from app.models.vandermonde import Vandermonde
from core.alternate import Alternate
from app.models.clustering.kmeans import KMeans
from app.models.clustering.boosting import Boosting
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.least_square import LeastSquare
from app.models.logger import Logger


def estimate_l2_lambda(ratio, std_err, n_eq, std_est, n_est):
    return ratio * (std_err*n_eq) / (n_est*std_est)


def load_data(loadpath, filename_tr, filename_te):
    edges_notmapped_tr = get_edge_list_from_file(loadpath, filename_tr)
    edges_notmapped_te = get_edge_list_from_file(loadpath, filename_te)

    edges, map_u, map_i, num_user, num_item = map_ids(edges_notmapped_tr + edges_notmapped_te)
    edges_tr = edges[:len(edges_notmapped_tr)]
    edges_te = edges[len(edges_notmapped_tr):]

    rat_mat_tr = get_rating_mat(edges_tr, num_user, num_item)
    rat_mat_te = get_rating_mat(edges_te, num_user, num_item)

    return rat_mat_tr, rat_mat_te, num_user, num_item


def get_kmeans_approx_settings():
    sett = {}

    method = 'kmeans_approx'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4
    sett['l2_lambda_ratio'] = 0.01

    # Clustering settings
    sett['n_cluster'] = 5
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 0.1

    # Estimate regularization coefficients
    sett['l2_lambda'] = estimate_l2_lambda(ratio=sett['l2_lambda_ratio'],
                                           std_err=1,
                                           n_eq=8e4 / sett['n_cluster'],
                                           std_est=sett['cls_init_std'],
                                           n_est=(sett['m'] + 1) ** sett['dim_x'])
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_settings():
    sett = {}

    method = 'boosted_kmeans_approx'

    sett['method'] = method

    # Boosting settings
    sett['n_learner'] = 10
    sett['n_iter_cls'] = 3

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 0.2

    # Estimate regularization coefficients
    sett['l2_lambda'] = 0
    sett['l2_lambda_cls'] = 0

    return sett


def get_kmeans_ls_settings():
    sett = {}

    method = 'kmeans_ls'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4
    sett['l2_lambda_ratio'] = 0.01

    # Clustering settings
    sett['n_cluster'] = 5
    sett['cls_init_std'] = 0.1

    # Updater settings

    # Estimate regularization coefficients
    sett['l2_lambda'] = estimate_l2_lambda(ratio=sett['l2_lambda_ratio'],
                                           std_err=1,
                                           n_eq=8e4 / sett['n_cluster'],
                                           std_est=sett['cls_init_std'],
                                           n_est=(sett['m'] + 1) ** sett['dim_x'])
    sett['l2_lambda_cls'] = 0

    return sett


if __name__ == '__main__':
    # ------- Settings -------
    # Method settings
    settings = get_kmeans_ls_settings()

    # General settings
    do_plot = False

    # Load settings
    load_path = os.path.join('..', 'data', 'ml-100k')
    file_name_tr = 'u3.base'
    file_name_te = 'u3.test'
    save_path = os.path.join('..', 'results')

    # Dataset settings
    min_value = 1
    max_value = 5

    # Alternation settings
    n_alter = 40

    print(settings['l2_lambda'])

    # ------- Load data -------
    rating_mat_tr, rating_mat_te, n_user, n_item = load_data(load_path, file_name_tr, file_name_te)

    #
    # rating_mat_tr = rating_mat_tr[:, :n_item]
    # rating_mat_te = rating_mat_te[:, :n_item]
    #

    # ------- Initialization -------
    #  Init. Vandermonde
    vm = Vandermonde(dim_x=settings['dim_x'],
                     m=settings['m'],
                     l2_lambda=settings['l2_lambda'])

    #  Init. "x" and "a_c"
    x_mat_0 = np.random.random((settings['dim_x'], n_item))
    a_c_mat_0 = np.random.normal(loc=0, scale=settings['cls_init_std'], size=(vm.dim_a, settings['n_cluster']))

    #  Init. clustering and updater
    kmeans = KMeans(n_cluster=settings['n_cluster'],
                    a_c_mat_0=a_c_mat_0,
                    l2_lambda=settings['l2_lambda_cls'])

    # approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
    #                                gamma=settings['gamma'])
    ls_upd = LeastSquare(x_mat_0=x_mat_0)

    # Init. boosting
    # boost = Boosting(cls=kmeans,
    #                 n_learner=settings['n_learner'],
    #                 n_iter_cls=settings['n_iter_cls'])

    # Init. alternate
    alt = Alternate(cls=kmeans, upd=ls_upd)

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    # ------- Fit Vandermonde -------
    vm.fit()
    vm.transform(x_mat_0)

    # ------- Do the alternation -------
    alt.run(vm, rating_mat_tr, rating_mat_te, n_alter, min_value, max_value, logger=logger)

    # ------- Save the results -------
    logger.save()
