import numpy as np
import os

from app.utils.data_handler import get_edge_list_from_file, map_ids, get_rating_mat
from app.models.vandermonde import Vandermonde
from core.alternate import Alternate
from app.models.clustering.kmeans import KMeans
# from app.models.clustering.boosting import Boosting
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.logger import Logger


def estimate_l2_lambda(ratio, std_err, n_eq, std_est, n_est):
    return ratio * (std_err*n_eq) / (n_est*1)


if __name__ == '__main__':
    # ------- Settings -------
    settings = {}

    # General settings
    do_plot = False
    settings['method'] = 'kmeans_approx'

    # Load settings
    load_path = os.path.join('..', 'data', 'ml-100k')
    file_name_tr = 'u3.base'
    file_name_te = 'u3.test'
    save_path = os.path.join('..', 'results')

    # Vandermonde settings
    settings['dim_x'] = 3
    settings['m'] = 8
    settings['l2_lambda_ratio'] = 0.01

    # Clustering settings
    settings['n_cluster'] = 5
    settings['cls_init_std'] = 0.1

    # Updater settings
    settings['gamma'] = 0.1

    # Alternation settings
    n_alter = 40

    # Estimate regularization coefficients
    settings['l2_lambda'] = estimate_l2_lambda(ratio=settings['l2_lambda_ratio'],
                                               std_err=1,
                                               n_eq=8e4/settings['n_cluster'],
                                               std_est=settings['cls_init_std'],
                                               n_est=(settings['m'] + 1)**settings['dim_x'])
    settings['l2_lambda_cls'] = 0

    print(settings['l2_lambda'])

    # ------- Load data -------
    edges_notmapped_tr = get_edge_list_from_file(load_path, file_name_tr)
    edges_notmapped_te = get_edge_list_from_file(load_path, file_name_te)

    edges, map_u, map_i, n_user, n_item = map_ids(edges_notmapped_tr + edges_notmapped_te)
    edges_tr = edges[:len(edges_notmapped_tr)]
    edges_te = edges[len(edges_notmapped_tr):]

    rating_mat_tr = get_rating_mat(edges_tr, n_user, n_item)
    rating_mat_te = get_rating_mat(edges_te, n_user, n_item)

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

    approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
                                    gamma=settings['gamma'])

    # Init. alternate
    alt = Alternate(cls=kmeans, upd=approx_upd)

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    # ------- Fit Vandermonde -------
    vm.fit()
    vm.transform(x_mat_0)

    # ------- Do the alternation -------
    alt.run(vm, rating_mat_tr, rating_mat_te, n_alter, logger=logger)

    # ------- Save the results -------
    logger.save()
