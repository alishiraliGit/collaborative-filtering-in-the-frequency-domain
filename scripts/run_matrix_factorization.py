import numpy as np
import os

from app.utils.data_handler import load_dataset
from app.models.vandermonde import Vandermonde
from core.alternate import Alternate
from app.models.clustering.one_user_one_cluster import OneUserOneCluster
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.least_square import LeastSquare
from app.models.updating.randomizer import Randomizer
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger


def estimate_l2_lambda(ratio, std_err, n_eq, std_est, n_est):
    return ratio * (std_err*n_eq) / (n_est*std_est)


def get_ls_settings():
    sett = {}

    method = 'ls'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 4

    # Clustering settings
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 0.1
    sett['max_nfev'] = 10

    # Estimate regularization coefficients
    sett['l2_lambda'] = 10

    return sett


if __name__ == '__main__':
    # ------- Settings -------
    # Method settings
    settings = get_ls_settings()

    # General
    do_plot = True

    # Path
    load_path = os.path.join('..', 'data', 'jester')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'jester'
    min_value = -10
    max_value = 10

    # Cross-validation
    test_split = 0.1
    val_split = 0.1

    # Item-based or user-based
    do_transpose = True

    # Alternation
    n_alter = 100

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item =\
        load_dataset(load_path, dataset_name, te_split=test_split, va_split=val_split, do_transpose=do_transpose)

    print('Data loaded ...')

    # ------- Initialization -------
    #  Init. Vandermonde
    vm = Vandermonde(dim_x=settings['dim_x'],
                     m=settings['m'],
                     l2_lambda=settings['l2_lambda'])

    #  Init. "x" and "a_c"
    x_mat_0 = np.random.random((settings['dim_x'], n_item))
    a_c_mat_0 = np.random.normal(loc=0, scale=settings['cls_init_std'], size=(vm.dim_a, n_user))

    #  Init. clustering
    one_user_one_cluster = OneUserOneCluster(n_cluster=n_user,
                                             a_c_mat_0=a_c_mat_0)

    # Init. updaters
    approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
                                    gamma=settings['gamma'])

    ls_upd = LeastSquare(x_mat_0=x_mat_0,
                         max_nfev=settings['max_nfev'])

    multi_upd = MultiUpdaterWrapper(upds=[approx_upd, ls_upd])

    # Init. alternate
    alt = Alternate(cls=one_user_one_cluster, upd=multi_upd)

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    print("Init. done ...")

    # ------- Fit Vandermonde -------
    vm.fit()
    vm.transform(x_mat_0)

    # ------- Do the alternation -------
    alt.run(vm, rating_mat_tr, rating_mat_va, n_alter, min_value, max_value,
            logger=logger,
            rating_mat_te=rating_mat_te)

    # ------- Save the results -------
    logger.save()
