import numpy as np
from numpy.random import default_rng
import os

from app.utils.data_handler import load_dataset
from app.models.vandermonde import Vandermonde, VandermondeType
from core.alternate import Alternate
from app.models.clustering.one_user_one_cluster import OneUserOneCluster
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.least_square import LeastSquare
from app.models.updating.bfgs import BFGS
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger


def estimate_l2_lambda(ratio, var_err, n_sample, var_a, dim_a):
    return ratio * (n_sample*var_err) / (dim_a*var_a)


def get_bfgs_settings():
    sett = {}

    method = 'bfgs'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 6
    sett['vm_type'] = VandermondeType.COS

    # Clustering settings
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 1  # default: 1
    sett['max_iter'] = 5  # default: 5

    # Estimate regularization coefficients
    sett['l2_lambda'] = 1000  # default: 1000

    return sett


if __name__ == '__main__':
    print('---- Started! ----')

    # ------- Settings -------
    # Method settings
    settings = get_bfgs_settings()

    print(settings)

    # General
    do_plot = True

    # Init.
    do_init_from_file = False
    init_filename = 'result-methodals-dim2-w_init_std1e-01-l2_lambda1-2021-02-06 12-06-36'
    init_load_path = os.path.join('..', 'results')

    # Path
    load_path = os.path.join('..', 'data', 'monday_offers')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'monday_offers'
    min_value = -0.5
    max_value = 0.5

    # Cross-validation
    test_split = 0.1
    val_split = 0.1/(1 - test_split)

    # Item-based (True) or user-based
    do_transpose = True

    # Alternation
    n_alter = 8

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item =\
        load_dataset(load_path, dataset_name, te_split=test_split, va_split=val_split, do_transpose=do_transpose)

    print('Data loaded ...')

    # ------- Initialization -------
    rng = default_rng(1)
    #  Init. Vandermonde
    vm = Vandermonde.get_instance(
        dim_x=settings['dim_x'],
        m=settings['m'],
        l2_lambda=settings['l2_lambda'],
        vm_type=settings['vm_type']
    )

    #  Init. "x" and "a_c"
    if do_init_from_file:
        ext_dic = Logger.load(init_load_path, init_filename, load_ext=True)
        x_mat_0 = ext_dic['w_i']

        min_x_0 = np.min(x_mat_0)
        max_x_0 = np.max(x_mat_0)

        x_mat_0 = (x_mat_0 - min_x_0) / (max_x_0 - min_x_0)

    else:
        x_mat_0 = rng.random((settings['dim_x'], n_item), )

    a_c_mat_0 = rng.normal(loc=0, scale=settings['cls_init_std'], size=(vm.dim_a, n_user))

    #  Init. clustering
    one_user_one_cluster = OneUserOneCluster(n_cluster=n_user,
                                             a_c_mat_0=a_c_mat_0)

    # Init. updaters
    approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
                                    gamma=settings['gamma'])

    # ls_upd = LeastSquare(x_mat_0=x_mat_0,
    #                     max_nfev=settings['max_nfev'])

    bfgs_upd = BFGS(x_mat_0=x_mat_0,
                    max_iter=settings['max_iter'])

    multi_upd = MultiUpdaterWrapper(upds=[approx_upd, bfgs_upd])

    # Init. alternate
    alt = Alternate(cls=one_user_one_cluster, upd=multi_upd)

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    print("Init. done ...")

    # ------- Fit Vandermonde -------
    vm.fit()
    vm.transform(x_mat_0)

    # ------- Do the alternation -------
    a_mat = alt.run(vm, rating_mat_tr, rating_mat_va, n_alter, min_value, max_value,
                    logger=logger,
                    rating_mat_te=rating_mat_te)

    # ------- Save the results -------
    logger.save(ext={
        'x_mat': alt.upd.x_mat,
        'a_mat': a_mat,
        'rating_mat_tr': rating_mat_tr,
        'rating_mat_va': rating_mat_va,
        'rating_mat_te': rating_mat_te,
    })
