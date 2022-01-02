import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import os

from app.utils.data_handler import load_dataset
from app.models.vandermonde import Vandermonde, VandermondeType, RegularizationType
from core.alternate import Alternate
from app.models.clustering.one_user_one_cluster import OneUserOneCluster, OneUserOneClusterBiasCorrected
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.bfgs import BFGS
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger

rng = default_rng(11)


def get_approx_bfgs_settings():
    sett = {}

    method = 'bfgs'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 1
    sett['m'] = 2
    sett['vm_type'] = VandermondeType.COS_MULT
    # sett['reg_type'] = RegularizationType.L2
    # sett['reg_params'] = {'l2_lambda': 1, 'exclude_zero_freq': True}
    sett['reg_type'] = RegularizationType.MAX_SNR
    sett['reg_params'] = {'bound': (0, 1), 'exclude_zero_freq': True}
    # sett['reg_type'] = RegularizationType.POW
    # sett['reg_params'] = {'l2_lambda': 0.5, 'z': 1}

    # Clustering settings
    sett['n_iter_alpha'] = 1
    sett['estimate_sigma_n'] = True
    sett['sigma_n'] = 0
    sett['min_alpha'] = 0

    # Updater settings
    sett['gamma'] = 1  # default: 1
    sett['max_iter'] = 5  # default: 5

    return sett


if __name__ == '__main__':
    print('---- Started! ----')

    # ------- Settings -------
    # Method settings
    settings = get_approx_bfgs_settings()

    print(settings)

    # General
    do_plot = True
    do_save = False

    # Path
    load_path = os.path.join('..', 'data', 'coat')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'coat'
    min_value = 1
    max_value = 5

    # Cross-validation
    val_split = 0.1

    # Item-based (True) or user-based
    do_transpose = False

    # Alternation
    n_alter = 30

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item =\
        load_dataset(load_path, dataset_name, va_split=val_split, do_transpose=do_transpose, random_state=1)

    print('Data loaded ...')

    # ------- Initialization -------
    #  Init. VM
    vm = Vandermonde.get_instance(
        dim_x=settings['dim_x'],
        m=settings['m'],
        vm_type=settings['vm_type'],
        reg_type=settings['reg_type'],
        reg_params=settings['reg_params']
    )

    #  Init. "x"
    x_mat_0 = rng.random((settings['dim_x'], n_item))

    #  Init. clustering
    one_user_one_cluster = OneUserOneClusterBiasCorrected(
        n_iter=settings['n_iter_alpha'],
        estimate_sigma_n=settings['estimate_sigma_n'],
        sigma_n=settings['sigma_n'],
        min_alpha=settings['min_alpha']
    )
    # one_user_one_cluster = OneUserOneCluster()

    # Init. updaters
    approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
                                    gamma=settings['gamma'])

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
                    rating_mat_te=rating_mat_te,
                    logger=logger,
                    verbose=True)

    # ------- Print the best validated result -------
    best_iter = np.argmin(logger.rmse_va)
    print('---> best iter: %d, rmse train: %.3f, rmse val: %.3f rmse test: %.3f' %
          (int(best_iter), logger.rmse_tr[best_iter], logger.rmse_va[best_iter], logger.rmse_te[best_iter]))

    # ------- Plot predictions -------
    rating_mat_pr = vm.predict(a_mat)
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    mask_tr = ~np.isnan(rating_mat_tr)
    plt.plot(rating_mat_tr[mask_tr], rating_mat_pr[mask_tr], 'k*')

    plt.subplot(1, 2, 2)
    mask_te = ~np.isnan(rating_mat_te)
    plt.plot(rating_mat_te[mask_te], rating_mat_pr[mask_te], 'k*')

    # ------- Save the results -------
    if do_save:
        logger.save(ext={
            'x_mat': alt.upd.x_mat,
            'a_mat': a_mat,
            'rating_mat_tr': rating_mat_tr,
            'rating_mat_va': rating_mat_va,
            'rating_mat_te': rating_mat_te,
        })
