import numpy as np
from numpy.random import default_rng
import os

from app.utils.data_handler import load_dataset
from app.models.vandermonde import Vandermonde, VandermondeType
from core.alternate import Alternate
from app.models.clustering.kmeans import KMeans
from app.models.clustering.boosting import Boosting
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.bfgs import BFGS
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger


def estimate_l2_lambda(ratio, std_err, n_eq, std_est, n_est):
    return ratio * (std_err*n_eq) / (n_est*std_est)


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
    sett['m'] = 5
    sett['l2_lambda_ratio'] = 0.01

    # Clustering settings
    sett['n_cluster'] = 5
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['max_nfev'] = 5

    # Estimate regularization coefficients
    sett['l2_lambda'] = estimate_l2_lambda(ratio=sett['l2_lambda_ratio'],
                                           std_err=1,
                                           n_eq=8e4 / sett['n_cluster'],
                                           std_est=sett['cls_init_std'],
                                           n_est=(sett['m'] + 1) ** sett['dim_x'])
    sett['l2_lambda_cls'] = 0

    return sett


def get_kmeans_approx_ls_settings():
    sett = {}

    method = 'kmeans_approx_ls'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 5
    sett['l2_lambda_ratio'] = 0.01

    # Clustering settings
    sett['n_cluster'] = 5
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 0.3
    sett['max_nfev'] = 5

    # Estimate regularization coefficients
    sett['l2_lambda'] = estimate_l2_lambda(ratio=sett['l2_lambda_ratio'],
                                           std_err=1,
                                           n_eq=8e4 / sett['n_cluster'],
                                           std_est=sett['cls_init_std'],
                                           n_est=(sett['m'] + 1) ** sett['dim_x'])
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_ls_settings():
    sett = {}

    method = 'boosted_kmeans_approx_ls'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 5

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = 4
    sett['n_iter_cls'] = 3

    # Updater settings
    sett['gamma'] = 0.1
    sett['max_nfev'] = 4

    # Estimate regularization coefficients
    sett['l2_lambda'] = 0
    sett['l2_lambda_cls'] = 0

    return sett


def get_kmeans_approx_bfgs_settings():
    sett = {}

    method = 'kmeans_approx_bfgs'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4
    sett['vm_type'] = VandermondeType.COS_MULT

    # Clustering settings
    sett['n_cluster'] = 7
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 1
    sett['max_iter_bfgs'] = 5

    # Regularization coefficients
    sett['l2_lambda'] = 100
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_bfgs_settings():
    sett = {}

    method = 'boosted_kmeans_approx_bfgs'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 2
    sett['vm_type'] = VandermondeType.COS_MULT

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = 13
    sett['n_iter_cls'] = 3

    # Updater settings
    sett['gamma'] = 1
    sett['max_iter_bfgs'] = 5

    # Regularization coefficients
    sett['l2_lambda'] = 10
    sett['l2_lambda_cls'] = 0

    return sett


if __name__ == '__main__':
    # ------- Settings -------
    # Method
    settings = get_boosted_kmeans_approx_bfgs_settings()

    print(settings)

    # General
    do_plot = True

    # Path
    load_path = os.path.join('..', 'data', 'monday_offers')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'monday_offers'
    min_value = 0
    max_value = 1

    # Cross-validation
    test_split = 0.1
    val_split = 0.1 / (1 - test_split)

    # Item-based (True) or user-based
    do_transpose = False

    # Alternation
    n_alter = 10

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(load_path, dataset_name, te_split=test_split, va_split=val_split, do_transpose=do_transpose)

    print('Data loaded ...')

    # ------- Initialization -------
    rng = default_rng(1)

    #  Init. Vandermonde
    vm = Vandermonde.get_instance(dim_x=settings['dim_x'],
                                  m=settings['m'],
                                  l2_lambda=settings['l2_lambda'],
                                  vm_type=settings['vm_type'])

    #  Init. "x" and "a_c"
    x_mat_0 = rng.random((settings['dim_x'], n_item))
    a_c_mat_0 = rng.normal(loc=0, scale=settings['cls_init_std'], size=(vm.dim_a, settings['n_cluster']))

    #  Init. clustering
    kmeans = KMeans(n_cluster=settings['n_cluster'],
                    a_c_mat_0=a_c_mat_0,
                    l2_lambda=settings['l2_lambda_cls'])

    boost = Boosting(cls=kmeans,
                     n_learner=settings['n_learner'],
                     n_iter_cls=settings['n_iter_cls'])

    # Init. updaters
    approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
                                    gamma=settings['gamma'])

    bfgs_upd = BFGS(x_mat_0=x_mat_0,
                    max_iter=settings['max_iter_bfgs'])

    multi_upd = MultiUpdaterWrapper(upds=[approx_upd, bfgs_upd])

    # Init. alternate
    alt = Alternate(cls=boost, upd=multi_upd)

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    print('Init. done ...')

    # ------- Fit Vandermonde -------
    vm.fit()
    vm.transform(x_mat_0)

    # ------- Do the alternation -------
    a_mat = alt.run(vm, rating_mat_tr, rating_mat_va, n_alter, min_value, max_value,
                    logger=logger,
                    rating_mat_te=rating_mat_te)

    # ------- Print the best validated result -------
    best_iter = np.argmin(logger.rmse_va)
    print('---> best iter: %d, rmse train: %.3f, rmse val: %.3f rmse test: %.3f' %
          (int(best_iter), logger.rmse_tr[best_iter], logger.rmse_va[best_iter], logger.rmse_te[best_iter]))

    # ------- Save the results -------
    logger.save(ext={
        'x_mat': alt.upd.x_mat,
        'a_mat': a_mat,
        'rating_mat_tr': rating_mat_tr,
        'rating_mat_va': rating_mat_va,
        'rating_mat_te': rating_mat_te,
    })
