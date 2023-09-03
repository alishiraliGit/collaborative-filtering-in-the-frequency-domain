import numpy as np
from numpy.random import default_rng
import os
import torch.optim as optim

from app.utils.data_handler import load_dataset, load_propensity_scores
from app.utils.pytorch_optim_utils import OptimizerSpec, ConstantSchedule
from app.models.vandermonde import Vandermonde, VandermondeType, RegularizationType
from core.alternate import Alternate
from app.models.clustering.kmeans import KMeans, KMeansBiasCorrected
from app.models.clustering.boosting import Boosting, BoostingBiasCorrected
from app.models.updating.approximate_updater import ApproximateUpdater, ApproximateExpandedUpdater
from app.models.updating.bfgs import BFGS
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger


def get_kmeans_approx_bfgs_settings():
    sett = {}

    method = 'kmeans_approx_bfgs'

    sett['method'] = method

    # --- Vandermonde settings ---
    sett['dim_x'] = 3
    sett['m'] = 4
    sett['vm_type'] = VandermondeType.COS_MULT
    # Reg.
    sett['reg_type'] = RegularizationType.L2
    sett['reg_params'] = {'l2_lambda': 1, 'exclude_zero_freq': True}

    # --- Clustering settings ---
    sett['clust_method'] = 'k-means'
    sett['n_cluster'] = 7
    sett['n_iter_clust'] = 5
    sett['std_init_clust'] = 1e-2

    # --- Updater(s) settings ---
    # Approx.
    sett['gamma'] = 1
    # BFGS
    sett['max_iter_bfgs'] = 5

    return sett


def get_kmeans_approx_bfgs_bc_settings():
    sett = {}

    method = 'kmeans_approx_bfgs_bc'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 3
    sett['vm_type'] = VandermondeType.COS_MULT
    sett['reg_type'] = RegularizationType.L2
    sett['reg_params'] = {'l2_lambda': 1, 'exclude_zero_freq': True}
    # sett['reg_type'] = RegularizationType.POST_MAX_SNR
    # sett['reg_params'] = {'bound': (0, 1), 'exclude_zero_freq': True}
    # sett['reg_type'] = RegularizationType.POW
    # sett['reg_params'] = {'l2_lambda': 1, 'z': 1, 'exclude_zero_freq': True}  # z=1.2

    # Clustering settings
    sett['clust_method'] = 'k-means-bc'
    sett['n_cluster'] = 10
    sett['n_iter_clust'] = 1
    sett['std_init_clust'] = 0.01  # default: 1e-2

    # Bias correction
    sett['n_iter_alpha'] = 1
    sett['estimate_sigma_n'] = False
    sett['sigma_n'] = 0.6  # 0.6
    sett['min_alpha'] = 0

    # Updater settings
    sett['gamma'] = 1
    sett['max_iter_bfgs'] = 5

    # MNAR
    sett['obtain_tt'] = True  # True if at test time the predictor needs to be updated

    return sett


def get_boosted_kmeans_approx_adam_settings():
    sett = {}

    method = 'boosted_kmeans_approx_bfgs'

    sett['method'] = method

    # --- Vandermonde settings ---
    sett['dim_x'] = 3
    sett['m'] = 2
    sett['vm_type'] = VandermondeType.COS_MULT
    # Reg.
    sett['reg_type'] = RegularizationType.L2
    sett['reg_params'] = {'l2_lambda': 30, 'exclude_zero_freq': True}

    # --- Clustering settings ---
    sett['clust_method'] = 'boosting'
    sett['n_learner'] = 10
    sett['n_cluster'] = 2  # Default: 2
    sett['n_iter_clust'] = 5  # Default: 5
    sett['std_init_clust'] = 1e-2  # Default: 1e-2

    # --- Updater(s) settings ---
    # Approx.
    sett['gamma'] = 1  # Default: 1
    # Adam
    sett['n_iter'] = 10
    sett['optimizer_spec'] = OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=ConstantSchedule(1e-1).value,
    )

    return sett


def get_boosted_kmeans_approx_bfgs_settings():
    sett = {}

    method = 'boosted_kmeans_approx_bfgs'

    sett['method'] = method

    # --- Vandermonde settings ---
    sett['dim_x'] = 2
    sett['m'] = 4
    sett['vm_type'] = VandermondeType.COS_MULT
    # Reg.
    sett['reg_type'] = RegularizationType.L2
    sett['reg_params'] = {'l2_lambda': 10, 'exclude_zero_freq': True}

    # --- Clustering settings ---
    sett['clust_method'] = 'boosting'
    sett['n_learner'] = 10
    sett['n_cluster'] = 2  # Default: 2
    sett['n_iter_clust'] = 5  # Default: 5
    sett['std_init_clust'] = 1e-2  # Default: 1e-2

    # --- Updater(s) settings ---
    # Approx.
    sett['gamma'] = 1  # Default: 1
    # BFGS
    sett['max_iter_bfgs'] = 5  # Default: 5

    return sett


def get_boosted_kmeans_approx_bfgs_bc_settings():
    sett = {}

    method = 'boosted_kmeans_approx_bfgs_bc'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 3
    sett['vm_type'] = VandermondeType.COS_MULT
    # sett['reg_type'] = RegularizationType.L2
    # sett['reg_params'] = {'l2_lambda': 10, 'exclude_zero_freq': True}
    sett['reg_type'] = RegularizationType.POW
    sett['reg_params'] = {'l2_lambda': 10, 'z': 1.2, 'exclude_zero_freq': True}  # z=1.3, iter=4

    # Clustering settings
    sett['clust_method'] = 'boosting-bc'
    sett['n_learner'] = 10
    sett['n_cluster'] = 2  # Default: 2
    sett['n_iter_clust'] = 5  # Default: 5
    sett['std_init_clust'] = 1e-2  # Default: 1e-2

    # Bias correction
    sett['n_iter_alpha'] = 1
    sett['estimate_sigma_n'] = True
    sett['sigma_n'] = np.nan  # 0.6 or 0.8
    sett['min_alpha'] = 0

    # Updater settings
    sett['gamma'] = 1
    sett['max_iter_bfgs'] = 5

    # MNAR
    sett['obtain_tt'] = True  # True if at test time the predictor needs to be updated

    return sett


def clust_selector(sett, a_clust_mat_0):
    if sett['clust_method'] == 'k-means':
        return KMeans(
            n_cluster=sett['n_cluster'],
            n_iter=sett['n_iter_clust'],
            a_c_mat_0=a_clust_mat_0
        )
    elif sett['clust_method'] == 'k-means-bc':
        return KMeansBiasCorrected(
            n_cluster=sett['n_cluster'],
            n_iter=sett['n_iter_clust'],
            a_c_mat_0=a_clust_mat_0,
            n_iter_alpha=sett['n_iter_alpha'],
            estimate_sigma_n=sett['estimate_sigma_n'],
            sigma_n=sett['sigma_n'],
            min_alpha=sett['min_alpha']
        )
    elif sett['clust_method'] == 'boosting':
        kmeans = KMeans(
            n_cluster=sett['n_cluster'],
            n_iter=sett['n_iter_clust'],
            a_c_mat_0=a_clust_mat_0
        )
        return Boosting(
           clust=kmeans,
           n_learner=sett['n_learner']
        )
    elif sett['clust_method'] == 'boosting-bc':
        kmeans_bc = KMeansBiasCorrected(
            n_cluster=sett['n_cluster'],
            n_iter=sett['n_iter_clust'],
            a_c_mat_0=a_clust_mat_0,
            n_iter_alpha=sett['n_iter_alpha'],
            estimate_sigma_n=sett['estimate_sigma_n'],
            sigma_n=sett['sigma_n'],
            min_alpha=sett['min_alpha']
        )
        return BoostingBiasCorrected(
           clust=kmeans_bc,
           n_learner=sett['n_learner']
        )


if __name__ == '__main__':
    # ----- Settings -----
    # Method
    settings = get_boosted_kmeans_approx_bfgs_settings()
    print(settings)

    # General
    do_plot = True
    do_save = True
    random_seed = 1
    rng = default_rng(random_seed)

    # Item-based (True) or user-based
    do_transpose = True

    # Alternation
    n_alter = 10

    # Dataset
    dataset_name = 'yahoo-r3'
    dataset_part = np.nan

    # Cross-validation
    test_split = 0.1
    val_split = 0.05

    # Path
    load_path = os.path.join('..', 'data', 'yahoo-r3')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # IPS
    do_ips = False

    # MNAR
    obtain_tt = settings.get('obtain_tt', False)

    # ----- Load data -----
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item, min_value, max_value = load_dataset(
        load_path, dataset_name, part=dataset_part,
        va_split=val_split, te_split=test_split,
        do_transpose=do_transpose,
        random_state=random_seed
    )

    print('Data loaded ...')

    # ----- Load IPS -----
    if do_ips:
        prop_mat = load_propensity_scores(load_path, dataset_name, do_transpose=do_transpose)
        prop_mat[np.isnan(rating_mat_tr)] = np.nan
        # Normalize propensity scores
        prop_mat /= np.mean((~np.isnan(rating_mat_tr))*1)
    else:
        prop_mat = rating_mat_tr.copy()
        prop_mat[~np.isnan(rating_mat_tr)] = 1

    # ----- Initialization -----
    #  Init. Vandermonde
    vm = Vandermonde.get_instance(
        dim_x=settings['dim_x'],
        m=settings['m'],
        vm_type=settings['vm_type'],
        reg_type=settings['reg_type'],
        reg_params=settings['reg_params']
    )

    #  Init. "x" and "a_c"
    x_mat_0 = rng.random((settings['dim_x'], n_item))
    a_c_mat_0 = rng.normal(loc=0, scale=settings['std_init_clust'], size=(vm.dim_a, settings['n_cluster']))

    #  Init. clustering
    clustering = clust_selector(settings, a_c_mat_0)

    # Init. updaters
    approx_upd = ApproximateUpdater(x_mat_0=x_mat_0,
                                    gamma=settings['gamma'])

    bfgs_upd = BFGS(x_mat_0=x_mat_0,
                    max_iter=settings['max_iter_bfgs'])

    multi_upd = MultiUpdaterWrapper(upds=[approx_upd, bfgs_upd])

    # Init. alternate
    alt = Alternate(clust=clustering, upd=multi_upd)

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    print('Init. done ...')

    # ----- Fit Vandermonde -----
    vm.fit()
    vm.transform(x_mat_0)

    # ----- Do the alternation -----
    a_mat_opt, x_mat_opt = alt.run(
        vm, rating_mat_tr, rating_mat_va, n_alter, min_value, max_value,
        logger=logger,
        rating_mat_te=rating_mat_te,
        propensity_mat=prop_mat,
        obtain_tt=obtain_tt,
        verbose=True
    )

    # ----- Print the best validated result -----
    best_iter = np.argmin(logger.rmse_va)
    print('--->\tbest it\trmse tr\trmse va\trmse te\n\t\t%.3f\t%.3f\t%.3f\t%.3f' %
          (best_iter + 1, logger.rmse_tr[best_iter], logger.rmse_va[best_iter], logger.rmse_te[best_iter]))

    # ----- Save the results -----
    if do_save:
        logger.save(ext={
            'x_mat': x_mat_opt,
            'a_mat': a_mat_opt,
            'rating_mat_tr': rating_mat_tr,
            'rating_mat_va': rating_mat_va,
            'rating_mat_te': rating_mat_te,
        })
