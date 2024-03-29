import numpy as np
from numpy.random import default_rng
import os

from app.utils.data_handler import load_dataset
from app.models.logger import Logger
from app.models.matrixfactorization.mf import MatrixFactorization


def get_als_settings():
    sett = {}

    method = 'als'

    sett['method'] = method

    # Hidden dim
    sett['dim'] = 20

    # Weight init. std
    sett['w_init_std'] = 0.1

    # Regularization coefficients
    sett['l2_lambda'] = 1

    return sett


if __name__ == '__main__':
    print('---- Started! ----')

    # ------- Settings -------
    # Method settings
    settings = get_als_settings()

    # General
    do_plot = False
    do_save = False
    random_seed = 1
    rng = default_rng(random_seed)

    # Path
    load_path = os.path.join('..', 'data', 'coat')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'coat'
    part = np.nan

    # Cross-validation
    test_split = np.nan
    val_split = 0.1

    # Item-based or user-based
    do_transpose = False

    # Alternation
    n_alter = 20

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item, min_value, max_value = load_dataset(
        load_path, dataset_name,
        te_split=test_split, va_split=val_split,
        do_transpose=do_transpose,
        random_state=random_seed
    )

    print('Data loaded ...')

    # ------- Initialization -------
    #  Init. "w_u" and "w_i"
    w_u_0 = rng.normal(loc=0, scale=settings['w_init_std'], size=(settings['dim'], n_user))
    w_i_0 = rng.normal(loc=0, scale=settings['w_init_std'], size=(settings['dim'], n_item))

    # Init. MF
    mf = MatrixFactorization(w_u_0=w_u_0,
                             w_i_0=w_i_0,
                             l2_lambda=settings['l2_lambda'])

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    print('Init. done ...')

    # ------- Do the alternation -------
    mf.run(
        rat_mat_tr=rating_mat_tr,
        rat_mat_va=rating_mat_va,
        rat_mat_te=rating_mat_te,
        n_alt=n_alter,
        min_val=min_value,
        max_val=max_value,
        logger=logger
    )

    # ------- Print the best validated result -------
    best_iter = np.argmin(logger.rmse_va)
    print('--->\tbest it\trmse tr\trmse va\trmse te\n\t\t%.3f\t%.3f\t%.3f\t%.3f' %
          (best_iter + 1, logger.rmse_tr[best_iter], logger.rmse_va[best_iter], logger.rmse_te[best_iter]))

    # ------- Save the results -------
    if do_save:
        logger.save(ext={
            'w_u': mf.als.w_u,
            'w_i': mf.als.w_i,
            'b_u': mf.als.b_u,
            'b_i': mf.als.b_i,
            'rating_mat_tr': rating_mat_tr,
            'rating_mat_va': rating_mat_va,
            'rating_mat_te': rating_mat_te
        })
