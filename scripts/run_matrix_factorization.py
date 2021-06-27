import numpy as np
import os

from app.utils.data_handler import load_dataset
from app.models.logger import Logger
from app.models.matrixfactorization.mf import MatrixFactorization


def get_als_settings():
    sett = {}

    method = 'als'

    sett['method'] = method

    # Hidden dim
    sett['dim'] = 3

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
    do_plot = True

    # Path
    load_path = os.path.join('..', 'data', 'ml-100k')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'ml-100k'
    min_value = 1
    max_value = 5
    part = 5

    # Cross-validation
    test_split = 0.1
    val_split = 0.01/(1 - test_split)

    # Item-based or user-based
    do_transpose = False

    # Alternation
    n_alter = 20

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item =\
        load_dataset(load_path, dataset_name, te_split=test_split, va_split=val_split, do_transpose=do_transpose, part=part)

    print('Data loaded ...')

    # ------- Initialization -------
    #  Init. "w_u" and "w_i"
    w_u_0 = np.random.normal(loc=0, scale=settings['w_init_std'], size=(settings['dim'], n_user))
    w_i_0 = np.random.normal(loc=0, scale=settings['w_init_std'], size=(settings['dim'], n_item))

    # Init. MF
    mf = MatrixFactorization(w_u_0=w_u_0,
                             w_i_0=w_i_0,
                             l2_lambda=settings['l2_lambda'])

    # Init. logger
    logger = Logger(settings=settings, save_path=save_path, do_plot=do_plot)

    print("Init. done ...")

    # ------- Do the alternation -------
    mf.run(rat_mat_tr=rating_mat_tr,
           rat_mat_va=rating_mat_va,
           rat_mat_te=rating_mat_te,
           n_alt=n_alter,
           min_val=min_value,
           max_val=max_value,
           logger=logger)

    # ------- Save the results -------
    logger.save(ext={
        'w_u': mf.als.w_u,
        'w_i': mf.als.w_i,
        'b_u': mf.als.b_u,
        'b_i': mf.als.b_i,
        'rating_mat_tr': rating_mat_tr,
        'rating_mat_va': rating_mat_va,
        'rating_mat_te': rating_mat_te
    })
