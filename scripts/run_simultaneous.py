import numpy as np
from numpy.random import default_rng
import os
from sklearn.linear_model import LinearRegression

from app.utils.data_handler import load_dataset
from app.models.vandermonde import Vandermonde, VandermondeType
from core.alternate import Alternate
from app.models.clustering.kmeans import KMeans
from app.models.clustering.boosting import Boosting
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.bfgs import BFGS
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger


def calc_rmse(y_predict, y_g_truth, min_val, max_val):
    y_predict_copy = y_predict.copy()
    y_predict_copy[y_predict_copy > max_val] = max_val
    y_predict_copy[y_predict_copy < min_val] = min_val

    return np.sqrt(np.mean((y_predict_copy - y_g_truth)**2))


def calc_row_mean_matrix(mat):
    return np.tile(np.nanmean(mat, axis=1).reshape((-1, 1)), reps=(1, mat.shape[1]))


def fit_reg(v_m_u, a_m_u, mean_u, v_m_i, a_m_i, mean_i, rat_mat):
    rat_mat_pr_u = v_m_u.predict(a_m_u) + mean_u
    rat_mat_pr_i = v_m_i.predict(a_m_i).T + mean_i.T

    rat_mat_pr_u[np.isnan(mean_u)] = rat_mat_pr_i[np.isnan(mean_u)]
    rat_mat_pr_i[np.isnan(mean_i.T)] = rat_mat_pr_u[np.isnan(mean_i.T)]

    mask_not_nan = ~np.isnan(rat_mat)

    x_pr = np.concatenate((rat_mat_pr_u[mask_not_nan].reshape((-1, 1)),
                           rat_mat_pr_i[mask_not_nan].reshape((-1, 1))), axis=1)

    y = rat_mat[mask_not_nan]

    reg_mdl = LinearRegression()

    reg_mdl.fit(x_pr, y)

    return reg_mdl, x_pr, y


def estimate_l2_lambda(ratio, std_err, n_eq, std_est, n_est):
    return ratio * (std_err*n_eq) / (n_est*std_est)


def get_kmeans_approx_settings():
    sett = {}

    method = 'kmeans_approx'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4

    # Clustering settings
    sett['n_cluster'] = 5
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 0.1

    # Estimate regularization coefficients
    sett['l2_lambda'] = 1
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_settings():
    sett = {}

    method = 'kmeans_approx'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = 4
    sett['n_iter_cls'] = 3

    # Updater settings
    sett['gamma'] = 1

    # Estimate regularization coefficients
    sett['l2_lambda'] = 0
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_ls_settings():
    sett = {}

    method = 'boosted_kmeans_approx_ls'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = 4
    sett['n_iter_cls'] = 3

    # Updater settings
    sett['gamma'] = 1
    sett['max_nfev'] = 3

    # Estimate regularization coefficients
    sett['l2_lambda'] = 0
    sett['l2_lambda_cls'] = 0

    return sett


def get_kmeans_approx_bfgs_settings():
    sett = {}

    method = 'kmeans_approx_bfgs'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 3
    sett['vm_type'] = VandermondeType.COS_MULT

    # Clustering settings
    sett['n_cluster'] = 3
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 1
    sett['max_iter_bfgs'] = 5

    # Regularization coefficients
    sett['l2_lambda'] = 10
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_bfgs_settings():
    sett = {}

    method = 'kmeans_boosted_approx_bfgs'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4
    sett['vm_type'] = VandermondeType.COS_MULT

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = 6
    sett['n_iter_cls'] = 3

    # Updater settings
    sett['gamma'] = 1
    sett['max_iter_bfgs'] = 5

    # Regularization coefficients
    sett['l2_lambda'] = 0
    sett['l2_lambda_cls'] = 0

    return sett


if __name__ == '__main__':
    # ------- Settings -------
    # Method
    settings_u = get_boosted_kmeans_approx_bfgs_settings()
    settings_u['method'] += '_user_based'
    settings_i = get_boosted_kmeans_approx_bfgs_settings()
    settings_i['method'] += '_item_based'

    print(settings_u)

    # General
    do_plot = False

    # Path
    load_path = os.path.join('..', 'data', 'ml-1m')

    save_path = os.path.join('..', 'results')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_name = 'ml-1m'
    min_value = 1
    max_value = 5

    # Cross-validation
    test_split = 0.05
    val_split = 0.05/(1 - test_split)

    # Alternation settings
    n_iter = 5

    # ------- Load data -------
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(load_path, dataset_name, te_split=test_split, va_split=val_split, do_transpose=False)

    print('Data loaded ...')

    # ------- Initialization for the big loop -------
    rng = default_rng(1)

    # Init. logger
    logger_u = Logger(settings=settings_u, save_path=save_path, do_plot=do_plot)
    logger_i = Logger(settings=settings_i, save_path=save_path, do_plot=do_plot)

    logger = Logger(settings={'method': settings_u['method']}, save_path=save_path, do_plot=True)

    # Init. for big loop
    rating_mat_tr_res = rating_mat_tr.copy()
    rating_mat_va_res = rating_mat_va.copy()
    rating_mat_te_res = rating_mat_te.copy()

    mean_mat_u_res = calc_row_mean_matrix(rating_mat_tr_res)
    mean_mat_i_res = calc_row_mean_matrix(rating_mat_tr_res.T)

    rating_mat_tr_u_res = rating_mat_tr_res - mean_mat_u_res
    rating_mat_tr_i_res = rating_mat_tr_res.T - mean_mat_i_res
    rating_mat_va_u_res = rating_mat_va_res - mean_mat_u_res
    rating_mat_va_i_res = rating_mat_va_res.T - mean_mat_i_res
    rating_mat_te_u_res = rating_mat_te_res - mean_mat_u_res
    rating_mat_te_i_res = rating_mat_te_res.T - mean_mat_i_res

    y_tr_pr = 0
    y_va_pr = 0
    y_te_pr = 0

    mask_tr = ~np.isnan(rating_mat_tr)
    mask_va = ~np.isnan(rating_mat_va)
    mask_te = ~np.isnan(rating_mat_te)

    for it in range(n_iter):
        # ToDo
        if it == 0:
            n_alter = 8
        else:
            n_alter = 3
        # ------- Initialization -------
        #  Init. Vandermonde
        vm_u = Vandermonde.get_instance(dim_x=settings_u['dim_x'],
                                        m=settings_u['m'],
                                        l2_lambda=settings_u['l2_lambda'],
                                        vm_type=settings_u['vm_type'])

        vm_i = Vandermonde.get_instance(dim_x=settings_i['dim_x'],
                                        m=settings_i['m'],
                                        l2_lambda=settings_i['l2_lambda'],
                                        vm_type=settings_i['vm_type'])

        #  Init. "x" and "a_c"
        x_mat_0_u = rng.random((settings_u['dim_x'], n_item))
        x_mat_0_i = rng.random((settings_i['dim_x'], n_user))

        a_c_mat_0_u = rng.normal(loc=0, scale=settings_u['cls_init_std'], size=(vm_u.dim_a, settings_u['n_cluster']))
        a_c_mat_0_i = rng.normal(loc=0, scale=settings_i['cls_init_std'], size=(vm_i.dim_a, settings_i['n_cluster']))

        #  Init. clustering
        kmeans_u = KMeans(n_cluster=settings_u['n_cluster'],
                          a_c_mat_0=a_c_mat_0_u,
                          l2_lambda=settings_u['l2_lambda_cls'])

        kmeans_i = KMeans(n_cluster=settings_i['n_cluster'],
                          a_c_mat_0=a_c_mat_0_i,
                          l2_lambda=settings_i['l2_lambda_cls'])

        boost_u = Boosting(cls=kmeans_u,
                           n_learner=settings_u['n_learner'],
                           n_iter_cls=settings_u['n_iter_cls'])

        boost_i = Boosting(cls=kmeans_i,
                           n_learner=settings_i['n_learner'],
                           n_iter_cls=settings_i['n_iter_cls'])

        # Init. updaters
        approx_upd_u = ApproximateUpdater(x_mat_0=x_mat_0_u,
                                          gamma=settings_u['gamma'])

        approx_upd_i = ApproximateUpdater(x_mat_0=x_mat_0_i,
                                          gamma=settings_i['gamma'])

        bfgs_upd_u = BFGS(x_mat_0=x_mat_0_u,
                          max_iter=settings_u['max_iter_bfgs'])

        bfgs_upd_i = BFGS(x_mat_0=x_mat_0_i,
                          max_iter=settings_i['max_iter_bfgs'])

        multi_upd_u = MultiUpdaterWrapper(upds=[approx_upd_u, bfgs_upd_u])
        multi_upd_i = MultiUpdaterWrapper(upds=[approx_upd_i, bfgs_upd_i])

        # Init. alternates
        alt_u = Alternate(cls=boost_u, upd=multi_upd_u)
        alt_i = Alternate(cls=boost_i, upd=multi_upd_i)

        # ------- Fit Vandermonde -------
        vm_u.fit()
        vm_u.transform(x_mat_0_u)

        vm_i.fit()
        vm_i.transform(x_mat_0_i)

        # ------- Do alternations -------
        # ToDo
        logger_u.rmse_va = []
        logger_i.rmse_va = []

        print('--- Alternating for user-based method: ---')
        a_mat_u_res = alt_u.run(vm_u, rating_mat_tr_u_res, rating_mat_va_u_res, n_alter, -np.inf, np.inf,
                                logger=logger_u,
                                rating_mat_te=rating_mat_te_u_res)

        print('--- Alternating for item-based method: ---')
        a_mat_i_res = alt_i.run(vm_i, rating_mat_tr_i_res, rating_mat_va_i_res, n_alter, -np.inf, np.inf,
                                logger=logger_i,
                                rating_mat_te=rating_mat_te_i_res)

        # ------- Do regression -------
        _, x_ui_tr, y_tr = fit_reg(vm_u, a_mat_u_res, mean_mat_u_res,
                                   vm_i, a_mat_i_res, mean_mat_i_res,
                                   rating_mat_tr_res)
        _, x_ui_te, y_te = fit_reg(vm_u, a_mat_u_res, mean_mat_u_res,
                                   vm_i, a_mat_i_res, mean_mat_i_res,
                                   rating_mat_te_res)
        reg, x_ui_va, y_va = fit_reg(vm_u, a_mat_u_res, mean_mat_u_res,
                                     vm_i, a_mat_i_res, mean_mat_i_res,
                                     rating_mat_va_res)

        y_tr_pr_res = reg.predict(X=x_ui_tr)
        y_va_pr_res = reg.predict(X=x_ui_va)
        y_te_pr_res = reg.predict(X=x_ui_te)

        y_tr_pr += y_tr_pr_res
        y_va_pr += y_va_pr_res
        y_te_pr += y_te_pr_res

        rmse_tr = calc_rmse(y_tr_pr, rating_mat_tr[mask_tr], min_value, max_value)
        rmse_va = calc_rmse(y_va_pr, rating_mat_va[mask_va], min_value, max_value)
        rmse_te = calc_rmse(y_te_pr, rating_mat_te[mask_te], min_value, max_value)

        print('---------------- Combined: ----------------')
        logger.log(rmse_tr, rmse_va, rmse_te)

        # ------- Calc. residuals -------
        rating_mat_tr_pr_res = rating_mat_tr.copy()
        rating_mat_tr_pr_res[mask_tr] = y_tr_pr_res
        rating_mat_va_pr_res = rating_mat_va.copy()
        rating_mat_va_pr_res[mask_va] = y_va_pr_res
        rating_mat_te_pr_res = rating_mat_te.copy()
        rating_mat_te_pr_res[mask_te] = y_te_pr_res

        rating_mat_tr_res -= rating_mat_tr_pr_res
        rating_mat_va_res -= rating_mat_va_pr_res
        rating_mat_te_res -= rating_mat_te_pr_res

        mean_mat_u_res = calc_row_mean_matrix(rating_mat_tr_res)
        mean_mat_i_res = calc_row_mean_matrix(rating_mat_tr_res.T)

        rating_mat_tr_u_res = rating_mat_tr_res - mean_mat_u_res
        rating_mat_tr_i_res = rating_mat_tr_res.T - mean_mat_i_res
        rating_mat_va_u_res = rating_mat_va_res - mean_mat_u_res
        rating_mat_va_i_res = rating_mat_va_res.T - mean_mat_i_res
        rating_mat_te_u_res = rating_mat_te_res - mean_mat_u_res
        rating_mat_te_i_res = rating_mat_te_res.T - mean_mat_i_res

    # ------- Save the results -------
    logger.save()
    logger_u.save()
    logger_i.save()
