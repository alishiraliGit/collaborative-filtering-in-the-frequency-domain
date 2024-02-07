import os
from sklearn.linear_model import LinearRegression
# from scipy.io import savemat

from app.models.logger import Logger
from app.models.vandermonde import Vandermonde, VandermondeType, RegularizationType
from app.utils.metrics import *


def fit_vm_and_predict(x_mat, a_mat, settings, min_val, max_val, truncate=True):
    # Init. and fit a VM
    vm = Vandermonde.get_instance(
        dim_x=settings['dim_x'],
        m=settings['m'],
        vm_type=settings['vm_type'],
        reg_type=settings['reg_type'],
        reg_params=settings['reg_params']
    )

    vm.fit()
    vm.transform(x_mat)

    # Predict
    rat_pre = vm.predict(a_mat)
    if truncate:
        rat_pre[rat_pre > max_val] = max_val
        rat_pre[rat_pre < min_val] = min_val

    return rat_pre


if __name__ == '__main__':
    # ------- Settings -------
    # Path
    model_load_path = os.path.join('..', 'results')

    # Files
    u_filename = 'result-methodboosted_kmeans_approx_bfgs-dim_x3-m7-vm_typecos_mult-reg_typel2-reg_params-l2_lambda30-exclude_zero_freqTrue-clust_methodboosting-n_learner10-n_cluster2-n_iter_clust5-std_init_clust1e-02--2024-02-06 21-41-51'
    i_filename = 'result-methodboosted_kmeans_approx_bfgs-dim_x3-m7-vm_typecos_mult-reg_typel2-reg_params-l2_lambda30-exclude_zero_freqTrue-clust_methodboosting-n_learner10-n_cluster2-n_iter_clust5-std_init_clust1e-02--2024-02-06 21-42-14'

    # Dataset
    min_value = 1
    max_value = 5
    truncate_ui = False

    # Other
    u_sett = {
        'dim_x': 3,
        'm': 7,
        'vm_type': VandermondeType.COS_MULT,
        'reg_type': RegularizationType.L2,
        'reg_params': {'l2_lambda': 30, 'exclude_zero_freq': True},
    }

    i_sett = {
        'dim_x': 3,
        'm': 7,
        'vm_type': VandermondeType.COS_MULT,
        'reg_type': RegularizationType.L2,
        'reg_params': {'l2_lambda': 30, 'exclude_zero_freq': True},
    }

    # ------- Load model -------
    u_ext_dic = Logger.load(model_load_path, u_filename, load_ext=True)
    i_ext_dic = Logger.load(model_load_path, i_filename, load_ext=True)

    print('Model loaded ...')

    rating_mat_va = u_ext_dic['rating_mat_va']
    rating_mat_te = u_ext_dic['rating_mat_te']

    # Save data for consistency in evaluation of baselines
    # savemat(os.path.join(model_load_path, 'yahoo_r3_split.mat'), {
    #     'tr': u_ext_dic['rating_mat_tr'],
    #     'va': u_ext_dic['rating_mat_va'],
    #     'te': u_ext_dic['rating_mat_te']
    # })

    # ------- Predict per model ------
    u_rating_mat_pr = fit_vm_and_predict(
        x_mat=u_ext_dic['x_mat'],
        a_mat=u_ext_dic['a_mat'],
        settings=u_sett,
        min_val=min_value,
        max_val=max_value,
        truncate=truncate_ui
    )

    i_rating_mat_pr = fit_vm_and_predict(
        x_mat=i_ext_dic['x_mat'],
        a_mat=i_ext_dic['a_mat'],
        settings=i_sett,
        min_val=min_value,
        max_val=max_value,
        truncate=truncate_ui
    ).T

    # ------- Combine predictions -------
    reg = LinearRegression()

    # Fitting
    mask_va = ~np.isnan(rating_mat_va)

    X_va = np.concatenate((u_rating_mat_pr[mask_va].reshape((-1, 1)),
                           i_rating_mat_pr[mask_va].reshape((-1, 1))), axis=1)
    y_va = rating_mat_va[mask_va]

    reg.fit(X_va, y_va)

    # Prediction
    mask_te = ~np.isnan(rating_mat_te)

    X_te = np.concatenate((u_rating_mat_pr[mask_te].reshape((-1, 1)),
                           i_rating_mat_pr[mask_te].reshape((-1, 1))), axis=1)

    y_pr = reg.predict(X_te)

    y_pr[y_pr > max_value] = max_value
    y_pr[y_pr < min_value] = min_value

    rating_mat_pr = rating_mat_te.copy()
    rating_mat_pr[mask_te] = y_pr

    # ------- Evaluation -------
    print('rmse u test is: %.3f' % rmse(u_rating_mat_pr, rating_mat_te, mask_te))
    print('rmse i test is: %.3f' % rmse(i_rating_mat_pr, rating_mat_te, mask_te))
    print('rmse combined test is: %.3f' % rmse(rating_mat_pr, rating_mat_te, mask_te))
    print('additional metrics:')
    print('mae combined is: %.3f' % mae(rating_mat_pr, rating_mat_te, mask_te))
    print('recall@%d combined is: %.3f' % (5, recall(rating_mat_pr, rating_mat_te, mask_te, 5)))
    print('recall@%d combined is: %.3f' % (10, recall(rating_mat_pr, rating_mat_te, mask_te, 10)))
    print('ndcg@%d combined is: %.3f' % (5, ndcg(rating_mat_pr, rating_mat_te, mask_te, 5)))
    print('ndcg@%d combined is: %.3f' % (10, ndcg(rating_mat_pr, rating_mat_te, mask_te, 10)))
