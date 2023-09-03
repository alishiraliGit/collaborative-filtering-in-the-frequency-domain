import os
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.models.logger import Logger
from app.models.vandermonde import Vandermonde, VandermondeType, RegularizationType


def fit_vm_and_predict(x_mat, a_mat, settings, min_val, max_val):
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
    rat_pre[rat_pre > max_val] = max_val
    rat_pre[rat_pre < min_val] = min_val

    return rat_pre


if __name__ == '__main__':
    # ------- Settings -------
    # Path
    model_load_path = os.path.join('..', 'results')

    # Files
    u_filename = 'result-methodboosted_kmeans_approx_bfgs-dim_x2-m3-vm_typecos_mult-reg_typel2-reg_params-l2_lambda10-exclude_zero_freqTrue-clust_methodboosting-n_learner7-n_cluster2-n_iter_clust5-std_init_clust1e-02-g-2023-09-01 23-13-53'
    i_filename = 'result-methodboosted_kmeans_approx_bfgs-dim_x2-m4-vm_typecos_mult-reg_typel2-reg_params-l2_lambda10-exclude_zero_freqTrue-clust_methodboosting-n_learner10-n_cluster2-n_iter_clust5-std_init_clust1e-02--2023-09-01 23-26-39'

    # Dataset
    min_value = 1
    max_value = 5

    # Other
    u_sett = {
        'dim_x': 2,
        'm': 3,
        'vm_type': VandermondeType.COS_MULT,
        'reg_type': RegularizationType.L2,
        'reg_params': {'l2_lambda': 10, 'exclude_zero_freq': True},
    }

    i_sett = {
        'dim_x': 2,
        'm': 4,
        'vm_type': VandermondeType.COS_MULT,
        'reg_type': RegularizationType.L2,
        'reg_params': {'l2_lambda': 10, 'exclude_zero_freq': True},
    }

    # ------- Load model -------
    u_ext_dic = Logger.load(model_load_path, u_filename, load_ext=True)
    i_ext_dic = Logger.load(model_load_path, i_filename, load_ext=True)

    print('Model loaded ...')

    rating_mat_va = u_ext_dic['rating_mat_va']
    rating_mat_te = u_ext_dic['rating_mat_te']

    # ------- Predict per model ------
    u_rating_mat_pr = fit_vm_and_predict(
        x_mat=u_ext_dic['x_mat'],
        a_mat=u_ext_dic['a_mat'],
        settings=u_sett,
        min_val=min_value,
        max_val=max_value
    )

    i_rating_mat_pr = fit_vm_and_predict(
        x_mat=i_ext_dic['x_mat'],
        a_mat=i_ext_dic['a_mat'],
        settings=i_sett,
        min_val=min_value,
        max_val=max_value
    ).T

    # ------- Combine predictions -------
    reg = LinearRegression()
    # reg = LogisticRegression()

    # Fitting
    mask_va = ~np.isnan(rating_mat_va)

    X_va = np.concatenate((u_rating_mat_pr[mask_va].reshape((-1, 1)),
                           i_rating_mat_pr[mask_va].reshape((-1, 1))), axis=1)
    y_va = rating_mat_va[mask_va]

    # c = 10
    # X_va = np.arctanh((X_va - (max_value + min_value)/2)/(max_value - min_value))/c
    # y_va = np.arctanh((y_va - (max_value + min_value)/2)/(max_value - min_value))/c
    reg.fit(X_va, y_va)

    # Prediction
    mask_te = ~np.isnan(rating_mat_te)

    X_te = np.concatenate((u_rating_mat_pr[mask_te].reshape((-1, 1)),
                           i_rating_mat_pr[mask_te].reshape((-1, 1))), axis=1)
    y_te = rating_mat_te[mask_te]

    # X_te = np.arctanh((X_te - (max_value + min_value)/2)/(max_value - min_value))/c

    y_pr = reg.predict(X_te)
    # y_pr = np.tanh(y_pr*c)*(max_value - min_value) + (max_value + min_value)/2

    y_pr[y_pr > max_value] = max_value
    y_pr[y_pr < min_value] = min_value

    # ------- Evaluation -------
    print('rmse u test is: %.3f' % np.sqrt(np.mean((u_rating_mat_pr[mask_te] - y_te)**2)))
    print('rmse i test is: %.3f' % np.sqrt(np.mean((i_rating_mat_pr[mask_te] - y_te) ** 2)))
    print('rmse combined test is: %.3f' % np.sqrt(np.mean((y_pr - y_te)**2)))
