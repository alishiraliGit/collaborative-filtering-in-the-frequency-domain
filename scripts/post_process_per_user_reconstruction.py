import numpy as np
import matplotlib.pyplot as plt
import os

from app.models.logger import Logger
from app.models.vandermonde import Vandermonde
from app.utils.plotting import get_2d_grid


def fit_vm_and_predict(x_mat, a_mat, settings, users, min_val, max_val):
    # Init. and fit a VM
    vm = Vandermonde.get_instance(
        dim_x=settings['dim_x'],
        m=settings['m'],
        l2_lambda=settings['l2_lambda'],
        vm_type=settings['vm_type']
    )

    vm.fit()
    vm.transform(x_mat)

    # Predict
    rat_pre = vm.predict(a_mat[:, users])
    rat_pre[rat_pre > max_val] = max_val
    rat_pre[rat_pre < min_val] = min_val

    return rat_pre


def plot2d(w_i, a_mat, settings, rat_mat, user, min_val, max_val):
    is_rated = np.where(~np.isnan(rat_mat[user]))

    plt.figure()

    # ----- Plot heatmap -----
    # Init. a grid
    xy_mat, n, extent = get_2d_grid(3.1)

    # Predict
    rat_pre = fit_vm_and_predict(xy_mat, a_mat, settings, [user], min_val, max_val)

    # Plot
    plt.imshow(rat_pre.reshape((n, n), order='F'), cmap='hot', extent=extent, origin='lower')

    # ----- Plot items -----
    plt.scatter(w_i[0][is_rated], w_i[1][is_rated], c=rat_mat[user][is_rated], cmap='hot', edgecolors='k')


def calc_rmse_per_user(rat_mat_te, rat_mat_pr):
    err = rat_mat_te - rat_mat_pr

    return np.sqrt(np.nanmean(err ** 2, axis=1))


if __name__ == '__main__':
    # ------- Settings -------
    # Path
    model_load_path = os.path.join('..', 'results')

    save_path = os.path.join('..', 'results', 'figs')
    os.makedirs(save_path, exist_ok=True)

    # File
    # REAL-Replicated
    # filename = 'result-methodbfgs-dim_x2-m2-vm_typereal-cls_init_std1e-01-gamma1-max_iter20-l2_lambda100-2021-02-07 20-55-29'
    # REAL
    # filename = 'result-methodbfgs-dim_x2-m2-vm_typereal-cls_init_std1e-01-gamma1-max_iter20-l2_lambda1-2021-02-07 20-20-44'
    # COS-Replicated
    # filename = 'result-methodbfgs-dim_x2-m2-vm_typecos-cls_init_std1e-01-gamma1-max_iter20-l2_lambda100-2021-02-07 21-31-51'
    # COS
    # filename = 'result-methodbfgs-dim_x2-m2-vm_typecos-cls_init_std1e-01-gamma1-max_iter20-l2_lambda1-2021-02-07 21-38-18'
    # COS tanh
    filename = 'result-methodbfgs-dim_x2-m5-vm_typecos-cls_init_std1e-01-gamma1-max_iter5-l2_lambda1000-2021-04-03 13-53-23'

    # Dataset
    min_value = -0.5
    max_value = 0.5

    # ------- Load model -------
    ext_dic = Logger.load(model_load_path, filename, load_ext=True)
    dic = Logger.load(model_load_path, filename, load_ext=False)

    # ToDo: Remove these lines
    ext_dic['a_mat'] = ext_dic['a_mat'][:, -8:]
    ext_dic['rating_mat_tr'] = ext_dic['rating_mat_tr'][-8:]
    ext_dic['rating_mat_va'] = ext_dic['rating_mat_va'][-8:]
    ext_dic['rating_mat_te'] = ext_dic['rating_mat_te'][-8:]

    print('Model loaded ...')

    # ------- Predict ratings -------
    n_user = ext_dic['a_mat'].shape[1]

    rating_mat_pr = fit_vm_and_predict(x_mat=ext_dic['x_mat'],
                                       a_mat=ext_dic['a_mat'],
                                       settings=dic['settings'],
                                       users=list(range(n_user)),
                                       min_val=min_value,
                                       max_val=max_value)

    # ------- Calc. err and select the user -------
    rmse = calc_rmse_per_user(ext_dic['rating_mat_tr'], rating_mat_pr)

    arg_sorted = np.argsort(rmse)

    u = arg_sorted[0]

    print('User %d selected with rmse = %.3f' % (u, rmse[u]))

    # ------- Plot -------
    cv = 'tr'
    plot2d(ext_dic['x_mat'], ext_dic['a_mat'], dic['settings'], ext_dic['rating_mat_' + cv],
           user=u,
           min_val=min_value,
           max_val=max_value)

    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))

    plt.colorbar()

    # plt.savefig(os.path.join(save_path, cv + '_' + filename))

    #
    mask = ~np.isnan(ext_dic['rating_mat_' + cv])
    rating_mat_pr_bin = rating_mat_pr > 0
    print(np.mean(rating_mat_pr_bin[mask] == (ext_dic['rating_mat_' + cv][mask] > 0)))

