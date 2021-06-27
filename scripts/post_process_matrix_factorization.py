import numpy as np
import matplotlib.pyplot as plt
import os

from app.models.logger import Logger
from app.models.matrixfactorization.mf import MatrixFactorization
from app.utils.plotting import get_2d_grid


def plot2d(w_u, w_i, rat_mat, user, max_x, min_val, max_val):
    is_rated = np.where(~np.isnan(rat_mat[user]))

    plt.figure()

    # ----- Plot heatmap -----
    # Init. a grid
    xy_mat, n, extent = get_2d_grid(max_x)

    rat_pre = MatrixFactorization.predict(w_u[:, user:(user + 1)], xy_mat, min_val, max_val)

    plt.imshow(rat_pre.reshape((n, n)), cmap='hot', extent=extent, origin='lower')

    # ----- Plot items -----
    plt.scatter(w_i[0][is_rated], w_i[1][is_rated], c=rat_mat[user][is_rated], cmap='hot',
                edgecolors='k')

    # ----- Plot the user's arrow -----
    plt.annotate('', xy=(w_u[0][user], w_u[1][user]), xytext=(0, 0),
                 arrowprops={'arrowstyle': 'fancy', 'facecolor': 'b', 'edgecolor': 'w', 'mutation_scale': 40})


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
    filename = 'result-methodals-dim2-w_init_std1e-01-l2_lambda1-2021-04-04 01-26-05'

    # Dataset
    min_value = -0.5
    max_value = 0.5

    # Plot
    max_x_y = 1  # 4.7
    u = 7

    # ------- Load model -------
    ext_dic = Logger.load(model_load_path, filename, load_ext=True)

    print('Model loaded ...')

    # ------- Calc. rmse -------
    cv = 'tr'

    rating_mat_pr = MatrixFactorization.predict(ext_dic['w_u'], ext_dic['w_i'], min_value, max_value)

    rmse = calc_rmse_per_user(ext_dic['rating_mat_' + cv], rating_mat_pr)

    print('User %d is selected with rmse = %.3f' % (u, rmse[u]))

    # ------- Plot -------
    plot2d(ext_dic['w_u'], ext_dic['w_i'], ext_dic['rating_mat_' + cv],
           user=u,
           max_x=max_x_y,
           min_val=min_value,
           max_val=max_value)

    plt.colorbar()

    plt.savefig(os.path.join(save_path, cv + '_' + filename))
