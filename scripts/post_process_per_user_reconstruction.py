import numpy as np
import matplotlib.pyplot as plt
import os

from app.models.logger import Logger
from app.models.matrixfactorization.mf import MatrixFactorization


def plot2d(w_u, w_i, rat_mat, user, max_x, min_val, max_val):
    is_rated = np.where(~np.isnan(rat_mat[user]))

    plt.figure()

    # Plot heatmap
    vec = np.arange(-max_x, max_x, 0.1).reshape((-1, 1), order='F')
    d_vec = (vec[1, 0] - vec[0, 0]) / 2
    extent = [vec[0, 0] - d_vec, vec[-1, 0] + d_vec, vec[0, 0] - d_vec, vec[-1, 0] + d_vec]

    x_mat = np.tile(vec, reps=(1, vec.shape[0]))
    y_mat = np.tile(vec.T, reps=(vec.shape[0], 1))

    x_mat_vec = x_mat.reshape((1, -1), order='F')
    y_mat_vec = y_mat.reshape((1, -1), order='F')
    xy_mat = np.concatenate((x_mat_vec, y_mat_vec), axis=0)

    rat_pre = MatrixFactorization.predict(w_u[:, user:(user + 1)], xy_mat, min_val, max_val)

    plt.imshow(rat_pre.reshape((vec.shape[0], vec.shape[0])), cmap='hot', extent=extent, origin='lower')

    # Plot items
    plt.scatter(w_i[0][is_rated], w_i[1][is_rated], c=rat_mat[user][is_rated], cmap='hot',
                edgecolors='k')

    # Plot the user's arrow
    plt.annotate('', xy=(w_u[0][user], w_u[1][user]), xytext=(0, 0),
                 arrowprops={'arrowstyle': 'fancy', 'color': 'k'})


if __name__ == '__main__':
    # ------- Settings -------
    # Path
    model_load_path = os.path.join('..', 'results')

    save_path = os.path.join('..', 'results', 'figs')
    os.makedirs(save_path, exist_ok=True)

    # File
    filename = 'result-methodals-dim2-w_init_std1e-01-l2_lambda1-2021-02-06 12-06-36'

    # Dataset
    min_value = -10
    max_value = 10

    # Plot
    max_x_y = 4.1
    u = 10

    # ------- Load model -------
    ext_dic = Logger.load(model_load_path, filename, load_ext=True)

    print('Model loaded ...')

    # ------- Plot -------
    plot2d(ext_dic['w_u'], ext_dic['w_i'], ext_dic['rating_mat_te'],
           user=u,
           max_x=max_x_y,
           min_val=min_value,
           max_val=max_value)
