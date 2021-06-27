from matplotlib import pyplot as plt
import numpy as np


def plot_with_conf_interval(x, mat, color):
    mean = np.mean(mat, axis=1)
    std = np.std(mat, axis=1)

    n_obs = mat.shape[1]

    plt.plot(x, mean, color=color)

    plt.fill_between(x, mean + std/np.sqrt(n_obs - 1), mean - std/np.sqrt(n_obs - 1),
                     facecolor=color,
                     alpha=0.5)


def get_2d_grid(max_x):
    vec = np.arange(-max_x, max_x, 0.01).reshape((-1, 1), order='F')
    d_vec = (vec[1, 0] - vec[0, 0]) / 2

    x_mat = np.tile(vec, reps=(1, vec.shape[0]))
    y_mat = np.tile(vec.T, reps=(vec.shape[0], 1))

    x_mat_vec = x_mat.reshape((1, -1), order='F')
    y_mat_vec = y_mat.reshape((1, -1), order='F')
    xy_mat = np.concatenate((x_mat_vec, y_mat_vec), axis=0)

    # Calc. extent for imshow plot
    extent = [vec[0, 0] - d_vec, vec[-1, 0] + d_vec, vec[0, 0] - d_vec, vec[-1, 0] + d_vec]

    return xy_mat, vec.shape[0], extent

