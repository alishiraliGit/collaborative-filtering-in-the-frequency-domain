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
