import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import os
from datetime import datetime
from scipy.io import loadmat, savemat

from app.models.vandermonde import Vandermonde
from app.models.clustering.kmeans import KMeans
from app.models.clustering.boosting import Boosting
from app.utils.mat_ops import get_ari
from app.utils.plotting import plot_with_conf_interval
from app.models.logger import Logger


# plt.rcParams['font.family'] = 'Times New Roman'


def get_file_name(savepath, sett):
    return os.path.join(savepath, 'cls',
                        'result' + Logger.stringify(sett) +
                        '-' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.mat')


def calc_ari(u_clusters_predicted, u_clusters_truth):
    k = len(u_clusters_predicted)
    s = len(u_clusters_truth)
    contingency = np.zeros((k, s))

    for u_user, cls_user in enumerate(u_clusters_truth):
        cls_user_pr = u_clusters_predicted[u_user]

        contingency[cls_user_pr, cls_user] += 1

    return get_ari(contingency)


def get_synthetic_data(sett):
    x_mat = np.random.random((sett['dim_x'], sett['n_item']))

    vm = Vandermonde(sett['dim_x'], sett['m'], 0)
    vm.fit()
    vm.transform(x_mat)

    a_c_mat = np.random.normal(loc=0, scale=sett['cls_init_std'], size=(vm.dim_a, sett['n_cluster']))
    users_clusters = np.random.randint(low=0, high=sett['n_cluster'], size=(sett['n_user'],))

    a_mat = a_c_mat[:, users_clusters] \
        + np.random.normal(loc=0, scale=sett['user_std'], size=(vm.dim_a, sett['n_user']))

    rating_mat = vm.predict(a_mat)

    user_indices = np.tile(np.array(range(sett['n_user'])).reshape((-1, 1)), (1, sett['n_item'])).reshape((-1,))
    item_indices = np.tile(np.array(range(sett['n_item'])).reshape((1, -1)), (sett['n_user'], 1)).reshape((-1,))

    perm_indices = np.random.permutation(sett['n_user']*sett['n_item'])

    n_selected = np.floor(sett['n_user']*sett['n_item']*sett['density']).astype(int)

    rating_mat[user_indices[perm_indices][n_selected:], item_indices[perm_indices][n_selected:]] = np.nan

    return x_mat, a_c_mat, rating_mat, users_clusters


def get_synthetic_settings():
    sett = {}

    method = 'sim_1'

    sett['method'] = method

    # General settings
    sett['n_user'] = 50
    sett['n_item'] = 200

    sett['density'] = 0.1

    # Vandermonde settings
    sett['dim_x'] = 2
    sett['m'] = 3

    # Clustering settings
    sett['n_cluster'] = 4
    sett['cls_init_std'] = 0.1

    sett['user_std'] = 0.01

    return sett


def get_compatible_kmeans_settings(sim_sett):
    sett = {}

    method = 'kmeans'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = sim_sett['dim_x']
    sett['m'] = sim_sett['m']

    # Clustering settings
    sett['n_cluster'] = sim_sett['n_cluster']
    sett['cls_init_std'] = 0.1
    sett['n_iter_cls'] = 5

    # Estimate regularization coefficients
    sett['l2_lambda_cls'] = 0

    return sett


def get_compatible_boosted_kmeans_settings(sim_sett):
    sett = {}

    method = 'boosted_kmeans'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = sim_sett['dim_x']
    sett['m'] = sim_sett['m']

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = np.round(sim_sett['n_cluster']).astype(int)
    sett['n_iter_cls'] = 5

    # Estimate regularization coefficients
    sett['l2_lambda'] = 0.1
    sett['l2_lambda_cls'] = 0

    return sett


def run_kmeans(sim_sett, x_mat, rating_mat, l2_lambda=0):
    sett_kmean = get_compatible_kmeans_settings(sim_sett)

    # VM
    vm_kmean = Vandermonde(dim_x=sett_kmean['dim_x'],
                           m=sett_kmean['m'],
                           l2_lambda=l2_lambda)

    #  Init. "x" and "a_c"
    x_0 = x_mat.copy()
    a_c_mat_kmeans_0 = np.random.normal(loc=0,
                                        scale=sett_kmean['cls_init_std'],
                                        size=(vm_kmean.dim_a, sett_kmean['n_cluster']))

    vm_kmean.fit()
    vm_kmean.transform(x_0)

    # K-mean
    cls_kmean = KMeans(n_cluster=sett_kmean['n_cluster'],
                       a_c_mat_0=a_c_mat_kmeans_0,
                       l2_lambda=sett_kmean['l2_lambda_cls'])

    a_c_mat_kmean, users_clusters_kmean = cls_kmean.fit_transform(vm_kmean, rating_mat)
    for _ in range(sett_kmean['n_iter_cls'] - 1):
        a_c_mat_kmean, users_clusters_kmean = cls_kmean.fit_transform(vm_kmean, rating_mat)

    return a_c_mat_kmean, users_clusters_kmean


def run_boosted_kmeans(sim_sett, x_mat, rating_mat):
    sett_boost = get_compatible_boosted_kmeans_settings(sim_sett)

    # VM
    vm_boost = Vandermonde(dim_x=sett_boost['dim_x'],
                           m=sett_boost['m'],
                           l2_lambda=sett_boost['l2_lambda'])

    # Init. "x" and "a_c"
    x_0 = x_mat.copy()
    a_c_mat_boost_0 = np.random.normal(loc=0,
                                       scale=sett_boost['cls_init_std'],
                                       size=(vm_boost.dim_a, sett_boost['n_cluster']))

    vm_boost.fit()
    vm_boost.transform(x_0)

    # K-mean
    cls_kmean = KMeans(n_cluster=sett_boost['n_cluster'],
                       a_c_mat_0=a_c_mat_boost_0,
                       l2_lambda=sett_boost['l2_lambda_cls'])

    # Boosting
    cls_boost = Boosting(cls_kmean, sett_boost['n_learner'], sett_boost['n_iter_cls'])

    a_c_mat_boost, pseudo_users_clusters_boost = cls_boost.fit_transform(vm_boost, rating_mat)

    a_mat_boost = a_c_mat_boost[:, pseudo_users_clusters_boost]

    dist = squareform(pdist(a_mat_boost.T))

    users_clusters_boost = -np.ones((sim_sett['n_user'],)).astype(int)
    cls = 0
    for user in range(sim_sett['n_user']):
        if users_clusters_boost[user] < 0:
            users_clusters_boost[dist[user] < 1e-6] = cls
            cls += 1

    return a_c_mat_boost, users_clusters_boost


def calc_ari_vs_num_cls(savepath):
    sim_settings = get_synthetic_settings()

    n_rept = 30
    n_cluster_range = range(2, 8)
    n_user_per_cluster = sim_settings['n_user'] / np.array(n_cluster_range)

    n_user_per_cluster_truth = sim_settings['n_user']/sim_settings['n_cluster']

    kmean_ari = np.zeros((len(n_cluster_range), n_rept))
    kmean_reg_ari = np.zeros((len(n_cluster_range), n_rept))
    boost_ari = np.zeros((len(n_cluster_range), n_rept))

    for i_rept in range(n_rept):
        for i_cls, n_cls in enumerate(n_cluster_range):
            # sim_settings['n_cluster'] = n_cls
            sim_settings_copy = sim_settings.copy()

            sim_settings_copy['n_cluster'] = n_cls

            x, a_c, rat_mat, u_clusters = get_synthetic_data(sim_settings)

            try:
                _a_c_kmean, u_clusters_kmean = run_kmeans(sim_settings_copy, x, rat_mat, l2_lambda=0)
                kmean_ari[i_cls, i_rept] = calc_ari(u_clusters, u_clusters_kmean)
            except np.linalg.LinAlgError:
                kmean_ari[i_cls, i_rept] = np.nan

            try:
                _a_c_kmean_reg, u_clusters_kmean_reg = run_kmeans(sim_settings_copy, x, rat_mat, l2_lambda=10)
                kmean_reg_ari[i_cls, i_rept] = calc_ari(u_clusters, u_clusters_kmean_reg)
            except np.linalg.LinAlgError:
                kmean_reg_ari[i_cls, i_rept] = np.nan

            try:
                _a_c_boost, u_clusters_boost = run_boosted_kmeans(sim_settings_copy, x, rat_mat)
                boost_ari[i_cls, i_rept] = calc_ari(u_clusters, u_clusters_boost)
            except np.linalg.LinAlgError:
                boost_ari[i_cls, i_rept] = np.nan

    sim_settings.pop('n_cluster')
    sim_settings['method'] = 'n_cluster'

    savemat(get_file_name(savepath, sim_settings), {'kmean': kmean_ari,
                                                    'kmean_reg': kmean_reg_ari,
                                                    'boost': boost_ari,
                                                    'rng': n_user_per_cluster,
                                                    'truth': n_user_per_cluster_truth})


def calc_ari_vs_density(savepath):
    sim_settings = get_synthetic_settings()

    n_rept = 20
    density_range = np.logspace(np.log10(0.005), np.log10(0.5), 10)

    kmean_ari = np.zeros((len(density_range), n_rept))
    kmean_reg_ari = np.zeros((len(density_range), n_rept))
    boost_ari = np.zeros((len(density_range), n_rept))

    for i_rept in range(n_rept):
        for i_sp, density in enumerate(density_range):
            sim_settings['density'] = density

            x, a_c, rat_mat, u_clusters = get_synthetic_data(sim_settings)

            try:
                _a_c_kmean, u_clusters_kmean = run_kmeans(sim_settings, x, rat_mat, l2_lambda=0)
                kmean_ari[i_sp, i_rept] = calc_ari(u_clusters, u_clusters_kmean)
            except np.linalg.LinAlgError:
                kmean_ari[i_sp, i_rept] = np.nan

            try:
                _a_c_kmean_reg, u_clusters_kmean_reg = run_kmeans(sim_settings, x, rat_mat, l2_lambda=10)
                kmean_reg_ari[i_sp, i_rept] = calc_ari(u_clusters, u_clusters_kmean_reg)
            except np.linalg.LinAlgError:
                kmean_reg_ari[i_sp, i_rept] = np.nan

            try:
                _a_c_boost, u_clusters_boost = run_boosted_kmeans(sim_settings, x, rat_mat)
                boost_ari[i_sp, i_rept] = calc_ari(u_clusters, u_clusters_boost)
            except np.linalg.LinAlgError:
                boost_ari[i_sp, i_rept] = np.nan

    sim_settings.pop('density')
    sim_settings['method'] = 'density'

    savemat(get_file_name(savepath, sim_settings), {'kmean': kmean_ari,
                                                    'kmean_reg': kmean_reg_ari,
                                                    'boost': boost_ari,
                                                    'rng': density_range})


def calc_ari_vs_std_ratio(savepath):
    sim_settings = get_synthetic_settings()

    n_rept = 50
    user_std_range = np.logspace(np.log10(sim_settings['cls_init_std']/100), np.log10(sim_settings['cls_init_std']), 15)

    kmean_ari = np.zeros((len(user_std_range), n_rept))
    kmean_reg_ari = np.zeros((len(user_std_range), n_rept))
    boost_ari = np.zeros((len(user_std_range), n_rept))

    for i_rept in range(n_rept):
        for i_std, user_std in enumerate(user_std_range):
            sim_settings['user_std'] = user_std

            x, a_c, rat_mat, u_clusters = get_synthetic_data(sim_settings)

            try:
                _a_c_kmean, u_clusters_kmean = run_kmeans(sim_settings, x, rat_mat, l2_lambda=0)
                kmean_ari[i_std, i_rept] = calc_ari(u_clusters, u_clusters_kmean)
            except np.linalg.LinAlgError:
                kmean_ari[i_std, i_rept] = np.nan

            try:
                _a_c_kmean_reg, u_clusters_kmean_reg = run_kmeans(sim_settings, x, rat_mat, l2_lambda=10)
                kmean_reg_ari[i_std, i_rept] = calc_ari(u_clusters, u_clusters_kmean_reg)
            except np.linalg.LinAlgError:
                kmean_reg_ari[i_std, i_rept] = np.nan

            try:
                _a_c_boost, u_clusters_boost = run_boosted_kmeans(sim_settings, x, rat_mat)
                boost_ari[i_std, i_rept] = calc_ari(u_clusters, u_clusters_boost)
            except np.linalg.LinAlgError:
                boost_ari[i_std, i_rept] = np.nan

    sim_settings.pop('user_std')
    sim_settings['method'] = 'user_std'

    savemat(get_file_name(savepath, sim_settings), {'kmean': kmean_ari,
                                                    'kmean_reg': kmean_reg_ari,
                                                    'boost': boost_ari,
                                                    'rng': user_std_range})


def plot_one(data_dic, x_label, scale):
    # plot_with_conf_interval(data_dic['rng'][0], data_dic['kmean'], [0.5, 0.5, 1])
    plot_with_conf_interval(data_dic['rng'][0], data_dic['kmean_reg'], [0.5, 0.5, 1])
    plot_with_conf_interval(data_dic['rng'][0], data_dic['boost'], [1, 0.5, 0.5])
    plot_with_conf_interval(data_dic['rng'][0], data_dic['nn'], [0.5, 1, 0.5])

    plt.xlabel(x_label)
    plt.ylim((-0.25, 1))

    plt.xscale(scale)

    plt.tight_layout()


def plot_all(load_path):
    density_nn_data = loadmat(os.path.join(load_path, 'cls', 'nn_ari_vs_density.mat'))
    n_cls_nn_data = loadmat(os.path.join(load_path, 'cls', 'nn_ari_vs_n_cluster.mat'))
    user_std_nn_data = loadmat(os.path.join(load_path, 'cls', 'nn_ari_vs_user_std.mat'))

    density_data = loadmat(os.path.join(load_path, 'cls', 'result-methoddensity-n_user50-n_item200-dim_x2-m3-n_cluster4-cls_init_std1e-01-user_std1e-02-2020-10-01 22-42-09.mat'))
    n_cls_data = loadmat(os.path.join(load_path, 'cls', 'result-methodn_cluster-n_user50-n_item200-density1e-01-dim_x2-m3-cls_init_std1e-01-user_std1e-02-2020-10-01 22-42-29.mat'))
    user_std_data = loadmat(os.path.join(load_path, 'cls', 'result-methoduser_std-n_user50-n_item200-density1e-01-dim_x2-m3-n_cluster4-cls_init_std1e-01-2020-10-01 22-48-38.mat'))

    n_cls_data['rng'] = 50/n_cls_data['rng']
    n_cls_data['truth'] = 50/n_cls_data['truth']

    user_std_data['rng'] = 0.1/user_std_data['rng']

    density_data['nn'] = density_nn_data['ari']
    n_cls_data['nn'] = n_cls_nn_data['ari']
    user_std_data['nn'] = user_std_nn_data['ari']

    plt.figure(figsize=(3, 7))

    plt.subplot(3, 1, 1)
    plot_one(user_std_data, x_label='discriminability', scale='log')

    plt.legend(('k-rep', 'boosted r-rep', 'SmoothRecNet'))
    plt.ylabel('ARI')

    plt.subplot(3, 1, 2)
    plot_one(density_data, x_label='density', scale='log')
    plt.ylabel('ARI')

    plt.subplot(3, 1, 3)
    plot_one(n_cls_data, x_label='#cluster', scale='linear')
    min_y, max_y = plt.ylim()
    plt.plot([n_cls_data['truth'][0]]*2, [min_y, max_y], 'k--', linewidth=2)
    plt.ylabel('ARI')


if __name__ == '__main__':
    save_path = os.path.join('..', 'results')

    plot_all(save_path)

    plt.savefig(os.path.join(save_path, 'figs', 'demo_clustering.png'))
