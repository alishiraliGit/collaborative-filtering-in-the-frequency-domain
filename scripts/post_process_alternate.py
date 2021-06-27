from matplotlib import pyplot as plt
import numpy as np
import os

from app.models.logger import Logger


if __name__ == '__main__':
    loadpath = '../results'
    i_filename = 'result-ml1M-methodboosted_kmeans_approx_ls-dim_x3-m4-n_cluster2-cls_init_std1e-01-n_learner4-n_iter_cls3-gamma1-max_nfev3-l2_lambda0-l2_lambda_cls0-2020-10-02 01-08-26'
    u_filename = 'result-ml1M-methodboosted_kmeans_approx_ls_user_based_item_based-dim_x3-m4-n_cluster2-cls_init_std1e-01-n_learner4-n_iter_cls3-gamma1-max_nfev3-l2_lambda0-l2_lambda_cls0-2020-10-02 01-08-26'
    tot_filename = 'result-ml1M-methodboosted_kmeans_approx_ls_user_based_item_based-2020-10-02 01-08-26'

    i_dic = Logger.load(loadpath, i_filename)
    u_dic = Logger.load(loadpath, u_filename)
    tot_dic = Logger.load(loadpath, tot_filename)

    plt.figure(figsize=(4.5, 4))

    plt.plot(u_dic['rmse_tr'], 'b--', alpha=0.5, linewidth=1.5)
    plt.plot(u_dic['rmse_te'], 'r--', alpha=0.5, linewidth=1.5)

    plt.plot(i_dic['rmse_tr'], 'b.', alpha=1, linewidth=1.5)
    plt.plot(i_dic['rmse_te'], 'r.', alpha=1, linewidth=1.5)

    n_alter = 4

    plt.plot(np.array(range(1, 1 + len(tot_dic['rmse_va'])))*n_alter, np.array(tot_dic['rmse_va']), 'ko', alpha=1, linewidth=0.5)

    plt.legend(('user-based training', 'user-based validation', 'item-based training', 'item-based validation', 'combined validation'))

    plt.xlabel('#iteration')
    plt.ylabel('RMSE')

    # plt.yscale('log')

    plt.tight_layout()

    plt.savefig(os.path.join(loadpath, 'figs', 'alter-1m-training.pdf'))
