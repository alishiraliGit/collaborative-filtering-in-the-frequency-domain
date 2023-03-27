import numpy as np

from app.models.vandermonde import Vandermonde, RegularizationType
from app.models.clustering.clustering_base import Clustering
from app.models.updating.updater_base import Updater
from app.models.logger import Logger


class Alternate:
    def __init__(self, clust: Clustering, upd: Updater):
        self.clust = clust
        self.upd = upd

    def run(self, vm: Vandermonde, rating_mat_tr, rating_mat_va, n_iter, min_val, max_val,
            rating_mat_te=None,
            logger: Logger = None,
            propensity_mat=None,
            obtain_tt=False,
            verbose=False):

        a_mat_opt = None
        x_mat_opt = None
        for it in range(n_iter):
            if vm.reg_type == RegularizationType.POST_MAX_SNR:
                obtain_tt_i = obtain_tt and (it == n_iter - 1)
            else:
                obtain_tt_i = obtain_tt

            # Do clustering
            a_c_mat, users_clusters = self.clust.fit_transform(
                vm, rating_mat_tr, propensity_mat=propensity_mat,
                verbose=verbose
            )

            a_mat = a_c_mat[:, users_clusters]

            if obtain_tt_i:
                a_c_mat_tt, users_clusters_tt = self.clust.fit_transform(
                    vm, rating_mat_tr, propensity_mat=propensity_mat,
                    is_test=True,
                    verbose=verbose
                )
                a_mat_tt = a_c_mat_tt[:, users_clusters_tt]
            else:
                a_mat_tt = a_mat

            # Do updating
            self.upd.fit_transform(vm, a_mat, rating_mat_tr, propensity_mat=propensity_mat)

            # Calc rmse
            rmse_tr = self.calc_prediction_rmse(vm, a_mat, rating_mat_tr, min_val, max_val)
            rmse_va = self.calc_prediction_rmse(vm, a_mat, rating_mat_va, min_val, max_val)
            rmse_te = self.calc_prediction_rmse(vm, a_mat_tt, rating_mat_te, min_val, max_val)

            if isinstance(logger, Logger):
                logger.log(rmse_tr, rmse_va, rmse_te)

                if rmse_va > np.min(logger.rmse_va):
                    continue

                if obtain_tt_i:
                    a_mat_opt = a_mat_tt.copy()
                else:
                    a_mat_opt = a_mat.copy()
                x_mat_opt = self.upd.x_mat.copy()
                print('(this iteration has been optimum till now!)')
            else:
                if rating_mat_te is not None:
                    a_mat_opt = a_mat_tt
                else:
                    a_mat_opt = a_mat.copy()
                x_mat_opt = self.upd.x_mat.copy()

        return a_mat_opt, x_mat_opt

    @staticmethod
    def calc_prediction_rmse(vm: Vandermonde, a_mat, rating_mat_te, min_val, max_val):
        rating_mat_pr = vm.predict(a_mat)

        rating_mat_pr[rating_mat_pr > max_val] = max_val
        rating_mat_pr[rating_mat_pr < min_val] = min_val

        err = rating_mat_te - rating_mat_pr

        err_not_nan = err[~np.isnan(err)]

        return np.sqrt(np.mean(err_not_nan**2))
