import numpy as np

from app.models.vandermonde import Vandermonde
from app.models.clustering.clustering_base import Clustering
from app.models.updating.updater_base import Updater
from app.models.logger import Logger


class Alternate:
    def __init__(self, cls: Clustering, upd: Updater):
        self.cls = cls
        self.upd = upd

    def run(self, vm: Vandermonde, rating_mat_tr, rating_mat_va, n_iter, min_val, max_val,
            rating_mat_te=None,
            logger: Logger=None):

        a_mat_opt = None
        for it in range(n_iter):
            # Do clustering
            a_c_mat, users_clusters = self.cls.fit_transform(vm, rating_mat_tr)
            a_mat = a_c_mat[:, users_clusters]

            if rating_mat_te is not None:
                a_c_mat_tt, users_clusters_tt = self.cls.fit_transform(vm, rating_mat_tr, is_test=True)
                a_mat_tt = a_c_mat_tt[:, users_clusters_tt]

            # Do updating
            self.upd.fit_transform(vm, a_mat, rating_mat_tr)

            # Calc rmse
            rmse_tr = self.calc_prediction_rmse(vm, a_mat, rating_mat_tr, min_val, max_val)
            rmse_va = self.calc_prediction_rmse(vm, a_mat, rating_mat_va, min_val, max_val)
            if rating_mat_te is not None:
                rmse_te = self.calc_prediction_rmse(vm, a_mat_tt, rating_mat_te, min_val, max_val)
            else:
                rmse_te = np.NaN

            if isinstance(logger, Logger):
                logger.log(rmse_tr, rmse_va, rmse_te)

                if rmse_va > np.min(logger.rmse_va):
                    # ToDo
                    continue

                if rating_mat_te is not None:
                    a_mat_opt = a_mat_tt
                else:
                    a_mat_opt = a_mat
                print('(this iteration has been optimum till now!)')
            else:
                if rating_mat_te is not None:
                    a_mat_opt = a_mat_tt
                else:
                    a_mat_opt = a_mat

        return a_mat_opt

    @staticmethod
    def calc_prediction_rmse(vm: Vandermonde, a_mat, rating_mat_te, min_val, max_val):
        rating_mat_pr = vm.predict(a_mat)

        rating_mat_pr[rating_mat_pr > max_val] = max_val
        rating_mat_pr[rating_mat_pr < min_val] = min_val

        err = rating_mat_te - rating_mat_pr

        err_not_nan = err[~np.isnan(err)]

        return np.sqrt(np.mean(err_not_nan**2))
