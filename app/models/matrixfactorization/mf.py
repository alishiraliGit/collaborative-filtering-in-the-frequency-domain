import numpy as np

from .als import ALS
from ..logger import Logger


class MatrixFactorization:
    def __init__(self, w_u_0, w_i_0, l2_lambda):
        self.als = ALS(w_u_0=w_u_0, w_i_0=w_i_0, l2_lambda=l2_lambda)

    def run(self, rat_mat_tr, rat_mat_va, n_alt, min_val, max_val,
            rat_mat_te=None,
            logger: Logger=None):

            for alt in range(n_alt):
                w_u, w_i, b_u, b_i = self.als.fit_transform(rat_mat_tr, n_alt=1)

                if isinstance(logger, Logger):
                    rmse_tr = self.calc_prediction_rmse(w_u, w_i, b_u, b_i, rat_mat_tr, min_val, max_val)
                    rmse_va = self.calc_prediction_rmse(w_u, w_i, b_u, b_i, rat_mat_va, min_val, max_val)
                    rmse_te = self.calc_prediction_rmse(w_u, w_i, b_u, b_i, rat_mat_te, min_val, max_val)

                    logger.log(rmse_tr, rmse_va, rmse_te)

    @staticmethod
    def predict(w_u, w_i, b_u, b_i, min_val, max_val):
        n_user = w_u.shape[1]
        n_item = w_i.shape[1]
        rat_mat_pr = w_u.T.dot(w_i) + np.tile(b_u.reshape((-1, 1)), reps=(1, n_item)) + np.tile(b_i, reps=(n_user, 1))

        rat_mat_pr[rat_mat_pr > max_val] = max_val
        rat_mat_pr[rat_mat_pr < min_val] = min_val

        return rat_mat_pr

    @staticmethod
    def calc_prediction_rmse(w_u, w_i, b_u, b_i, rat_mat_true, min_val, max_val):
        rat_mat_pr = MatrixFactorization.predict(w_u, w_i, b_u, b_i, min_val, max_val)

        err = rat_mat_true - rat_mat_pr

        err_not_nan = err[~np.isnan(err)]

        return np.sqrt(np.mean(err_not_nan ** 2))
