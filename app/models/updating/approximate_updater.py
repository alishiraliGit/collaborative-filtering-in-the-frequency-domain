import numpy as np
from tqdm import tqdm

from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater


class ApproximateUpdater(Updater):
    def __init__(self, x_mat_0, gamma):
        Updater.__init__(self)

        self.x_mat = x_mat_0  # [dim_x x n_item]
        self.gamma = gamma

    def fit(self, vm: Vandermonde, a_mat, rating_mat):
        _n_user, n_item = rating_mat.shape

        # Init.
        x_mat_new = np.zeros(self.x_mat.shape)

        # Calc. current prediction
        rating_mat_pr = vm.predict(a_mat)

        # Loop on items
        for item in tqdm(range(n_item), desc='fit (approx)'):
            s_i = rating_mat[:, item:item+1]

            observed_users_i = ~np.isnan(s_i[:, 0])

            s_i_rep = np.tile(s_i[observed_users_i], reps=(1, n_item))

            err = s_i_rep - rating_mat_pr[observed_users_i]

            mse = np.sum(err**2, axis=0)

            item_opt = np.argmin(mse)

            x_mat_new[:, item] = (1 - self.gamma)*self.x_mat[:, item] + self.gamma*self.x_mat[:, item_opt]

        self.x_mat = x_mat_new

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat
