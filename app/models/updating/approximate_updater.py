import numpy as np
from tqdm import tqdm

from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater


class ApproximateUpdater(Updater):
    def __init__(self, x_mat_0, gamma):
        Updater.__init__(self)

        self.x_mat = x_mat_0  # [dim_x x n_item]
        self.gamma = gamma

    def update_x(self, x_mat):
        self.x_mat = x_mat

    def fit(self, vm: Vandermonde, a_mat, rating_mat, propensity_mat=None):
        n_user, n_item = rating_mat.shape

        # Init.
        x_mat_new = np.zeros(self.x_mat.shape)

        # Calc. current prediction
        rating_mat_pr = vm.predict(a_mat)

        # Loop on items
        for item in tqdm(range(n_item), desc='fit (approx)'):
            # Get the item's ratings
            s_i = rating_mat[:, item:(item + 1)]

            # Get the rated users
            observed_users_i = ~np.isnan(s_i[:, 0])

            # Filter unrated users out
            s_i_rep = np.tile(s_i[observed_users_i], reps=(1, n_item))

            # Calc error
            err = s_i_rep - rating_mat_pr[observed_users_i]

            # Get propensity scores
            if propensity_mat is None:
                n_observed_users_i = np.sum(observed_users_i)
                p_i = np.ones((n_observed_users_i, 1))*n_observed_users_i/n_user  # Normalization is unnecessary
            else:
                p_i = propensity_mat[observed_users_i, item:(item + 1)]

            # Calc weighted mse
            mse = np.sum((err**2)/p_i, axis=0)

            # Select the min error item
            item_opt = np.argmin(mse)

            # Update x_mat
            x_mat_new[:, item] = (1 - self.gamma)*self.x_mat[:, item] + self.gamma*self.x_mat[:, item_opt]

        self.x_mat = x_mat_new

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat
