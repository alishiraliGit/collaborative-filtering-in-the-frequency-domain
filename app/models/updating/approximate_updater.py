import numpy as np
from tqdm import tqdm

from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater

rng = np.random.default_rng(1)


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
                p_i = np.ones((n_observed_users_i, 1))  # Normalization is unnecessary
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


class ApproximateExpandedUpdater(Updater):
    def __init__(self, x_mat_0, gamma):
        Updater.__init__(self)

        self.x_mat = x_mat_0  # [dim_x x n_item]
        self.gamma = gamma

    def update_x(self, x_mat):
        self.x_mat = x_mat

    def expand_x(self, factor, noise_w):
        x_mat_expanded = np.tile(self.x_mat, reps=(1, factor))

        # Add noisy versions
        dim_x, n_item = self.x_mat.shape
        x_mat_expanded[:, n_item:] += (rng.random((dim_x, (factor - 1)*n_item)) - 0.5)*noise_w
        x_mat_expanded %= 1

        return x_mat_expanded

    def fit(self, vm: Vandermonde, a_mat, rating_mat, propensity_mat=None):
        n_user, n_item = rating_mat.shape

        # Init.
        x_mat_new = np.zeros(self.x_mat.shape)
        # TODO: Hard-coded
        expansion_factor = 10
        x_mat_expanded = self.expand_x(factor=expansion_factor, noise_w=0.1)
        vm_expanded = vm.copy()
        vm_expanded.transform(x_mat_expanded)

        # Calc. current prediction
        rating_mat_pr = vm_expanded.predict(a_mat)

        # Loop on items
        for item in tqdm(range(n_item), desc='fit (approx)'):
            # Get the item's ratings
            s_i = rating_mat[:, item:(item + 1)]

            # Get the rated users
            observed_users_i = ~np.isnan(s_i[:, 0])

            # Filter unrated users out
            s_i_rep = np.tile(s_i[observed_users_i], reps=(1, n_item*expansion_factor))

            # Calc error
            err = s_i_rep - rating_mat_pr[observed_users_i]

            # Get propensity scores
            if propensity_mat is None:
                n_observed_users_i = np.sum(observed_users_i)
                p_i = np.ones((n_observed_users_i, 1))  # Normalization is unnecessary
            else:
                p_i = propensity_mat[observed_users_i, item:(item + 1)]

            # Calc weighted mse
            mse = np.sum((err**2)/p_i, axis=0)

            # Select the min error item
            item_opt = np.argmin(mse)

            # Update x_mat
            x_mat_new[:, item] = (1 - self.gamma)*x_mat_expanded[:, item] + self.gamma*x_mat_expanded[:, item_opt]

        self.x_mat = x_mat_new

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat
