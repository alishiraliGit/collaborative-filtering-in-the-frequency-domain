import numpy as np

from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater


class Randomizer(Updater):
    def __init__(self, upd: Updater, n_sub_user, n_rept):
        Updater.__init__(self)

        self.upd = upd
        self.n_sub_user = n_sub_user
        self.n_rept = n_rept

        self.x_mat = None

    def fit(self, vm: Vandermonde, a_mat, rating_mat):
        n_user, _n_item = rating_mat.shape

        for rept in range(self.n_rept):
            users = np.random.permutation(n_user)[:self.n_sub_user]

            self.x_mat = self.upd.fit_transform(vm, a_mat[:, users], rating_mat[users, :])

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat
