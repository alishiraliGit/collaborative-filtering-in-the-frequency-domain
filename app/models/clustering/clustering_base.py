import abc
import numpy as np

from app.models.vandermonde import Vandermonde


class Clustering(abc.ABC):

    @abc.abstractmethod
    def fit(self, vm: Vandermonde, rating_mat):
        pass

    @abc.abstractmethod
    def transform(self, vm: Vandermonde, rating_mat, **kwargs):
        pass

    def fit_transform(self, vm: Vandermonde, rating_mat, **kwargs):
        self.fit(vm, rating_mat)

        return self.transform(vm, rating_mat, **kwargs)

    @abc.abstractmethod
    def copy(self, do_init):
        pass

    @staticmethod
    def calc_a_clusters(users_clusters, vm: Vandermonde, rating_mat, n_cluster):
        a_c = np.zeros((vm.dim_a, n_cluster))
        are_valid = np.ones((n_cluster, ), dtype=bool)

        for cluster in range(n_cluster):
            users_c = np.argwhere(users_clusters == cluster)[:, 0]

            if len(users_c) == 0:
                are_valid[cluster] = False
                continue

            a_c[:, cluster] = vm.calc_a_users(users_c, rating_mat)[:, 0]

        return a_c, are_valid
