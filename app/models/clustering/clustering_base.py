import abc
import numpy as np
from tqdm import tqdm

from app.models.vandermonde import Vandermonde


class Clustering(abc.ABC):

    @abc.abstractmethod
    def fit(self, vm: Vandermonde, rating_mat, propensity_mat=None):
        pass

    @abc.abstractmethod
    def transform(self, vm: Vandermonde, rating_mat, propensity_mat=None, **kwargs):
        pass

    def fit_transform(self, vm: Vandermonde, rating_mat, propensity_mat=None, **kwargs):
        self.fit(vm, rating_mat, propensity_mat)

        return self.transform(vm, rating_mat, propensity_mat, **kwargs)

    @abc.abstractmethod
    def copy(self, do_init):
        pass

    @staticmethod
    def calc_a_clusters(users_clusters, vm: Vandermonde, rating_mat, n_cluster, propensity_mat=None, verbose=False):
        a_c = np.zeros((vm.dim_a, n_cluster))
        are_valid = np.ones((n_cluster, ), dtype=bool)

        for cluster in tqdm(range(n_cluster), disable=not verbose, desc='calc_a_clusters'):
            users_c = np.argwhere(users_clusters == cluster)[:, 0]

            if len(users_c) == 0:
                are_valid[cluster] = False
                continue

            a_c[:, cluster] = vm.calc_a_users(users_c, rating_mat, propensity_mat)[:, 0]

        return a_c, are_valid
