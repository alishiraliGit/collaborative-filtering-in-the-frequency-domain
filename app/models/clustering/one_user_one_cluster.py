import numpy as np
from tqdm import tqdm

from app.models.clustering.clustering_base import Clustering
from app.models.vandermonde import Vandermonde
from app.models.debiasing.biascorrection import FirstOrderBiasCorrection


class OneUserOneCluster(Clustering):
    def __init__(self):
        super().__init__()

        self.n_cluster = None
        self.users_clusters = None
        self.a_c_mat = None

    def fit(self, vm: Vandermonde, rating_mat):
        n_user, _ = rating_mat.shape

        self.n_cluster = n_user
        self.users_clusters = np.array(range(n_user))

        return self

    def transform(self, vm: Vandermonde, rating_mat, **_kwargs):
        # Calc. "a" of new clusters
        a_c_mat_new, are_valid = self.calc_a_clusters(self.users_clusters, vm, rating_mat, self.n_cluster)

        assert np.all(are_valid)

        # Save the "a" of new clusters
        self.a_c_mat = a_c_mat_new

        return self.a_c_mat, self.users_clusters

    def copy(self, _do_init):
        return OneUserOneCluster()


class OneUserOneClusterBiasCorrected(OneUserOneCluster):
    def __init__(self, sigma_n, n_iter=1):
        super().__init__()

        self.debiaser = FirstOrderBiasCorrection(sigma_n, n_iter)

    def transform(self, vm: Vandermonde, rating_mat, verbose=False, **_kwargs):
        # Calc. "a" of new clusters
        a_c_mat_new, are_valid = self.calc_a_clusters(self.users_clusters, vm, rating_mat, self.n_cluster)
        assert np.all(are_valid)

        # Debiasing
        for cls in tqdm(range(self.n_cluster), disable=not verbose):
            a = a_c_mat_new[:, cls]
            r = rating_mat[cls]

            a_c_mat_new[:, cls] = self.debiaser.debias(vm, a, r)

        # Save the "a" of new clusters
        self.a_c_mat = a_c_mat_new

        return self.a_c_mat, self.users_clusters

    def copy(self, _do_init):
        return OneUserOneClusterBiasCorrected(self.debiaser.sigma_n, self.debiaser.n_iter)

