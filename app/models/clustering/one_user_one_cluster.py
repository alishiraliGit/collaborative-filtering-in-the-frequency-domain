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
    def __init__(self, n_iter=1, estimate_sigma_n=False, sigma_n=None, min_alpha=-1., max_alpha=1.):
        super().__init__()

        self.debiaser = FirstOrderBiasCorrection(n_iter, estimate_sigma_n, sigma_n, min_alpha, max_alpha)

    def transform(self, vm: Vandermonde, rating_mat, verbose=False, is_test=False, **_kwargs):
        # Calc. "a" of new clusters
        a_c_mat_new, are_valid = self.calc_a_clusters(self.users_clusters, vm, rating_mat, self.n_cluster)
        assert np.all(are_valid)

        if is_test:
            # Debiasing
            for cls in tqdm(range(self.n_cluster), disable=not verbose):
                a = a_c_mat_new[:, cls]
                r = rating_mat[cls]

                a_c_mat_new[:, cls] = self.debiaser.debias(vm, a, r)

        # Save the "a" of new clusters
        self.a_c_mat = a_c_mat_new

        return self.a_c_mat, self.users_clusters

    def copy(self, _do_init):
        return OneUserOneClusterBiasCorrected(
            self.debiaser.n_iter,
            self.debiaser.estimate_sigma_n,
            self.debiaser.sigma_n,
            min_alpha=self.debiaser.min_alpha,
            max_alpha=self.debiaser.max_alpha
        )
