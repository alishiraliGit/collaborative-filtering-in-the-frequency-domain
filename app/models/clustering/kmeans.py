import numpy as np
from tqdm import tqdm

from app.models.clustering.clustering_base import Clustering
from app.models.vandermonde import Vandermonde, RegularizationType
from app.models.debiasing.biascorrection import FirstOrderBiasCorrection
from app.utils.mat_ops import vectorize_rows


class KMeansOneIter(Clustering):
    def __init__(self, n_cluster, a_c_mat_0):
        Clustering.__init__(self)

        self.n_cluster = n_cluster
        self.a_c_mat = a_c_mat_0  # [dim_a, n_cluster]

        self.users_clusters = None

    def fit(self, vm: Vandermonde, rating_mat):
        n_user, _ = rating_mat.shape

        # Do the 1-iteration clustering
        self.users_clusters = np.zeros((n_user, ), dtype=int)
        for user in range(n_user):
            s_u = vectorize_rows(user, rating_mat)

            v_u_mat = vm.get_v_users(user, rating_mat)

            err = v_u_mat.T.dot(self.a_c_mat) - np.tile(s_u, (1, self.n_cluster))

            mse = np.sum(err**2, axis=0)

            self.users_clusters[user] = np.argmin(mse)

        return

    def transform(self, vm: Vandermonde, rating_mat, **_kwargs):
        # Calc. "a" of new clusters
        a_c_mat_new, are_valid = Clustering.calc_a_clusters(self.users_clusters, vm, rating_mat, self.n_cluster)

        # Replace empty clusters with previous values
        a_c_mat_new[:, ~are_valid] = self.a_c_mat[:, ~are_valid]

        # Save the "a" of new clusters in the object
        self.a_c_mat = a_c_mat_new

        return self.a_c_mat, self.users_clusters

    def copy(self, do_init):
        if do_init:
            a_c_mat = np.random.normal(loc=0, scale=np.std(self.a_c_mat), size=self.a_c_mat.shape)
        else:
            a_c_mat = self.a_c_mat.copy()

        cls = KMeansOneIter(self.n_cluster, a_c_mat)

        return cls


class KMeans(Clustering):
    def __init__(self, n_cluster, n_iter, a_c_mat_0):
        Clustering.__init__(self)

        self.n_cluster = n_cluster
        self.n_iter = n_iter

        self.a_c_mat = a_c_mat_0  # [dim_a, n_cluster]
        self.users_clusters = None

    def fit(self, vm: Vandermonde, rating_mat):
        kmeans_one_iter = KMeansOneIter(n_cluster=self.n_cluster, a_c_mat_0=self.a_c_mat)

        for it in range(self.n_iter):
            self.a_c_mat, self.users_clusters = kmeans_one_iter.fit_transform(vm, rating_mat)

        return

    def transform(self, vm: Vandermonde, rating_mat, **_kwargs):
        return self.a_c_mat, self.users_clusters

    def copy(self, do_init):
        if do_init:
            a_c_mat = np.random.normal(loc=0, scale=np.std(self.a_c_mat), size=self.a_c_mat.shape)
        else:
            a_c_mat = self.a_c_mat.copy()

        cls = KMeans(self.n_cluster, self.n_iter, a_c_mat)

        return cls


class KMeansBiasCorrected(KMeansOneIter):
    def __init__(self, n_cluster, a_c_mat_0,
                 n_iter=1, estimate_sigma_n=False, sigma_n=None, min_alpha=-1., max_alpha=1.):
        super().__init__(n_cluster, a_c_mat_0)

        self.debiaser = FirstOrderBiasCorrection(n_iter, estimate_sigma_n, sigma_n, min_alpha, max_alpha)

    def transform(self, vm: Vandermonde, rating_mat, verbose=False, is_test=False, **_kwargs):
        # Calc. "a" of new clusters
        a_c_mat_new, are_valid = \
            Clustering.calc_a_clusters(self.users_clusters, vm, rating_mat, self.n_cluster, verbose=verbose)

        # Replace empty clusters with previous values
        a_c_mat_new[:, ~are_valid] = self.a_c_mat[:, ~are_valid]

        # Save the "a" of new clusters in the object
        if not is_test:
            self.a_c_mat = a_c_mat_new
        else:
            # Debiasing
            for cls in tqdm(range(self.n_cluster), disable=not verbose, desc='KMeansBC:transform:debiasing'):
                if not are_valid[cls]:
                    continue

                a = a_c_mat_new[:, cls]

                users_c = np.argwhere(self.users_clusters == cls)[:, 0]
                r_cls_mat = rating_mat[users_c]

                # Update c_mat if required
                if vm.reg_type == RegularizationType.POST_MAX_SNR:
                    vm.update_c_mat(a, is_test=True)

                    vm.c_mat_is_updated = True

                    a = vm.calc_a_users(users_c, rating_mat)[:, 0]

                a_c_mat_new[:, cls] = self.debiaser.debias(vm, a, r_cls_mat)

            vm.c_mat_is_updated = False

        return a_c_mat_new, self.users_clusters

    def copy(self, do_init):
        if do_init:
            a_c_mat = np.random.normal(loc=0, scale=np.std(self.a_c_mat), size=self.a_c_mat.shape)
        else:
            a_c_mat = self.a_c_mat.copy()

        cls = KMeansBiasCorrected(
            self.n_cluster,
            a_c_mat,
            self.debiaser.n_iter,
            self.debiaser.estimate_sigma_n,
            self.debiaser.sigma_n,
            min_alpha=self.debiaser.min_alpha,
            max_alpha=self.debiaser.max_alpha)
        return cls
