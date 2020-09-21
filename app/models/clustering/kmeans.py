import numpy as np

from app.models.clustering.clustering_base import Clustering
from app.models.vandermonde import Vandermonde
from app.utils.mat_ops import vectorize_rows


class KMeans(Clustering):
    def __init__(self, n_cluster, a_c_mat_0, l2_lambda):
        Clustering.__init__(self)

        self.n_cluster = n_cluster
        self.a_c_mat = a_c_mat_0  # [dim_a, n_cluster]
        self.l2_lambda = l2_lambda

        self.users_clusters = None

    def fit(self, vm: Vandermonde, rating_mat):
        n_user, _ = rating_mat.shape

        # Do the 1-iteration clustering
        self.users_clusters = np.zeros((n_user, ), dtype=int)
        for user in range(n_user):
            s_u = vectorize_rows(user, rating_mat)

            v_u_mat = vm.get_v_users(user, rating_mat)

            err = v_u_mat.T.dot(self.a_c_mat) - np.tile(s_u, (1, self.n_cluster))

            mse = np.sum(err**2, axis=0) + self.l2_lambda*np.sum(self.a_c_mat**2, axis=0)

            self.users_clusters[user] = np.argmin(mse)

        return

    def transform(self, vm: Vandermonde, rating_mat):
        # Calc. "a" of new clusters
        a_c_mat_new, are_valid = Clustering.calc_a_clusters(self.users_clusters, vm, rating_mat, self.n_cluster)

        # Replace empty clusters with previous values
        a_c_mat_new[:, ~are_valid] = self.a_c_mat[:, ~are_valid]

        # Save the "a" of new clusters in the object
        self.a_c_mat = a_c_mat_new

        return self.a_c_mat, self.users_clusters

    def copy(self):
        return KMeans(self.n_cluster, self.a_c_mat.copy(), self.l2_lambda)
