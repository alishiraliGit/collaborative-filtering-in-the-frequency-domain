import numpy as np

from app.models.clustering.clustering_base import Clustering
from app.models.vandermonde import Vandermonde


class OneUserOneCluster(Clustering):
    def __init__(self, n_cluster, a_c_mat_0):
        Clustering.__init__(self)

        self.n_cluster = n_cluster
        self.a_c_mat = a_c_mat_0  # [dim_a, n_cluster]

        self.users_clusters = None

    def fit(self, vm: Vandermonde, rating_mat):
        n_user, _ = rating_mat.shape

        # Do the 1-iteration clustering
        self.users_clusters = np.array(range(n_user))

        return

    def transform(self, vm: Vandermonde, rating_mat):
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

        cls = OneUserOneCluster(self.n_cluster, a_c_mat)

        return cls
