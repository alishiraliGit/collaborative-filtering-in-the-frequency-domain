import numpy as np

from app.models.vandermonde import Vandermonde
from app.models.clustering.kmeans import Clustering


class Boosting(Clustering):
    def __init__(self, cls: Clustering, n_learner, n_iter_cls):
        self.cls = cls
        self.n_learner = n_learner
        self.n_iter_cls = n_iter_cls

        self.a_mat = None

    def fit(self, vm: Vandermonde, rating_mat):
        n_user, _ = rating_mat.shape

        # Init. "rating mat of residuals" and "a_mat"
        rating_mat_res = rating_mat.copy()
        self.a_mat = np.zeros((vm.dim_a, n_user))

        for learner in range(self.n_learner):
            # Do clustering multiple of times with "rating_mat_res"
            cls_learner = self.cls.copy(do_init=True)

            for _ in range(self.n_iter_cls - 1):
                cls_learner.fit_transform(vm, rating_mat_res)
            a_c_mat, users_clusters = cls_learner.fit_transform(vm, rating_mat_res)

            a_mat_learner = a_c_mat[:, users_clusters]

            # Update "a_mat"
            self.a_mat += a_mat_learner

            # Update "rating_mat"
            rating_mat_pr = vm.predict(a_mat_learner)
            rating_mat_res -= rating_mat_pr

        return

    def transform(self, vm: Vandermonde, rating_mat):
        n_user = self.a_mat.shape[1]
        pseudo_users_clusters = list(range(n_user))
        return self.a_mat, pseudo_users_clusters

    def copy(self, do_init):
        return Boosting(self.cls.copy(do_init), self.n_learner, self.n_iter_cls)
