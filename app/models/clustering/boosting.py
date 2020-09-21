

from app.models.vandermonde import Vandermonde
from app.models.clustering.kmeans import Clustering


class Boosting(Clustering):
    def __init__(self, cls: Clustering, n_learner, n_iter_cls):
        self.cls = cls
        self.n_learner = n_learner
        self.n_iter_cls = n_iter_cls

        self.a_c_mat = None
        self.users_clusters = None

    def fit(self, vm: Vandermonde, rating_mat):
        rating_mat_res = rating_mat.copy()

        for learner in range(self.n_learner):
            cls_learner = self.cls.copy()

            for _ in range(self.n_iter_cls - 1):
                cls_learner.fit_transform(vm, rating_mat_res)

                self.a_c_mat, self.users_clusters = cls_learner.fit_transform(vm, rating_mat_res)

            a_mat = self.a_c_mat[:, self.users_clusters]

            rating_mat_pr = vm.predict(a_mat)

            rating_mat_res -= rating_mat_pr

        return

    def transform(self, vm: Vandermonde, rating_mat):
        return self.a_c_mat, self.users_clusters

    def copy(self):
        return Boosting(self.cls.copy(), self.n_learner, self.n_iter_cls)
