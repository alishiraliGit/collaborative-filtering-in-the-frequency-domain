import abc

from app.models.vandermonde import Vandermonde


class Updater(abc.ABC):

    @abc.abstractmethod
    def fit(self, vm: Vandermonde, a_mat, rating_mat, propensity_mat=None):
        pass

    @abc.abstractmethod
    def transform(self, vm: Vandermonde):
        pass

    def fit_transform(self, vm: Vandermonde, a_mat, rating_mat, propensity_mat=None):
        self.fit(vm, a_mat, rating_mat, propensity_mat=propensity_mat)

        return self.transform(vm)

    @abc.abstractmethod
    def update_x(self, x):
        pass
