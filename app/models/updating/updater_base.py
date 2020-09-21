import abc

from app.models.vandermonde import Vandermonde


class Updater(abc.ABC):

    @abc.abstractmethod
    def fit(self, vm: Vandermonde, a_mat, rating_mat):
        pass

    @abc.abstractmethod
    def transform(self, vm: Vandermonde):
        pass

    def fit_transform(self, vm: Vandermonde, a_mat, rating_mat):
        self.fit(vm, a_mat, rating_mat)

        return self.transform(vm)

