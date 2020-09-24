from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater


class MultiUpdaterWrapper(Updater):
    def __init__(self, upds):
        Updater.__init__(self)

        self.upds = upds

        self.x_mat = upds[0].x_mat

    def fit(self, vm: Vandermonde, a_mat, rating_mat):
        vm.transform(self.x_mat)

        for upd in self.upds:

            upd.x_mat = self.x_mat

            self.x_mat = upd.fit_transform(vm, a_mat, rating_mat)

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat
