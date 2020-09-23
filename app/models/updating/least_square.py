import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater
from app.utils.mat_ops import vectorize, unvectorize


class LeastSquare(Updater):
    def __init__(self, x_mat_0, max_nfev):
        Updater.__init__(self)

        self.x_mat = x_mat_0  # [dim_x x n_item]
        self.max_nfev = max_nfev

    def fit(self, vm: Vandermonde, a_mat, rating_mat):
        x_0 = vectorize(self.x_mat, 'C')

        ls = least_squares(fun=LeastSquare.loss,
                           jac=LeastSquare.jac,
                           args=(vm, a_mat, rating_mat),
                           x0=x_0,
                           bounds=(0, 1),
                           max_nfev=self.max_nfev,
                           verbose=2)

        _n_user, n_item = rating_mat.shape
        self.x_mat = unvectorize(ls.x, n_item, 'C')

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat

    @staticmethod
    def loss(x_vec, vm_org: Vandermonde, a_mat, rating_mat):
        # Copy vm
        vm = vm_org.copy()

        # Reshape x to matrix
        _n_user, n_item = rating_mat.shape
        x_mat = unvectorize(x_vec, n_item, 'C')

        # Use x_mat in vm
        vm.transform(x_mat)

        # Predict rating
        rating_mat_pr = vm.predict(a_mat)

        # Calculate loss
        err = rating_mat_pr - rating_mat

        err_not_nan = err[~np.isnan(err)]

        return err_not_nan

    @staticmethod
    def jac_old(x_vec, vm: Vandermonde, a_mat, rating_mat):
        _n_user, n_item = rating_mat.shape

        x_mat = unvectorize(x_vec, n_item, 'F')

        dim_x = x_mat.shape[0]

        deriv = np.zeros(x_mat.shape)

        for it in range(n_item):
            v_i = vm.v_mat[:, it:it+1]  # [dim_a x 1]

            x_i = x_mat[:, it:it+1]  # [dim_x x 1]

            rated_users = np.argwhere(~np.isnan(rating_mat[:, it]))[:, 0]

            s_i = rating_mat[rated_users, it:it+1]  # [n_users_i x 1]
            a_mat_i = a_mat[:, rated_users]  # [dim_a x n_users_i]

            for dim in range(dim_x):
                a_mat_i_re = np.pi * a_mat_i * np.tile(vm.v_mult[:, dim:dim+1], reps=(1, len(rated_users)))

                deriv[dim, it] = -(v_i.T.dot(a_mat_i) - s_i.T).dot(a_mat_i_re.T.dot(np.sin(np.pi*vm.v_mult.dot(x_i))))

        deriv_vec = vectorize(deriv, 'F').reshape((1, -1), 'F')

        return deriv_vec

    @staticmethod
    def jac(x_vec, vm: Vandermonde, a_mat, rating_mat):
        """
        :param x_vec:
        :param vm:
        :param a_mat: [dim_a x n_user]
        :param rating_mat:
        :return:
        """
        n_user, n_item = rating_mat.shape

        # Init. the user and item number of each sample
        u_mat = np.tile(np.array(range(n_user)).reshape((-1, 1)), reps=(1, n_item))
        i_mat = np.tile(np.array(range(n_item)).reshape((1, -1)), reps=(n_user, 1))

        u_s = u_mat[~np.isnan(rating_mat)]
        i_s = i_mat[~np.isnan(rating_mat)]

        n_s = len(u_s)

        # Init. deriv
        deriv = np.zeros((n_s, len(x_vec)))

        #
        x_mat = unvectorize(x_vec, n_item, 'C')

        dim_x = x_mat.shape[0]

        #
        sin_mat = np.sin(np.pi * vm.v_mult.dot(x_mat))

        for dim in range(dim_x):
            d_v_mat = -np.pi * sin_mat * np.tile(vm.v_mult[:, dim:dim+1], reps=(1, n_item))  # [dim_a x n_item]

            deriv_d = np.sum(a_mat[:, u_s] * d_v_mat[:, i_s], axis=0)  # [n_s, ]

            deriv[range(n_s), i_s + dim*n_item] = deriv_d

        return csr_matrix(deriv)
