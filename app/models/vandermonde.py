import abc
import numpy as np

from app.utils.mat_ops import vectorize_rows


class VandermondeType:
    COS = 'cos'
    REAL = 'real'
    COS_MULT = 'cos_mult'


class Vandermonde(abc.ABC):
    def __init__(self, dim_x, dim_a, m, l2_lambda, vm_type):
        self.dim_x = dim_x
        self.dim_a = dim_a
        self.m = m
        self.l2_lambda = l2_lambda

        self.v_mult = None  # [dim_a x dim_x]
        self.v_mat = None  # [dim_a x n_item]

        self.vm_type = vm_type
        pass

    @staticmethod
    def get_instance(dim_x, m, l2_lambda, vm_type: VandermondeType):
        if vm_type == VandermondeType.COS:
            return VandermondeCos(dim_x, m, l2_lambda)
        elif vm_type == VandermondeType.REAL:
            return VandermondeReal(dim_x, m, l2_lambda)
        elif vm_type == VandermondeType.COS_MULT:
            return VandermondeCosMult(dim_x, m, l2_lambda)

    def get_v_users(self, users, rating_mat):
        if not isinstance(users, (list, np.ndarray)):  # if users is a number only
            users = [users]

        observed_indices = np.argwhere(~np.isnan(rating_mat[users, :]))  # order='C'
        return self.v_mat[:, observed_indices[:, 1]]

    def calc_a_users(self, users, rating_mat):
        v_u_mat = self.get_v_users(users, rating_mat)

        s_u = vectorize_rows(users, rating_mat)

        e1 = np.zeros((self.dim_a, self.dim_a))
        e1[0, 0] = 1  # Do not regularize a_0
        a_u = np.linalg.inv(v_u_mat.dot(v_u_mat.T) +
                            self.l2_lambda*np.eye(self.dim_a) - self.l2_lambda*e1).dot(v_u_mat).dot(s_u)

        return a_u

    def predict(self, a_mat):
        """
        :param a_mat: [dim_a x n_user]
        :return: s_pr: [n_user x n_item]
        """
        s_pr = a_mat.T.dot(self.v_mat)
        return s_pr

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def transform(self, x_mat):
        pass

    def copy(self):
        vm = self.__class__(self.dim_x, self.m, self.l2_lambda)

        vm.v_mult = self.v_mult.copy()

        return vm


class VandermondeCos(Vandermonde):
    def __init__(self, dim_x, m, l2_lambda):
        dim_a = self.__calc_dim_a(dim_x, m)
        Vandermonde.__init__(self, dim_x, dim_a, m, l2_lambda, vm_type=VandermondeType.COS)

    @staticmethod
    def __calc_dim_a(dim_x, m):
        return (m + 1) ** dim_x

    def fit(self):
        """
        sets v_mult, [dim_a x dim_x]
        :return: None
        """
        v_mult_row = np.zeros((self.dim_x,))
        v_mult = np.zeros((self.dim_a, self.dim_x))

        for i_row in range(1, self.dim_a):
            v_mult_row[0] += 1

            for i_dim in range(self.dim_x - 1):
                if v_mult_row[i_dim] >= (self.m + 1):
                    v_mult_row[i_dim + 1] += v_mult_row[i_dim] // (self.m + 1)
                    v_mult_row[i_dim] %= (self.m + 1)

            v_mult[i_row, :] = v_mult_row

        self.v_mult = v_mult

        return

    def transform(self, x_mat):
        """
        :param x_mat: [dim_x x n_item]
        :return: v_mat: [dim_a x n_item]
        """
        self.v_mat = np.cos(np.pi*self.v_mult.dot(x_mat))

        return self.v_mat


class VandermondeReal(Vandermonde):
    def __init__(self, dim_x, m, l2_lambda):
        dim_a = self.__calc_dim_a(dim_x, m)
        Vandermonde.__init__(self, dim_x, dim_a, m, l2_lambda, vm_type=VandermondeType.REAL)

    @staticmethod
    def __calc_dim_a(dim_x, m):
        return 2*((m + 1) ** dim_x)

    def fit(self):
        """
        sets v_mult, [dim_a x dim_x]
        :return: None
        """
        v_mult_row = np.zeros((self.dim_x,))
        v_mult = np.zeros((int(self.dim_a/2), self.dim_x))

        for i_row in range(1, int(self.dim_a/2)):
            v_mult_row[0] += 1

            for i_dim in range(self.dim_x - 1):
                if v_mult_row[i_dim] >= (self.m + 1):
                    v_mult_row[i_dim + 1] += v_mult_row[i_dim] // (self.m + 1)
                    v_mult_row[i_dim] %= (self.m + 1)

            v_mult[i_row, :] = v_mult_row

        self.v_mult = v_mult

        return

    def transform(self, x_mat):
        """
        :param x_mat: [dim_x x n_item]
        :return: v_mat: [dim_a x n_item]
        """
        self.v_mat = np.concatenate((np.cos(np.pi*self.v_mult.dot(x_mat)), np.sin(np.pi*self.v_mult.dot(x_mat))),
                                    axis=0)

        return self.v_mat


class VandermondeCosMult(Vandermonde):
    def __init__(self, dim_x, m, l2_lambda):
        dim_a = self.__calc_dim_a(dim_x, m)
        Vandermonde.__init__(self, dim_x, dim_a, m, l2_lambda, vm_type=VandermondeType.COS_MULT)

    @staticmethod
    def __calc_dim_a(dim_x, m):
        return (m + 1) ** dim_x

    def fit(self):
        """
        sets v_mult, [dim_a x dim_x]
        :return: None
        """
        v_mult_row = np.zeros((self.dim_x,))
        v_mult = np.zeros((self.dim_a, self.dim_x))

        for i_row in range(1, self.dim_a):
            v_mult_row[0] += 1

            for i_dim in range(self.dim_x - 1):
                if v_mult_row[i_dim] >= (self.m + 1):
                    v_mult_row[i_dim + 1] += v_mult_row[i_dim] // (self.m + 1)
                    v_mult_row[i_dim] %= (self.m + 1)

            v_mult[i_row, :] = v_mult_row

        self.v_mult = v_mult

        return

    def transform(self, x_mat):
        """
        :param x_mat: [dim_x x n_item]
        :return: v_mat: [dim_a x n_item]
        """
        n_item = x_mat.shape[1]

        self.v_mat = np.zeros((self.dim_a, n_item))

        for item in range(n_item):
            x_i_diag = np.diag(x_mat[:, item])

            self.v_mat[:, item] = np.prod(np.cos(np.pi*self.v_mult.dot(x_i_diag)), axis=1)

        return self.v_mat
