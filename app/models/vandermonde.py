import numpy as np

from app.utils.mat_ops import vectorize_rows


class Vandermonde:
    def __init__(self, dim_x, m, l2_lambda):
        self.dim_x = dim_x
        self.m = m
        self.dim_a = self.__calc_dim_a(dim_x, m)
        self.l2_lambda = l2_lambda

        self.v_mult = None  # [dim_a x dim_x]
        self.v_mat = None  # [dim_a x n_item]
        pass

    @staticmethod
    def __calc_dim_a(dim_x, m):
        return (m + 1) ** dim_x

    def get_v_users(self, users, rating_mat):
        if not isinstance(users, list):  # if users is a number only
            users = [users]

        observed_indices = np.argwhere(~np.isnan(rating_mat[users]))  # order='C'
        return self.v_mat[:, observed_indices[:, 1]]

    def calc_a_users(self, users, rating_mat):
        v_u_mat = self.get_v_users(users, rating_mat)

        s_u = vectorize_rows(users, rating_mat)

        a_u = np.linalg.inv(v_u_mat.dot(v_u_mat.T) + self.l2_lambda*np.eye(self.dim_a)).dot(v_u_mat).dot(s_u)

        return a_u

    def predict(self, a_mat):
        """
        :param a_mat: [dim_a x n_user]
        :return: s_pr: [n_user x n_item]
        """
        s_pr = a_mat.T.dot(self.v_mat)
        return s_pr

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

    def copy(self):
        vm = Vandermonde(self.dim_x, self.m, self.l2_lambda)

        vm.v_mult = self.v_mult.copy()
        vm.v_mat = self.v_mat.copy()

        return vm
