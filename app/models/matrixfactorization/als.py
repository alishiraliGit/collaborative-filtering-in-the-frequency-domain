import numpy as np


class ALS:
    def __init__(self, w_u_0, w_i_0, l2_lambda):
        self.w_u = w_u_0
        self.w_i = w_i_0
        self.l2_lambda = l2_lambda
        self.b_u = np.zeros((w_u_0.shape[1],))
        self.b_i = np.zeros((w_i_0.shape[1],))

    def fit(self, rat_mat, n_alt):
        w_u = self.w_u
        w_i = self.w_i
        b_u = self.b_u
        b_i = self.b_i

        for i_alt in range(n_alt):
            w_u_new, b_u_new = self._update_row_weights(w_u, w_i, b_u, b_i, rat_mat, self.l2_lambda)
            w_i_new, b_i_new = self._update_row_weights(w_i, w_u_new, b_i, b_u_new, rat_mat.T, self.l2_lambda)

            w_u = w_u_new
            w_i = w_i_new
            b_u = b_u_new
            b_i = b_i_new

        return w_u, w_i, b_u, b_i

    def transform(self, w_u, w_i, b_u, b_i):
        self.w_u = w_u
        self.w_i = w_i
        self.b_u = b_u
        self.b_i = b_i

        return w_u, w_i, b_u, b_i

    def fit_transform(self, rat_mat, n_alt):
        w_u_new, w_i_new, b_u_new, b_i_new = self.fit(rat_mat, n_alt)

        return self.transform(w_u_new, w_i_new, b_u_new, b_i_new)

    @staticmethod
    def _update_row_weights(w_users, w_items, b_users, b_items, rat_mat, l2_lambda):
        """
        :param w_users: [d x n_row]
        :param w_items: [d x n_col]
        :param b_users: [n_row,]
        :param b_items: [n_col,]
        :param rat_mat: [n_user, n_item]
        :param l2_lambda: a number
        :return:
        """
        n_user, n_item = rat_mat.shape

        d = w_users.shape[0]
        assert w_items.shape[0] == d
        assert b_users.shape[0] == n_user
        assert b_items.shape[0] == n_item

        w_u_new = np.empty((d, n_user))
        b_u_new = np.empty((n_user,))

        for row in range(n_user):
            rated_cols_r = np.where(~np.isnan(rat_mat[row]))[0]

            s_u = rat_mat[row, rated_cols_r].reshape((-1, 1))

            w_i_u = w_items[:, rated_cols_r]

            w_i_u_aug = np.concatenate((w_i_u, np.ones((1, w_i_u.shape[1]))), axis=0)

            b_i_u = b_items[rated_cols_r].reshape((-1, 1))

            _w = np.linalg.inv(w_i_u_aug.dot(w_i_u_aug.T) + l2_lambda*np.eye(d + 1)).dot(w_i_u_aug).dot(s_u - b_i_u)

            w_u_new[:, row] = _w[:-1, 0]
            b_u_new[row] = _w[-1, 0]

        return w_u_new, b_u_new


if __name__ == '__main__':
    rating_mat = np.random.random((5, 10))

    w_user = np.random.random((3, 5))
    w_item = np.random.random((3, 10))

    als = ALS(w_u_0=w_user, w_i_0=w_item, l2_lambda=1)

    als.fit_transform(rating_mat, 10)
