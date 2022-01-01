import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class OptimumRegularization:
    def __init__(self, bound=(0, 1)):
        self.bound = bound

    @staticmethod
    def fun_a_y_a(a, y_mat):
        """
        a^T (K + C)^-2 a
        :param a: [dim_a x 1]
        :param y_mat: [dim_a x dim_a]
        :return: cost
        """
        return a.T.dot(y_mat.dot(a))

    @staticmethod
    def jac_a_y_a_k(a, y_mat, k_mat, c_mat, k):
        dim_a = a.shape[0]

        c_kk_mat = np.zeros((dim_a, dim_a))
        c_kk_mat[k, k] = c_mat[k, k]

        k_k_mat = np.zeros((dim_a, dim_a))
        k_k_mat[k] = k_mat[k]

        return -a.T.dot(y_mat.dot((2*c_kk_mat + k_k_mat + k_k_mat.T).dot(y_mat.dot(a))))

    @staticmethod
    def fun_den(a, y_mat, k_mat):
        a_tilde = k_mat.dot(a)
        return OptimumRegularization.fun_a_y_a(a_tilde, y_mat)

    @staticmethod
    def jac_den_k(a, y_mat, k_mat, c_mat, k):
        a_tilde = k_mat.dot(a)
        return OptimumRegularization.jac_a_y_a_k(a_tilde, y_mat, k_mat, c_mat, k)

    @staticmethod
    def fun_num(v_mat, y_mat, _k_mat):
        n_item = v_mat.shape[1]

        num = 0
        for it in range(n_item):
            v_i = v_mat[:, it:it + 1]
            num += OptimumRegularization.fun_a_y_a(v_i, y_mat)

        return num

    @staticmethod
    def jac_num_k(v_mat, y_mat, k_mat, c_mat, k):
        n_item = v_mat.shape[1]

        deriv = 0
        for it in range(n_item):
            v_i = v_mat[:, it:it + 1]
            deriv += OptimumRegularization.jac_a_y_a_k(v_i, y_mat, k_mat, c_mat, k)

        return deriv

    @staticmethod
    def fun_snr(c, a, v_mat, k_mat):
        c_mat = np.diag(c)

        y_mat = np.linalg.matrix_power(k_mat + c_mat, -2)

        num = OptimumRegularization.fun_num(v_mat, y_mat, k_mat)
        den = OptimumRegularization.fun_den(a, y_mat, k_mat)

        return num/den

    @staticmethod
    def jac_snr_k(a, v_mat, y_mat, k_mat, c_mat, k):
        num = OptimumRegularization.fun_num(v_mat, y_mat, k_mat)
        den = OptimumRegularization.fun_den(a, y_mat, k_mat)
        d_num = OptimumRegularization.jac_num_k(v_mat, y_mat, k_mat, c_mat, k)
        d_den = OptimumRegularization.jac_den_k(a, y_mat, k_mat, c_mat, k)

        return (den*d_num - num*d_den)/den**2

    @staticmethod
    def jac_snr(c, a, v_mat, k_mat):
        dim_a = a.shape[0]

        c_mat = np.diag(c)

        y_mat = np.linalg.matrix_power(k_mat + c_mat, -2)

        deriv = np.zeros((dim_a,))
        for k in range(dim_a):
            deriv[k] = OptimumRegularization.jac_snr_k(a, v_mat, y_mat, k_mat, c_mat, k)

        return deriv

    @staticmethod
    def fun(c, v_mat, k_mat):
        c_mat = np.diag(c)

        y_mat = np.linalg.matrix_power(k_mat + c_mat, -2)

        return OptimumRegularization.fun_num(v_mat, y_mat, k_mat)

    @staticmethod
    def jac(c, v_mat, k_mat):
        dim_a = v_mat.shape[0]

        c_mat = np.diag(c)

        y_mat = np.linalg.matrix_power(k_mat + c_mat, -2)

        deriv = np.zeros((dim_a,))
        for k in range(dim_a):
            deriv[k] = OptimumRegularization.jac_num_k(v_mat, y_mat, k_mat, c_mat, k)

        return deriv

    def find_c(self, v_mat_z, k_mat_z):
        v_mat = v_mat_z[1:][:, 1:]
        k_mat = k_mat_z[1:][:, 1:]

        dim_a = v_mat.shape[0]

        c_0 = np.zeros((dim_a,))

        constraint = LinearConstraint(np.ones((dim_a,)), lb=self.bound[0], ub=self.bound[1]*dim_a)

        try:
            # noinspection PyTypeChecker
            obj = minimize(
                fun=self.fun,
                jac=self.jac,
                x0=c_0,
                args=(v_mat, k_mat),
                bounds=[(self.bound[0], None)]*dim_a,
                constraints=constraint,
                method='trust-constr',
                options={'disp': 0}
            )
        except np.linalg.LinAlgError:
            c_opt = np.ones((dim_a + 1,))*self.bound[1]
            c_opt[0] = 0
            return c_opt

        return np.concatenate(([0], obj.x))


class MinNoiseVarianceRegularization:
    def __init__(self, bound, exclude_zero_freq):
        self.bound = bound
        self.exclude_zero_freq = exclude_zero_freq

    @staticmethod
    def grad_a_y_b(a, y_mat, b):
        """
        Gradient of a^T y b
        :param a: (n, 1)
        :param y_mat: (n, n)
        :param b: (n, 1)
        :return: (n, )
        """
        grad_mat = y_mat.dot(a) * y_mat.dot(b)
        return grad_mat[:, 0]

    @staticmethod
    def fun(c, v_mat, k_mat):
        n_item = v_mat.shape[1]

        c_mat = np.diag(c)
        y_mat = np.linalg.inv(k_mat + c_mat)

        val = 0
        for it_1 in range(n_item):
            for it_2 in range(n_item):
                v_1 = v_mat[:, it_1:it_1 + 1]
                v_2 = v_mat[:, it_2:it_2 + 1]

                val += (v_1.T.dot(y_mat).dot(v_2)[0, 0])**2

        return val

    @staticmethod
    def grad(c, v_mat, k_mat):
        dim_a, n_item = v_mat.shape

        c_mat = np.diag(c)
        y_mat = np.linalg.inv(k_mat + c_mat)

        g = np.zeros((dim_a, 1))
        for it_1 in range(n_item):
            for it_2 in range(n_item):
                v_1 = v_mat[:, it_1:it_1 + 1]
                v_2 = v_mat[:, it_2:it_2 + 1]

                g += -2*v_1.T.dot(y_mat).dot(v_2) * y_mat.dot(v_1) * y_mat.dot(v_2)

        return g[:, 0]

    def find_c(self, v_mat_z, k_mat_z, c_0_mat=None):
        if self.exclude_zero_freq:
            b = 1
        else:
            b = 0

        v_mat = v_mat_z[b:][:, b:]
        k_mat = k_mat_z[b:][:, b:]

        dim_a = v_mat.shape[0]

        if c_0_mat is None:
            c_0 = np.zeros((dim_a,))
        else:
            c_0 = np.diag(c_0_mat)[b:]

        constraint = LinearConstraint(np.ones((dim_a,)), lb=self.bound[0], ub=self.bound[1]*dim_a)

        try:
            # noinspection PyTypeChecker
            obj = minimize(
                fun=self.fun,
                jac=self.grad,
                x0=c_0,
                args=(v_mat, k_mat),
                bounds=[(self.bound[0], None)] * dim_a,
                constraints=constraint,
                method='trust-constr',
                options={'disp': 0}
            )
            c_opt = obj.x
        except np.linalg.LinAlgError:
            warnings.warn('Minimization failed! Switching to L2 regularization.')
            c_opt = np.ones((dim_a,))*self.bound[1]

        if self.exclude_zero_freq:
            return np.concatenate(([0], c_opt))
        else:
            return c_opt


if __name__ == '__main__':
    _dim_a = 4
    _n_item = 10
    _m = 0
    _eps = 1e-5

    # Init.
    _a = np.random.randn(_dim_a, 1)

    _v_mat = np.random.randn(_dim_a, _n_item)

    _k_hat = _v_mat.dot(_v_mat.T)/_v_mat.shape[1]

    _c_mat = np.diag(np.random.random((_dim_a,)))

    _c_mat_p = _c_mat.copy()
    _c_mat_p[_m, _m] += _eps

    # Calc. inverse matrix
    _y_mat = np.linalg.inv(_k_hat + _c_mat)
    _y_mat_p = np.linalg.inv(_k_hat + _c_mat_p)

    # Calc. numerical deriv
    _deriv_num = (MinNoiseVarianceRegularization.fun(np.diag(_c_mat_p), _v_mat, _k_hat)
                  - MinNoiseVarianceRegularization.fun(np.diag(_c_mat), _v_mat, _k_hat))/_eps

    # Deriv.
    _deriv = MinNoiseVarianceRegularization.grad(np.diag(_c_mat), _v_mat, _k_hat)[_m]

    # Find optimum c
    _opt_reg = MinNoiseVarianceRegularization(bound=(0, 1))
    _c_opt = _opt_reg.find_c(_v_mat, _k_hat)
    print(_c_opt)
