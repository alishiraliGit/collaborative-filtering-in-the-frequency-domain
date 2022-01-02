import warnings
import abc

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class OptimumRegularizationBase(abc.ABC):
    def __init__(self, bound, exclude_zero_freq):
        self.bound = bound
        self.exclude_zero_freq = exclude_zero_freq

    @staticmethod
    def grad_a_y_b(a, y_mat, b):
        """
        Gradient of a^T y b where y = (K + C)^-1
        :param a: (n, 1)
        :param y_mat: (n, n)
        :param b: (n, 1)
        :return: (n, )
        """
        grad_mat = -y_mat.dot(a) * y_mat.dot(b)
        return grad_mat[:, 0]

    def find_c(self, *args_z, c_0_mat=None):
        if self.exclude_zero_freq:
            b = 1
        else:
            b = 0

        args = []
        for arg in args_z:
            if arg.ndim == 1:
                args.append(arg[b:])
            elif arg.ndim == 2:
                args.append(arg[b:][:, b:])
            else:
                raise Exception('bad input')

        dim_a = args[0].shape[0]

        if c_0_mat is None:
            c_0 = np.ones((dim_a,))*self.bound[1]
        else:
            c_0 = np.diag(c_0_mat)[b:]

        constraint = LinearConstraint(np.ones((dim_a,)), lb=-np.inf, ub=self.bound[1]*dim_a)

        try:
            # noinspection PyTypeChecker
            obj = minimize(
                fun=self.fun,
                jac=self.grad,
                x0=c_0,
                args=tuple(args),
                bounds=[(self.bound[0], None)]*dim_a,
                constraints=constraint,
                method='trust-constr',
                options={'disp': 0}
            )
            c_opt = obj.x
        except:
            warnings.warn('Minimization failed! Switching to L2 regularization.')
            c_opt = np.ones((dim_a,))*self.bound[1]

        if self.exclude_zero_freq:
            return np.concatenate(([0], c_opt))
        else:
            return c_opt


class MinNoiseVarianceRegularization(OptimumRegularizationBase):
    def __init__(self, bound, exclude_zero_freq):
        super().__init__(bound, exclude_zero_freq)

    @staticmethod
    def fun_old(c, v_mat, k_mat):
        n_item = v_mat.shape[1]

        c_mat = np.diag(c)
        y_mat = np.linalg.inv(k_mat + c_mat)

        val = 0
        for it_1 in range(n_item):
            v_1 = v_mat[:, it_1:it_1 + 1]
            left_mult_mat = y_mat.dot(v_1)

            val += np.sum(left_mult_mat.T.dot(v_mat)**2)

        return val

    @staticmethod
    def fun(c, v_mat, k_mat):
        c_mat = np.diag(c)
        y_mat = np.linalg.inv(k_mat + c_mat)

        val = np.sum(v_mat.T.dot(y_mat).dot(v_mat)**2)

        return val

    @staticmethod
    def grad(c, v_mat, k_mat):
        dim_a, n_item = v_mat.shape

        c_mat = np.diag(c)
        y_mat = np.linalg.inv(k_mat + c_mat)

        g = np.zeros((dim_a,))
        for it_1 in range(n_item):
            v_1 = v_mat[:, it_1:it_1 + 1]
            left_mult = y_mat.dot(v_1)
            for it_2 in range(n_item):
                v_2 = v_mat[:, it_2:it_2 + 1]

                g += 2*left_mult.T.dot(v_2)[0, 0] * OptimumRegularizationBase.grad_a_y_b(v_1, y_mat, v_2)

        return g


class MaxSNRRegularization(OptimumRegularizationBase):
    def __init__(self, bound, exclude_zero_freq):
        super().__init__(bound, exclude_zero_freq)

    @staticmethod
    def signal_power(a, k_mat, y_mat, v_mat):
        s = a.reshape((1, -1)).dot(k_mat).dot(y_mat).dot(v_mat)
        return np.sum(s**2)

    @staticmethod
    def grad_signal_power(a, k_mat, y_mat, v_mat):
        dim_a, n_item = v_mat.shape

        left_multiplier = k_mat.dot(a.reshape((-1, 1)))

        g = np.zeros((dim_a,))
        for it in range(n_item):
            v_i = v_mat[:, it:it + 1]
            g += 2*left_multiplier.T.dot(y_mat).dot(v_i)[0, 0] * \
                OptimumRegularizationBase.grad_a_y_b(left_multiplier, y_mat, v_i)

        return g

    @staticmethod
    def fun(c, a, v_mat, k_mat):
        y_mat = np.linalg.inv(k_mat + np.diag(c))

        s = MaxSNRRegularization.signal_power(a, k_mat, y_mat, v_mat)
        n = MinNoiseVarianceRegularization.fun(c, v_mat, k_mat)

        return n/s

    @staticmethod
    def grad(c, a, v_mat, k_mat):
        y_mat = np.linalg.inv(k_mat + np.diag(c))

        s = MaxSNRRegularization.signal_power(a, k_mat, y_mat, v_mat)
        n = MinNoiseVarianceRegularization.fun(c, v_mat, k_mat)
        g_s = MaxSNRRegularization.grad_signal_power(a, k_mat, y_mat, v_mat)
        g_n = MinNoiseVarianceRegularization.grad(c, v_mat, k_mat)

        return (s*g_n - n*g_s)/s**2


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

    # Select the regularization
    _Reg = MinNoiseVarianceRegularization

    # _args = (_a[:, 0], _v_mat, _k_hat)
    _args = (_v_mat, _k_hat)

    # Calc. numerical deriv
    _deriv_num = (_Reg.fun(np.diag(_c_mat_p), *_args) - _Reg.fun(np.diag(_c_mat), *_args))/_eps

    # Deriv.
    _deriv = _Reg.grad(np.diag(_c_mat), *_args)[_m]

    print(_deriv_num, _deriv)

    # Find optimum c
    _opt_reg = _Reg(bound=(0, 1), exclude_zero_freq=False)
    _c_opt = _opt_reg.find_c(_a[:, 0], _v_mat, _k_hat)
    print(_c_opt)
