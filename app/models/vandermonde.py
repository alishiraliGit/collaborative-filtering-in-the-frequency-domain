import abc
import numpy as np

from app.models.debiasing.optimumregularization import MinNoiseVarianceRegularization as MinNoiseReg
from app.models.debiasing.optimumregularization import MaxSNRRegularization as MaxSNRReg
from app.utils.mat_ops import vectorize_rows, e1


class VandermondeType:
    COS = 'cos'
    REAL = 'real'
    COS_MULT = 'cos_mult'


class RegularizationType:
    L2 = 'l2'
    POW = 'pow'
    MIN_NOISE_VAR = 'min_noise_var'
    MAX_SNR = 'max_snr'
    POST_MAX_SNR = 'post_max_snr'


class Vandermonde(abc.ABC):
    def __init__(self, dim_x, dim_a, m, vm_type, reg_type: RegularizationType, reg_params=None):
        self.dim_x = dim_x
        self.dim_a = dim_a
        self.m = m

        self.v_mult = None  # [dim_a x dim_x]
        self.v_mat = None  # [dim_a x n_item]

        self.vm_type = vm_type

        self.reg_type = reg_type
        self.reg_params = reg_params  # dict
        self.c_mat = None
        self.c_mat_is_updated = False
        # ToDo
        self.tmp = []

    @staticmethod
    def get_instance(dim_x, m, vm_type: VandermondeType, reg_type: RegularizationType, reg_params=None):
        if vm_type == VandermondeType.COS:
            return VandermondeCos(dim_x, m, reg_type=reg_type, reg_params=reg_params)
        elif vm_type == VandermondeType.REAL:
            return VandermondeReal(dim_x, m, reg_type=reg_type, reg_params=reg_params)
        elif vm_type == VandermondeType.COS_MULT:
            return VandermondeCosMult(dim_x, m, reg_type=reg_type, reg_params=reg_params)

    def get_v_users(self, users, rating_mat):
        if not isinstance(users, (list, np.ndarray)):  # if users is a number only
            users = [users]

        observed_indices = np.argwhere(~np.isnan(rating_mat[users, :]))  # order='C'
        return self.v_mat[:, observed_indices[:, 1]]

    @staticmethod
    def calc_k_hat(v_obs_mat):
        return v_obs_mat.dot(v_obs_mat.T)/v_obs_mat.shape[1]

    def update_c_mat(self, a=None, is_test=False):
        # Flag c_mat
        self.c_mat_is_updated = True

        if self.reg_type == RegularizationType.L2:
            e1_mat = np.diag(e1(self.dim_a))*(self.reg_params['exclude_zero_freq']*1)
            c_mat = (np.eye(self.dim_a) - e1_mat)*self.reg_params['l2_lambda']
        elif self.reg_type == RegularizationType.POW:
            z = self.reg_params['z']
            e1_mat = np.diag(e1(self.dim_a)) * (self.reg_params['exclude_zero_freq']*1)
            c_mat = (np.diag(z**np.array(range(self.dim_a))) - e1_mat)*self.reg_params['l2_lambda']
        elif self.reg_type == RegularizationType.MIN_NOISE_VAR:
            c_mat = np.diag(MinNoiseReg(self.reg_params['bound'], self.reg_params['exclude_zero_freq']).find_c(
                self.v_mat,
                self.calc_k_hat(v_obs_mat=self.v_mat),
                c_0_mat=self.c_mat
            ))
        elif self.reg_type == RegularizationType.MAX_SNR or\
                (self.reg_type == RegularizationType.POST_MAX_SNR and is_test):
            c_mat = np.diag(MaxSNRReg(self.reg_params['bound'], self.reg_params['exclude_zero_freq']).find_c(
                a,
                self.v_mat,
                self.calc_k_hat(v_obs_mat=self.v_mat),
                c_0_mat=self.c_mat
            ))
            self.c_mat_is_updated = False
        elif self.reg_type == RegularizationType.POST_MAX_SNR:
            e1_mat = np.diag(e1(self.dim_a))*(self.reg_params['exclude_zero_freq']*1)
            c_mat = (np.eye(self.dim_a) - e1_mat)*self.reg_params['bound'][1]
        else:
            raise Exception('unknown regularization type!')

        # Set c_mat
        self.c_mat = c_mat

        return c_mat

    def calc_a_users(self, users, rating_mat):
        v_u_mat = self.get_v_users(users, rating_mat)

        k_hat_mat = self.calc_k_hat(v_u_mat)

        s_u = vectorize_rows(users, rating_mat)

        # Check c_mat to be updated
        if not self.c_mat_is_updated:
            if self.reg_type == RegularizationType.MAX_SNR:
                vm_0 = self.copy()
                vm_0.v_mat = self.v_mat
                vm_0.reg_type = RegularizationType.L2
                vm_0.reg_params = {
                    'l2_lambda': self.reg_params['bound'][1],
                    'exclude_zero_freq': self.reg_params['exclude_zero_freq']
                }
                a_0 = vm_0.calc_a_users(users, rating_mat)
                self.update_c_mat(a_0[:, 0])
                # ToDo
                self.tmp.append(np.diag(self.c_mat).reshape((1, -1)))
            else:
                self.update_c_mat()

        a_u = np.linalg.inv(k_hat_mat + self.c_mat).dot(v_u_mat/v_u_mat.shape[1]).dot(s_u)

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
        vm = self.__class__(self.dim_x, self.m, reg_type=self.reg_type, reg_params=self.reg_params)

        vm.v_mult = self.v_mult.copy()

        return vm


class VandermondeCos(Vandermonde):
    def __init__(self, dim_x, m, reg_type: RegularizationType, reg_params=None):
        dim_a = self.__calc_dim_a(dim_x, m)
        Vandermonde.__init__(self, dim_x, dim_a, m, vm_type=VandermondeType.COS,
                             reg_type=reg_type, reg_params=reg_params)

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

        # Flag c_mat
        self.c_mat_is_updated = False

        return self.v_mat


class VandermondeReal(Vandermonde):
    def __init__(self, dim_x, m, reg_type: RegularizationType, reg_params=None):
        dim_a = self.__calc_dim_a(dim_x, m)
        Vandermonde.__init__(self, dim_x, dim_a, m, vm_type=VandermondeType.REAL,
                             reg_type=reg_type, reg_params=reg_params)

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

        # Flag c_mat
        self.c_mat_is_updated = False

        return self.v_mat


class VandermondeCosMult(Vandermonde):
    def __init__(self, dim_x, m, reg_type: RegularizationType, reg_params=None):
        dim_a = self.__calc_dim_a(dim_x, m)
        Vandermonde.__init__(self, dim_x, dim_a, m, vm_type=VandermondeType.COS_MULT,
                             reg_type=reg_type, reg_params=reg_params)

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

        # Flag c_mat
        self.c_mat_is_updated = False

        return self.v_mat
