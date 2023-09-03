import numpy as np
from scipy.optimize import minimize, Bounds
from tqdm import tqdm

from app.models.vandermonde import Vandermonde, VandermondeType
from app.models.updating.updater_base import Updater
from app.utils.mat_ops import unvectorize


class BFGS(Updater):
    def __init__(self, x_mat_0, max_iter):
        Updater.__init__(self)

        self.x_mat = x_mat_0  # [dim_x x n_item]
        self.max_iter = max_iter

    def update_x(self, x_mat):
        self.x_mat = x_mat

    def fit(self, vm: Vandermonde, a_mat, rating_mat, propensity_mat=None):
        n_item = rating_mat.shape[1]

        bounds = Bounds(lb=0, ub=1)

        for item in tqdm(range(n_item), desc='fit (bfgs)'):
            x_0_i_vec = self.x_mat[:, item]

            res = minimize(
                fun=BFGS.loss_jac_i,
                x0=x_0_i_vec,
                args=(vm.copy(), a_mat, rating_mat, item, propensity_mat),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': self.max_iter, 'disp': False}
            )

            self.x_mat[:, item] = res.x

        return

    def transform(self, vm: Vandermonde):
        vm.transform(self.x_mat)

        return self.x_mat

    @staticmethod
    def loss_i(x_i_vec, vm_org: Vandermonde, a_mat, rating_mat, item, propensity_mat=None):
        # Copy vm
        vm_copy = vm_org.copy()

        # Reshape x to col vec
        x_i = unvectorize(x_i_vec, 1, 'F')  # [dim_x x 1]

        # Use x_mat in vm
        vm_copy.transform(x_i)

        # Predict rating
        rating_mat_pr = vm_copy.predict(a_mat)

        # Select item's ratings
        rating_mat_i = rating_mat[:, item:(item + 1)]  # [n_user x 1]

        # Calculate error
        err = rating_mat_pr - rating_mat_i

        err_not_nan = err[~np.isnan(err)]

        # Get propensity scores
        if propensity_mat is None:
            p_i = np.ones(err_not_nan.shape)
        else:
            p_i = propensity_mat[~np.isnan(err[:, 0]), item:(item + 1)]

        #  Weighted sum
        return np.sum((err_not_nan**2)/p_i)

    @staticmethod
    def jac_i(x_i_vec, vm_org: Vandermonde, a_mat, rating_mat, item, propensity_mat=None):
        # Copy vm
        vm_copy = vm_org.copy()

        # Reshape x to col vec
        x_i = unvectorize(x_i_vec, 1, 'F')  # [dim_x x 1]

        # Use x_mat in vm
        vm_copy.transform(x_i)

        v_i = vm_copy.v_mat  # [dim_a x 1]

        # Select rated users
        rating_mat_i = rating_mat[:, item:(item + 1)]  # [n_user x 1]

        rated_users_i = np.nonzero(~np.isnan(rating_mat_i))[0]

        s_i = rating_mat_i[rated_users_i]  # [n_users_i x 1]

        a_mat_i = a_mat[:, rated_users_i]  # [dim_a x n_users_i]

        # Get propensity scores
        if propensity_mat is None:
            n_observed_users_i = len(rated_users_i)
            p_i = np.ones((n_observed_users_i, 1))  # Normalization is unnecessary
        else:
            p_i = propensity_mat[rated_users_i, item:(item + 1)]

        # Normalize s_i and a_mat_i
        sqrt_ips = 1/np.sqrt(p_i)

        s_i *= sqrt_ips
        a_mat_i *= sqrt_ips.T

        # Loop on dim_x
        dim_x = x_i.shape[0]

        deriv = np.zeros((dim_x,))  # [dim_x,]

        for q in range(dim_x):
            if vm_copy.vm_type == VandermondeType.REAL:
                c_q = np.concatenate((vm_copy.v_mult[:, q:(q + 1)], vm_copy.v_mult[:, q:(q + 1)]), axis=0)  # [dim_a, 1]
            elif vm_copy.vm_type == VandermondeType.COS:
                c_q = vm_copy.v_mult[:, q:(q + 1)]  # [dim_a, 1]
            else:
                raise Exception('Invalid VM type!')

            a_mat_i_re = np.pi * a_mat_i * np.tile(c_q, reps=(1, len(rated_users_i)))

            # Very smart implementation!!!
            if vm_copy.vm_type == VandermondeType.REAL:
                deriv[q] = -2 * (v_i.T.dot(a_mat_i) - s_i.T).dot(
                    a_mat_i_re.T.dot(
                        np.concatenate((np.sin(np.pi * vm_copy.v_mult.dot(x_i)),
                                        -np.cos(np.pi * vm_copy.v_mult.dot(x_i))),
                                       axis=0)
                    )
                )
            elif vm_copy.vm_type == VandermondeType.COS:
                deriv[q] = -2 * (v_i.T.dot(a_mat_i) - s_i.T).dot(
                    a_mat_i_re.T.dot(np.sin(np.pi * vm_copy.v_mult.dot(x_i)))
                )
            else:
                raise Exception('Invalid VM type!')

        return deriv

    @staticmethod
    def jac_cos_mult_i(x_i_vec, vm_org: Vandermonde, a_mat, rating_mat, item, propensity_mat=None):
        # Copy vm
        vm_copy = vm_org.copy()

        # Reshape x to col vec
        x_i = unvectorize(x_i_vec, 1, 'F')  # [dim_x x 1]

        # Use x_mat in vm
        vm_copy.transform(x_i)

        v_i = vm_copy.v_mat  # [dim_a x 1]

        # Select rated users
        rating_mat_i = rating_mat[:, item:(item + 1)]  # [n_user x 1]

        rated_users_i = np.nonzero(~np.isnan(rating_mat_i))[0]

        s_i = rating_mat_i[rated_users_i]  # [n_users_i x 1]

        a_mat_i = a_mat[:, rated_users_i]  # [dim_a x n_users_i]

        # Get propensity scores
        if propensity_mat is None:
            p_i = np.ones(s_i.shape)  # Normalization is unnecessary
        else:
            p_i = propensity_mat[rated_users_i, item:(item + 1)]

        # Normalize s_i and a_mat_i
        sqrt_ips = 1/np.sqrt(p_i)

        s_i *= sqrt_ips
        a_mat_i *= sqrt_ips.T

        # Loop on dim_x
        dim_x = x_i.shape[0]

        deriv = np.zeros((dim_x,))  # [dim_x,]

        for q in range(dim_x):
            x_i_q = x_i[q, 0]
            m_q = vm_copy.v_mult[:, q:(q+1)]

            dh = a_mat_i.T.dot(-v_i * np.pi*m_q * np.tan(np.pi*m_q*x_i_q))

            deriv[q] = 2*(vm_copy.predict(a_mat_i) - s_i).T.dot(dh)

        return deriv

    @staticmethod
    def loss_jac_i(x_i_vec, vm_org: Vandermonde, a_mat, rating_mat, item, propensity_mat=None):

        loss = BFGS.loss_i(x_i_vec, vm_org, a_mat, rating_mat, item, propensity_mat=propensity_mat)

        if vm_org.vm_type == VandermondeType.COS_MULT:
            jac = BFGS.jac_cos_mult_i(x_i_vec, vm_org, a_mat, rating_mat, item, propensity_mat=propensity_mat)
        else:
            jac = BFGS.jac_i(x_i_vec, vm_org, a_mat, rating_mat, item, propensity_mat=None)

        return loss, jac
