import numpy as np
from scipy.optimize import minimize_scalar

from app.models.vandermonde import Vandermonde
from app.utils.mat_ops import vectorize_rows


class FirstOrderBiasCorrection:
    def __init__(self, n_iter=1, estimate_sigma_n=False, sigma_n=None, min_alpha=-1., max_alpha=1.):
        self.n_iter = n_iter  # Num. of iterations to find alpha and bias
        self.estimate_sigma_n = estimate_sigma_n
        self.sigma_n = sigma_n  # Standard deviation of noise
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        assert (sigma_n is not None) or estimate_sigma_n

    @staticmethod
    def _likelihood(alpha, r, obs_mask, p_0):
        """
        Find the likelihood of observing obs_mask from r, assuming
        P(r_i is observed) = p_0*(1 + alpha*(r_i - r_mean))
        :param alpha: defines the severity of missing not at random
        :param r: all ratings (observed or estimated)
        :param obs_mask: True/False list where True indicates and observed rating
        :param p_0: completely at random probability of observing
        :return: likelihood
        """
        r_mean = np.mean(r)
        return np.prod(p_0*(1 + alpha*(r[obs_mask] - r_mean))) * \
            np.prod(1 - p_0*(1 + alpha*(r[~obs_mask] - r_mean)))

    @staticmethod
    def _find_alpha_feasible_range(r, obs_mask, p_0, min_alpha=-1., max_alpha=1.):
        max_rng = max_alpha
        if np.min(r[obs_mask]) < np.mean(r):
            max_rng = np.minimum(max_rng, 1/(np.mean(r) - np.min(r[obs_mask])))
        if np.max(r[~obs_mask]) > np.mean(r):
            max_rng = np.minimum(max_rng, (1 - p_0)/p_0/(np.max(r[~obs_mask]) - np.mean(r)))

        min_rng = min_alpha
        if np.max(r[obs_mask]) > np.mean(r):
            min_rng = np.maximum(min_rng, 1/(np.mean(r) - np.max(r[obs_mask])))
        if np.min(r[~obs_mask]) < np.mean(r):
            min_rng = np.maximum(min_rng, (1 - p_0)/p_0/(np.min(r[~obs_mask]) - np.mean(r)))

        return min_rng - 1e-4, max_rng + 1e-4  # To solve a bug in Scipy

    @staticmethod
    def find_alpha(r, obs_mask, p_0, min_alpha=-1., max_alpha=1.):
        min_alpha, max_alpha = \
            FirstOrderBiasCorrection._find_alpha_feasible_range(r, obs_mask, p_0, min_alpha, max_alpha)

        obj = minimize_scalar(lambda *args: -FirstOrderBiasCorrection._likelihood(*args),
                              args=(r, obs_mask, p_0),
                              bounds=(min_alpha, max_alpha),
                              method='bounded')

        if not obj.success:
            raise Exception('Minimizer failed!')

        return obj.x

    @staticmethod
    def calc_bias(v_mat, k_mat, alpha, sigma_n, c_mat):
        v_mean = np.mean(v_mat, axis=1).reshape((-1, 1))

        bias = np.linalg.inv(k_mat + c_mat).dot(v_mean)*(sigma_n**2)*alpha

        return bias[:, 0]

    def debias(self, vm: Vandermonde, a, r_mat):
        n_user = r_mat.shape[0]

        obs_mask = ~np.isnan(r_mat)

        # Estimate p_0
        p_0_hat = np.mean(obs_mask*1)

        # Calc. k_mat
        k_mat = vm.calc_k_hat(vm.get_v_users(range(n_user), r_mat))

        # Init.
        a_un = a
        for _ in range(self.n_iter):
            # --- Estimate alpha ---
            # Predict ratings
            a_un_mat = np.tile(a_un.reshape((-1, 1)), reps=(1, n_user))
            r_hat_mat = vm.predict(a_un_mat)

            # Estimate sigma_n
            if self.estimate_sigma_n:
                sigma_n = np.std(r_hat_mat[obs_mask] - r_mat[obs_mask])
            else:
                sigma_n = self.sigma_n

            # Fill known ratings
            r_hat_mat[obs_mask] = r_mat[obs_mask]

            # Find alpha
            alpha_hat = self.find_alpha(vectorize_rows(range(n_user), r_hat_mat)[:, 0],
                                        vectorize_rows(range(n_user), obs_mask)[:, 0],
                                        p_0_hat,
                                        min_alpha=self.min_alpha, max_alpha=self.max_alpha)

            # --- Estimate bias ---
            bias = self.calc_bias(vm.v_mat, k_mat, alpha_hat, sigma_n, vm.c_mat)

            # --- Update a_un ---
            a_un = a - bias

        return a_un
