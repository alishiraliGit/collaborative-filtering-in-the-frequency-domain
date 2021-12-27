import numpy as np
from scipy.optimize import minimize_scalar

from app.models.vandermonde import Vandermonde


class FirstOrderBiasCorrection:
    def __init__(self, sigma_n, n_iter=1):
        self.sigma_n = sigma_n  # Standard deviation of noise
        self.n_iter = n_iter  # Num. of iterations to find alpha and bias

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
    def _find_alpha_feasible_range(r, obs_mask, p_0, min_alpha=-1, max_alpha=1):
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
    def find_alpha(r, obs_mask, p_0, min_alpha=-1, max_alpha=1):
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
    def calc_k_hat(v_obs_mat):
        return v_obs_mat.dot(v_obs_mat.T) / v_obs_mat.shape[1]

    @staticmethod
    def calc_bias(v_obs_mat, alpha, sigma_n):
        k_mat = FirstOrderBiasCorrection.calc_k_hat(v_obs_mat)
        k_inv_mat = np.linalg.inv(k_mat)

        bias = k_inv_mat[:, 0]*(sigma_n ** 2)*alpha

        return bias

    def debias(self, vm: Vandermonde, a, r):
        obs_mask = ~np.isnan(r)

        # Estimate p_0
        p_0_hat = np.mean(obs_mask * 1)

        # Init.
        a_un = a
        for _ in range(self.n_iter):
            # --- Estimate alpha ---
            # Predict ratings
            r_hat = vm.predict(a_un.reshape((-1, 1)))[0]
            r_hat[obs_mask] = r[obs_mask]

            # Find alpha
            alpha_hat = self.find_alpha(r_hat, obs_mask, p_0_hat)

            # --- Estimate bias ---
            v_obs_mat = vm.v_mat[:, obs_mask]
            bias = self.calc_bias(v_obs_mat, alpha_hat, self.sigma_n)

            # --- Update a_un ---
            a_un = a - bias

        return a_un
