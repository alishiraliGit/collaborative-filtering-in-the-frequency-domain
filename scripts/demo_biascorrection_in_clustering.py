import numpy as np
from matplotlib import pyplot as plt

from app.models.vandermonde import Vandermonde, VandermondeType
from app.models.clustering.one_user_one_cluster import OneUserOneCluster, OneUserOneClusterBiasCorrected
from core.alternate import Alternate

rng = np.random.default_rng(1)


def miss_not_at_random(r, p_0, alpha):
    p_observe = p_0*(1 + alpha*(r - np.mean(r)))

    observe_mask = p_observe > rng.random(r.shape)
    observe_indices = np.where(observe_mask)

    return observe_mask, observe_indices


def simulate_mnar_data(n_user, n_item, dim_x, m, p_0, alpha, sigma_n):
    # ----- Init. a VM -----
    vm = Vandermonde.get_instance(
        dim_x=dim_x,
        m=m,
        l2_lambda=0,
        vm_type=VandermondeType.COS_MULT,
    )
    vm.fit()

    # ----- Simulate ground truth ratings -----
    # Random init.
    x_mat = rng.random((dim_x, n_item))
    a_mat = rng.normal(loc=0, scale=1, size=(vm.dim_a, n_user))

    # Predict
    vm.transform(x_mat)
    r_mat = vm.predict(a_mat)

    # Add noise
    r_n_mat = r_mat + rng.normal(loc=0, scale=sigma_n, size=r_mat.shape)

    # ----- Miss-Not-At-Random -----
    r_obs_mat = r_n_mat.copy()
    for user in range(n_user):
        obs_mask, obs_idx = miss_not_at_random(r_n_mat[user], p_0, alpha)

        r_obs_mat[user, ~obs_mask] = np.nan

    return x_mat, a_mat, r_n_mat, r_obs_mat


if __name__ == '__main__':
    # ----- Settings -----
    # Data simulation
    _n_user = 300
    _n_item = 300

    _dim_x = 1

    _simul_sett = {
        'm': 2,
        'p_0': 0.2,
        'alpha': 0.2,
        'sigma_n': 0.2,
    }

    # Debiaser
    _deb_sett = {
        'm': _simul_sett['m'],
        'l2_lambda': 1,
        'n_iter': 1,
        'estimate_sigma_n': True,
    }

    # ----- Init. -----
    # VM
    _vm_bc = Vandermonde.get_instance(
        dim_x=_dim_x,
        m=_deb_sett['m'],
        l2_lambda=_deb_sett['l2_lambda'],
        vm_type=VandermondeType.COS_MULT,
    )
    _vm_bc.fit()

    # Clustering
    _cls = OneUserOneCluster()
    _cls_deb = OneUserOneClusterBiasCorrected(
        n_iter=_deb_sett['n_iter'],
        estimate_sigma_n=_deb_sett['estimate_sigma_n']
    )

    # ----- Simulate ground truth ratings -----
    _x_mat, _a_mat, _r_mat, _r_obs_mat = simulate_mnar_data(_n_user, _n_item, _dim_x, **_simul_sett)

    # ----- Find a -----
    _vm_bc.transform(_x_mat)

    _a_hat_mat, _ = _cls.fit_transform(vm=_vm_bc, rating_mat=_r_obs_mat)

    _a_hat_un_mat, _ = _cls_deb.fit_transform(vm=_vm_bc, rating_mat=_r_obs_mat, verbose=True)

    # ----- Results -----
    # RMSE
    _rmse = Alternate.calc_prediction_rmse(_vm_bc, _a_hat_mat, _r_mat, -np.inf, np.inf)
    _rmse_un = Alternate.calc_prediction_rmse(_vm_bc, _a_hat_un_mat, _r_mat, -np.inf, np.inf)
    print('rmse before and after debiasing are: %.5f, %.5f (%%%.2f improved)'
          % (_rmse, _rmse_un, (_rmse - _rmse_un)/_rmse*100))

    # Compare a
    if _dim_x == 1 and _a_hat_mat.shape[0] > _a_mat.shape[0]:
        _a_mat = np.concatenate((_a_mat, np.zeros((_a_hat_mat.shape[0] - _a_mat.shape[0], _n_user))), axis=0)
    if _dim_x == 1 and _a_hat_mat.shape[0] < _a_mat.shape[0]:
        _a_hat_mat = np.concatenate((_a_hat_mat, np.zeros((_a_mat.shape[0] - _a_hat_mat.shape[0], _n_user))), axis=0)
        _a_hat_un_mat = \
            np.concatenate((_a_hat_un_mat, np.zeros((_a_mat.shape[0] - _a_hat_un_mat.shape[0], _n_user))), axis=0)
    if _a_mat.shape[0] == _a_hat_mat.shape[0]:
        plt.figure(figsize=(4, 4))
        plt.plot(np.mean((_a_mat - _a_hat_mat)**2, axis=1), 'k')
        plt.plot(np.mean((_a_mat - _a_hat_un_mat)**2, axis=1), 'r')
