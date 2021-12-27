import numpy as np

from app.models.vandermonde import Vandermonde, VandermondeType
from app.models.clustering.one_user_one_cluster import OneUserOneCluster, OneUserOneClusterBiasCorrected
from core.alternate import Alternate

rng = np.random.default_rng(1)


def miss_not_at_random(r, p_0, alpha):
    p_observe = p_0*(1 + alpha*(r - np.mean(r)))

    observe_mask = p_observe > rng.random(r.shape)
    observe_indices = np.where(observe_mask)

    return observe_mask, observe_indices


if __name__ == '__main__':
    # ----- Settings -----
    # Data simulation
    _n_user = 5000
    _n_item = 1000

    _dim_x = 1

    _simul_sett = {
        'm': 3,
        'alpha': 0.1,
        'p_0': 0.2,
        'sigma_n': 0.5,
    }

    # Debiaser
    _deb_sett = {
        'm': _simul_sett['m'],
        'sigma_n': _simul_sett['sigma_n'],
        'n_iter': 1,
    }

    # ----- Init. -----
    # VMs
    _vm_simul = Vandermonde.get_instance(
        dim_x=_dim_x,
        m=_simul_sett['m'],
        l2_lambda=0,
        vm_type=VandermondeType.COS_MULT,
    )
    _vm_simul.fit()

    _vm_deb = Vandermonde.get_instance(
        dim_x=_dim_x,
        m=_deb_sett['m'],
        l2_lambda=0,
        vm_type=VandermondeType.COS_MULT,
    )
    _vm_deb.fit()

    # Clustering
    _cls = OneUserOneCluster()
    _cls_deb = OneUserOneClusterBiasCorrected(_deb_sett['sigma_n'], _deb_sett['n_iter'])

    # ----- Simulate ground truth ratings -----
    # Random init.
    _x_mat = rng.random((_dim_x, _n_item))
    _a_mat = rng.normal(loc=0, scale=1, size=(_vm_simul.dim_a, _n_user))

    # Predict
    _vm_simul.transform(_x_mat)
    _r_mat = _vm_simul.predict(_a_mat)

    # Add noise
    _r_n_mat = _r_mat + rng.normal(loc=0, scale=_simul_sett['sigma_n'], size=_r_mat.shape)

    # ----- Miss-Not-At-Random -----
    _r_obs_mat = _r_n_mat.copy()
    for _user in range(_n_user):
        _obs_mask, _obs_idx = miss_not_at_random(_r_n_mat[_user], _simul_sett['p_0'], _simul_sett['alpha'])

        _r_obs_mat[_user, ~_obs_mask] = np.nan

    # ----- Find a -----
    _vm_deb.transform(_x_mat)

    _a_hat_mat, _ = _cls.fit_transform(vm=_vm_deb, rating_mat=_r_obs_mat)

    _a_hat_un_mat, _ = _cls_deb.fit_transform(vm=_vm_deb, rating_mat=_r_obs_mat, verbose=True)

    # ----- Results -----
    # RMSE
    _rmse = Alternate.calc_prediction_rmse(_vm_deb, _a_hat_mat, _r_mat, -np.inf, np.inf)
    _rmse_un = Alternate.calc_prediction_rmse(_vm_deb, _a_hat_un_mat, _r_mat, -np.inf, np.inf)
    print('rmse before and after debiasing are: %.5f, %.5f (%%%.2f improved)'
          % (_rmse, _rmse_un, (_rmse - _rmse_un)/_rmse*100))
