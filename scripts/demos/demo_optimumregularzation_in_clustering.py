import numpy as np
from matplotlib import pyplot as plt

from app.models.vandermonde import Vandermonde, VandermondeType, RegularizationType
from app.models.clustering.one_user_one_cluster import OneUserOneCluster
from core.alternate import Alternate
from demo_biascorrection_in_clustering import simulate_mnar_data

rng = np.random.default_rng(4)


if __name__ == '__main__':
    # ----- Settings -----
    # Data simulation
    _n_user = 300
    _n_item = 50

    _dim_x = 1
    _m = 2

    _simul_sett = {
        'p_0': 0.4,
        'alpha': 0,
        'sigma_n': 1,
        'z': 0.8
    }

    # Regularization
    _reg_type_1 = RegularizationType.L2
    _reg_params_1 = {'l2_lambda': 0.3, 'exclude_zero_freq': True}

    _reg_type_2 = RegularizationType.MAX_SNR
    _reg_params_2 = {'bound': (0.2, 0.3), 'exclude_zero_freq': True}
    # _reg_type_2 = RegularizationType.POW
    # _reg_params_2 = {'l2_lambda': 0.43, 'z': 1.1}

    # ----- Init. -----
    # VM
    _vm_1 = Vandermonde.get_instance(
        dim_x=_dim_x,
        m=_m,
        vm_type=VandermondeType.COS_MULT,
        reg_type=_reg_type_1,
        reg_params=_reg_params_1
    )
    _vm_1.fit()

    _vm_2 = Vandermonde.get_instance(
        dim_x=_dim_x,
        m=_m,
        vm_type=VandermondeType.COS_MULT,
        reg_type=_reg_type_2,
        reg_params=_reg_params_2
    )
    _vm_2.fit()

    # Clustering
    _cls = OneUserOneCluster()

    # ----- Simulate ground truth ratings -----
    _x_mat, _a_mat, _r_mat, _r_obs_mat = simulate_mnar_data(_n_user, _n_item, _dim_x, _m, **_simul_sett)

    # ----- Find a -----
    _vm_1.transform(_x_mat)
    _vm_2.transform(_x_mat)

    _a_hat_mat_1, _ = _cls.fit_transform(vm=_vm_1, rating_mat=_r_obs_mat, verbose=True)
    _a_hat_mat_2, _ = _cls.fit_transform(vm=_vm_2, rating_mat=_r_obs_mat, verbose=True)

    # ----- Results -----
    # RMSE
    _rmse_1 = Alternate.calc_prediction_rmse(_vm_1, _a_hat_mat_1, _r_mat, -np.inf, np.inf)
    _rmse_2 = Alternate.calc_prediction_rmse(_vm_2, _a_hat_mat_2, _r_mat, -np.inf, np.inf)
    print('rmse with reg. 1 and 2 are: %.5f, %.5f (%%%.2f improved)'
          % (_rmse_1, _rmse_2, (_rmse_1 - _rmse_2)/_rmse_1*100))

    # Compare a
    plt.figure(figsize=(4, 4))
    plt.plot(np.mean((_a_mat - _a_hat_mat_1)**2, axis=1), 'k')
    plt.plot(np.mean((_a_mat - _a_hat_mat_2)**2, axis=1), 'r')
    plt.plot(np.mean(_a_hat_mat_1**2, axis=1), 'k--')
    plt.plot(np.mean(_a_hat_mat_2**2, axis=1), 'r--')
    plt.plot(np.mean(_a_mat**2, axis=1), 'b')

    # Compare c_mats
    print('c_mat of reg. 1:')
    print(np.diag(_vm_1.c_mat))
    print('c_mat of reg. 2:')
    print(np.diag(_vm_2.c_mat))
