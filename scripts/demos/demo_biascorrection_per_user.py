import numpy as np

from app.models.vandermonde import Vandermonde, VandermondeType, RegularizationType
from app.models.clustering.one_user_one_cluster import OneUserOneCluster, OneUserOneClusterBiasCorrected
from app.models.updating.approximate_updater import ApproximateUpdater
from app.models.updating.bfgs import BFGS
from app.models.updating.multi_updater_wrapper import MultiUpdaterWrapper
from app.models.logger import Logger
from core.alternate import Alternate
from scripts.demos.demo_biascorrection_in_clustering import simulate_mnar_data

rng = np.random.default_rng(2)


if __name__ == '__main__':
    # ----- Settings -----
    _n_user = 5000
    _n_item = 1000

    _n_alter = 10

    # VM
    _dim_x = 1
    _m = 3
    _l2_lambda = 0.5

    # Data simulation
    _simul_sett = {
        'p_0': 0.1,
        'alpha': 0.2,
        'sigma_n': 0.3,
    }

    # Bias-corrected clustering
    _cls_bc_sett = {
        'n_iter': 1,
        'estimate_sigma_n': True,
        'sigma_n': 0,
        'min_alpha': 0,
    }

    # Approximate updater
    _approx_upd_sett = {
        'gamma': 1
    }

    # BFGS updater
    _bfgs_sett = {
        'max_iter': 5
    }

    # ----- Init. -----
    # Init. VM
    _vm = Vandermonde.get_instance(
        dim_x=_dim_x,
        m=_m,
        vm_type=VandermondeType.COS_MULT,
        reg_type=RegularizationType.L2,
        reg_params={'l2_lambda': _l2_lambda}
    )

    # Init x
    _x_mat_0 = rng.random((_dim_x, _n_item))

    # Init. clustering
    _cls = OneUserOneClusterBiasCorrected(**_cls_bc_sett)

    # Init. updaters
    _approx_upd = ApproximateUpdater(x_mat_0=_x_mat_0, **_approx_upd_sett)

    _bfgs_upd = BFGS(x_mat_0=_x_mat_0, **_bfgs_sett)

    _multi_upd = MultiUpdaterWrapper(upds=[_approx_upd, _bfgs_upd])

    # Init. alternate
    _alt = Alternate(clust=_cls, upd=_multi_upd)

    # Init. logger
    _logger = Logger(settings=Logger.dicts_to_dic([_simul_sett, _cls_bc_sett, _approx_upd_sett, _bfgs_sett]),
                     save_path='..',
                     do_plot=False)

    # ----- Simulate ground truth ratings -----
    _, _a_mat, _r_mat, _r_obs_mat = simulate_mnar_data(_n_user, _n_item, _dim_x, _m, **_simul_sett)

    # ----- Fit and transform VM -----
    _vm.fit()
    _vm.transform(_x_mat_0)

    # ----- Do the alternation -----
    # Validation RMSE: equivalent to clustering w/o bias correction
    # Test RMSE: clustering w/ bias correction
    _a_hat_mat = _alt.run(
        _vm, _r_obs_mat,
        rating_mat_va=_r_mat,
        rating_mat_te=_r_mat,
        n_iter=_n_alter,
        min_val=-np.inf,
        max_val=np.inf,
        logger=_logger
    )
