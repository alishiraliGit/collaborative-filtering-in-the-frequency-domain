

def get_kmeans_approx_settings():
    sett = {}

    method = 'kmeans_approx'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4

    # Clustering settings
    sett['n_cluster'] = 5
    sett['cls_init_std'] = 0.1

    # Updater settings
    sett['gamma'] = 0.1

    # Estimate regularization coefficients
    sett['l2_lambda'] = 1
    sett['l2_lambda_cls'] = 0

    return sett


def get_boosted_kmeans_approx_ls_settings():
    sett = {}

    method = 'boosted_kmeans_approx_ls'

    sett['method'] = method

    # Vandermonde settings
    sett['dim_x'] = 3
    sett['m'] = 4

    # Clustering settings
    sett['n_cluster'] = 2
    sett['cls_init_std'] = 0.1
    sett['n_learner'] = 4
    sett['n_iter_cls'] = 3

    # Updater settings
    sett['gamma'] = 1
    sett['max_nfev'] = 3

    # Estimate regularization coefficients
    sett['l2_lambda'] = 0
    sett['l2_lambda_cls'] = 0

    return sett
