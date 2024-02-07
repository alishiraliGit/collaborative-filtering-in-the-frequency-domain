import numpy as np
from sklearn.metrics import ndcg_score


def rmse(rat_mat_pr, rat_mat_te, msk):
    return np.sqrt(np.mean((rat_mat_pr[msk] - rat_mat_te[msk])**2))


def mae(rat_mat_pr, rat_mat_te, msk):
    return np.mean(np.abs(rat_mat_pr[msk] - rat_mat_te[msk]))


def recall(rat_mat_pr, rat_mat_te, msk, k):
    n_u = len(rat_mat_pr)

    scores = []
    for u in range(n_u):
        if np.sum(msk[u]) < 2*k:
            continue

        rat_u_pr = rat_mat_pr[u][msk[u]]
        rat_u_te = rat_mat_te[u][msk[u]]

        incl_th_te = np.sort(rat_u_te)[-k]
        tops_te = np.where(rat_u_te >= incl_th_te)[0]

        tops_pr = np.argsort(rat_u_pr)[-k:]

        rec = len(set(tops_pr).intersection(tops_te)) / len(tops_pr)

        scores.append(rec)

    return np.mean(scores)


def ndcg(rat_mat_pr: np.ndarray, rat_mat_te: np.ndarray, msk: np.ndarray, k):
    n_u = rat_mat_pr.shape[0]

    scores = []
    for u in range(n_u):
        if np.sum(msk[u]) < 2*k:
            continue

        rat_u_pr = rat_mat_pr[u][msk[u]]
        rat_u_te = rat_mat_te[u][msk[u]]

        scores.append(ndcg_score(rat_u_te.reshape((1, -1)), rat_u_pr.reshape((1, -1)), k))

    return np.mean(scores)