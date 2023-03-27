import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from app.models.vandermonde import Vandermonde
from app.models.updating.updater_base import Updater
from app.utils import pytorch_utils as ptu

ptu.init_gpu(use_gpu=False)


class Adam(Updater):
    def __init__(self, x_mat_0: np.ndarray, n_iter, optimizer_spec):
        Updater.__init__(self)

        self.x_mat: torch.tensor = ptu.from_numpy(x_mat_0)  # [dim_x x n_item]
        self.x_mat.requires_grad = True
        self.n_iter = n_iter
        self.optimizer_spec = optimizer_spec

        # Optimization
        self.optimizer = self.optimizer_spec.constructor(
            [self.x_mat],
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

    def update_x(self, x_mat):
        self.x_mat = ptu.from_numpy(x_mat)
        self.x_mat.requires_grad = True

    def fit(self, vm: Vandermonde, a_mat, rating_mat, propensity_mat=None):
        for _ in tqdm(range(self.n_iter), desc='fit (adam)'):
            loss = self.loss_tensor(vm, a_mat, rating_mat, propensity_mat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.learning_rate_scheduler.step()

        return

    def transform(self, vm: Vandermonde):
        x_mat = ptu.to_numpy(self.x_mat)

        vm.transform(x_mat)

        return x_mat

    def loss_tensor(self, vm_org: Vandermonde, a_mat: np.ndarray, rating_mat: np.ndarray, propensity_mat: np.ndarray):
        a_mat = ptu.from_numpy(a_mat)
        mask = ptu.from_numpy_bool(~np.isnan(rating_mat))
        rating_mat = ptu.from_numpy(rating_mat)
        propensity_mat = ptu.from_numpy(propensity_mat)

        # Copy vm
        vm_copy = vm_org.copy()

        # Use x_mat in vm
        vm_copy.transform_tensor(self.x_mat)

        # Predict rating
        rating_mat_pr = vm_copy.predict_tensor(a_mat)

        # Calculate error
        err = rating_mat_pr - rating_mat

        err_not_nan = err[mask]

        # Get propensity scores
        p = propensity_mat[mask]

        #  Weighted sum
        return torch.mean((err_not_nan**2)/p)
