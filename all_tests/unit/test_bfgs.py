import unittest
import numpy as np

from app.models.updating.bfgs import BFGS
from app.models.vandermonde import Vandermonde, VandermondeType


class TestBFGS(unittest.TestCase):
    def test_jac(self):
        # Init. vm
        v_m = Vandermonde.get_instance(dim_x=2, m=4, l2_lambda=0, vm_type=VandermondeType.REAL)

        # Init. rating matrix
        num_user = 4
        num_item = 3

        rating_matrix = np.random.random((num_user, num_item))

        # Init. "x"
        x_matrix_0 = np.random.random((2, num_item))

        # Fit and transform vm
        v_m.fit()

        # Init. "a"
        a_matrix = np.random.normal(loc=0, scale=0.1, size=(v_m.dim_a, num_user))

        # Select an item
        item = 2

        # Exact derivative
        der = BFGS.jac_i(x_matrix_0[:, item], v_m, a_matrix, rating_matrix, item)
        print('derivative:', der)

        # Calc. initial error
        errs_0 = BFGS.loss_i(x_matrix_0[:, item], v_m, a_matrix, rating_matrix, item)

        eps = 1e-7
        # ----- Test 1 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[0, item] += eps

        errs_1 = BFGS.loss_i(x_matrix_1[:, item], v_m, a_matrix, rating_matrix, item)

        der_num = (errs_1 - errs_0) / eps

        print('d/dx_1', der_num)

        self.assertAlmostEqual(0, np.sum((der_num - der[0])**2))

        # ----- Test 2 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[1, item] += eps

        errs_1 = BFGS.loss_i(x_matrix_1[:, item], v_m, a_matrix, rating_matrix, item)

        der_num = (errs_1 - errs_0) / eps

        print('d/dx_2', der_num)

        self.assertAlmostEqual(0, np.sum((der_num - der[1]) ** 2))

    def test_jac_cos_mult(self):
        # Init. vm
        v_m = Vandermonde.get_instance(dim_x=2, m=4, l2_lambda=0, vm_type=VandermondeType.COS_MULT)

        # Init. rating matrix
        num_user = 4
        num_item = 3

        rating_matrix = np.random.random((num_user, num_item))

        # Init. "x"
        x_matrix_0 = np.random.random((v_m.dim_x, num_item))

        # Fit and transform vm
        v_m.fit()

        # Init. "a"
        a_matrix = np.random.normal(loc=0, scale=0.1, size=(v_m.dim_a, num_user))

        # Select an item
        item = 2

        # Exact derivative
        der = BFGS.jac_cos_mult_i(x_matrix_0[:, item], v_m, a_matrix, rating_matrix, item)
        print('derivative:', der)

        # Calc. initial error
        errs_0 = BFGS.loss_i(x_matrix_0[:, item], v_m, a_matrix, rating_matrix, item)

        eps = 1e-7
        # ----- Test 1 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[0, item] += eps

        errs_1 = BFGS.loss_i(x_matrix_1[:, item], v_m, a_matrix, rating_matrix, item)

        der_num = (errs_1 - errs_0) / eps

        print('d/dx_1', der_num)

        self.assertAlmostEqual(0, np.sum((der_num - der[0])**2))

        # ----- Test 2 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[1, item] += eps

        errs_1 = BFGS.loss_i(x_matrix_1[:, item], v_m, a_matrix, rating_matrix, item)

        der_num = (errs_1 - errs_0) / eps

        print('d/dx_2', der_num)

        self.assertAlmostEqual(0, np.sum((der_num - der[1]) ** 2))


if __name__ == '__main__':
    unittest.main()
