import unittest
import numpy as np

from app.models.updating.least_square import LeastSquare
from app.models.vandermonde import Vandermonde
from app.utils.mat_ops import vectorize


class TestLeastSquare(unittest.TestCase):
    def test_jac(self):
        # Init. vm
        v_m = Vandermonde(2, 2, 0)

        # Init. rating matrix
        num_user = 4
        num_item = 3

        rating_matrix = np.random.random((num_user, num_item))

        # Init. "x"
        x_matrix_0 = np.random.random((2, num_item))

        # Fit and transform vm
        v_m.fit()
        v_m.transform(x_matrix_0)

        # Init. "a"
        a_matrix = np.random.normal(loc=0, scale=0.1, size=(v_m.dim_a, num_user))

        # Exact derivative
        der = LeastSquare.jac(vectorize(x_matrix_0, 'C'), v_m, a_matrix, rating_matrix)

        # Calc. initial error
        errs_0 = LeastSquare.loss(vectorize(x_matrix_0, 'C'), v_m, a_matrix, rating_matrix)

        eps = 1e-7
        # ----- Test 1 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[0, 0] += eps

        v_m.transform(x_matrix_1)

        errs_1 = LeastSquare.loss(vectorize(x_matrix_1, 'C'), v_m, a_matrix, rating_matrix)

        der_num = (errs_1 - errs_0) / eps

        self.assertAlmostEqual(0, np.sum((der_num - der[:, 0])**2))

        # ----- Test 2 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[0, 1] += eps

        v_m.transform(x_matrix_1)

        errs_1 = LeastSquare.loss(vectorize(x_matrix_1, 'C'), v_m, a_matrix, rating_matrix)

        der_num = (errs_1 - errs_0) / eps

        self.assertAlmostEqual(0, np.sum((der_num - der[:, 1]) ** 2))

        # ----- Test 3 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[1, 0] += eps

        v_m.transform(x_matrix_1)

        errs_1 = LeastSquare.loss(vectorize(x_matrix_1, 'C'), v_m, a_matrix, rating_matrix)

        der_num = (errs_1 - errs_0) / eps

        self.assertAlmostEqual(0, np.sum((der_num - der[:, num_item]) ** 2))

        # ----- Test 4 -----
        x_matrix_1 = x_matrix_0.copy()
        x_matrix_1[0, 0] += eps
        x_matrix_1[0, 1] += eps

        v_m.transform(x_matrix_1)

        errs_1 = LeastSquare.loss(vectorize(x_matrix_1, 'C'), v_m, a_matrix, rating_matrix)

        der_num = (errs_1 - errs_0) / eps

        self.assertAlmostEqual(0, np.sum((der_num - der[:, 0] -der[:, 1]) ** 2))



if __name__ == '__main__':
    unittest.main()
