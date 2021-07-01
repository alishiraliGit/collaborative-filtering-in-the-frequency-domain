import unittest
import numpy as np

from app.models.vandermonde import VandermondeCosMult


class TestVandermondeCosMult(unittest.TestCase):

    def test_fit(self):
        vm = VandermondeCosMult(dim_x=2, m=1, l2_lambda=0)

        vm.fit()

        self.assertListEqual([[0, 0],
                              [1, 0],
                              [0, 1],
                              [1, 1]],
                             vm.v_mult.tolist())

    def test_transform(self):
        vm = VandermondeCosMult(dim_x=2, m=1, l2_lambda=0)

        vm.fit()

        # Test 1
        vm.transform(np.array([[1/3], [1/3]]))

        v_mat_truth = [[1],
                       [0.5],
                       [0.5],
                       [0.25]]

        self.assertAlmostEqual(0, np.sum((np.array(v_mat_truth) - vm.v_mat)**2))

        # Test 2
        vm.transform(np.array([[1/3, 0],
                               [1/3, 1/3]]))

        v_mat_truth = [[1, 1],
                       [0.5, 1],
                       [0.5, 0.5],
                       [0.25, 0.5]]

        self.assertAlmostEqual(0, np.sum((np.array(v_mat_truth) - vm.v_mat)**2))

    def test_v_users(self):
        vm = VandermondeCosMult(dim_x=2, m=1, l2_lambda=0)

        vm.fit()

        vm.transform(np.array([[1/3, 0],
                               [0, 1/3]]))

        rating_mat = np.array([[1, 2],
                               [np.NaN, 3],
                               [5, np.NaN]])

        v_u = vm.get_v_users([0, 2], rating_mat)
        v_u_truth = vm.v_mat[:, [0, 1, 0]]

        self.assertAlmostEqual(0, np.sum((v_u_truth - v_u)**2))


if __name__ == '__main__':
    unittest.main()
