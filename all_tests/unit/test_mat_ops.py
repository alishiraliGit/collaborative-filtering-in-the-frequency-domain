import unittest
import numpy as np

from app.utils.mat_ops import vectorize_rows


class TestMatOps(unittest.TestCase):

    def test_vectorize_mat(self):
        rating_mat = np.array([[1, 2],
                               [np.NaN, 3],
                               [5, np.NaN],
                               [np.NaN, 3]])

        vec_rat_truth = np.array([[1], [2], [5], [3]])

        vec_rat = vectorize_rows([0, 2, 3], rating_mat)

        self.assertAlmostEqual(0, np.sum((vec_rat_truth - vec_rat)**2))


if __name__ == '__main__':
    unittest.main()
