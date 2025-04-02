import unittest
import numpy as np
from LagrangianPoints import LagrangianPoints

class TestLagrangianPoints(unittest.TestCase):
    def setUp(self):
        self.lagrangian_points = LagrangianPoints(
            pos1 = np.array([0.0, 0.0]), k1 = 1.0, 
            pos2 = np.array([1.0, 0.0]), k2 = 0.01,
            k_c = 1.0
        )

    def test_force_at_U_test(self):
        U = np.array([1.5, 0.0])
        result = self.lagrangian_points.calc_total_force(U)
        expected = np.array([1.00565457, 0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_jacobian_at_U_test(self):
        U = np.array([1.5, 0.0])
        result = self.lagrangian_points.calc_total_jac(U)
        expected = np.array([[1.75259259, 0.0], [0.0, 0.6237037]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_newton_raphson_convergence(self):
        initial_guess = np.array([1.5, 0.0])
        root = self.lagrangian_points.solve(initial_guess)
        self.assertLess(np.linalg.norm(self.lagrangian_points.total_force(root)), 1e-6)

if __name__ == "__main__":
    unittest.main()
