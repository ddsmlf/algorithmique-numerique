import unittest
import numpy as np
from CauchySolver import CauchySolver


class TestCauchySolver(unittest.TestCase):
    def setUp(self):
        def f_1(t, y):
            return y / (1 + t**2)

        def solve_1(t):
            return np.exp(np.arctan(t))

        def f_2(_, y):
            return np.array([-y[1], y[0]])

        def solve_2(t):
            return np.array([np.cos(t), np.sin(t)])

        self.tests_functions = [
            (f_1, solve_1, 0, 1),
            (f_2, solve_2, 0, np.array([1, 0]))
        ]
        self.methods = ['euler', 'runge_kutta_4', 'midpoint', 'heun']

    def test_methods(self):
        for method in self.methods:
            with self.subTest(method=method):
                for f, solve, t0, y0 in self.tests_functions:
                    solver = CauchySolver(f)
                    try:
                        tn, y = solver.meth_epsilon(y0, t0, method=method)
                        expected_y_at_tn = [solve(t) for t in tn]
                        np.testing.assert_almost_equal(y, expected_y_at_tn, decimal=3)
                    except AssertionError as e:
                        print(f"Error with method '{method}' and function '{f.__name__}'")
                        raise e


if __name__ == '__main__':
    unittest.main()
