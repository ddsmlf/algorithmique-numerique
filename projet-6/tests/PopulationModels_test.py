import unittest
import numpy as np
from PopulationModels import PopulationModels

class TestPopulationModels(unittest.TestCase):

    def test_malthus_model(self):
        model = PopulationModels(gamma=0.1)
        t = 0
        N = 100
        result = model.malthus(t, N)
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_verhulst_model(self):
        model = PopulationModels(gamma=0.1, kappa=500)
        t = 0
        N = 100
        result = model.verhulst(t, N)
        self.assertAlmostEqual(result, 8.0, places=4)

    def test_lotka_volterra_model(self):
        model = PopulationModels(gamma=0.1, b=0.02, c=0.01, d=0.1)
        t = 0
        y = [40, 9]
        result = model.lotka_volterra(t, y)
        self.assertAlmostEqual(result[0], -3.2, places=4)
        self.assertAlmostEqual(result[1], 2.7, places=4)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            PopulationModels(gamma=0.1, kappa=500, b=0.02, c=0.01, d=0.1)

        with self.assertRaises(ValueError):
            PopulationModels(gamma=0.1, b=0.02, c=0.01)

    def test_find_singular_points(self):
        model = PopulationModels(gamma=0.1, b=0.02, c=0.01, d=0.1)
        singular_points = model.find_singular_points()
        self.assertEqual(singular_points, [(0, 0), (10.0, 5.0)])

    def test_find_period(self):
        model = PopulationModels(gamma=0.1, b=0.02, c=0.01, d=0.1)
        t = np.linspace(0, 100, 1000)
        y = [[40 + 5 * np.sin(2 * np.pi * t_i / 20), 9 + 2 * np.sin(2 * np.pi * t_i / 20)] for t_i in t]
        period, periods, _, _, _ = model.find_period(t, y)
        self.assertAlmostEqual(period, -100, delta=1)

    def test_solve_malthus(self):
        model = PopulationModels(gamma=0.1)
        t0 = 0
        t_end = 10
        N0 = [100]
        t, N = model.solve(N0, t0, t_end=t_end, method='euler')
        self.assertEqual(len(t), len(N))
        self.assertGreater(N[-1][0], N0[0])

    def test_solve_verhulst(self):
        model = PopulationModels(gamma=0.1, kappa=500)
        t0 = 0
        t_end = 10
        N0 = [100]
        t, N = model.solve(N0, t0, t_end=t_end, method='runge_kutta_4')
        self.assertEqual(len(t), len(N))
        self.assertLess(N[-1][0], 500)

    def test_solve_lotka_volterra(self):
        model = PopulationModels(gamma=0.1, b=0.02, c=0.01, d=0.1)
        t0 = 0
        t_end = 10
        N0 = [40, 9]
        t, y = model.solve(N0, t0, t_end=t_end, method='midpoint')
        self.assertEqual(len(t), len(y))
        self.assertGreater(y[-1][0], 0)
        self.assertGreater(y[-1][1], 0)

if __name__ == '__main__':
    unittest.main()