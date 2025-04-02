import unittest
import numpy as np
from NewtonRaphsonSolver import NewtonRaphsonSolver  # Importer la classe NewtonRaphsonSolver

class TestNewtonRaphsonSolver(unittest.TestCase):
    def setUp(self):
            def f(x):
                return x**2 - 2
            self.function = f
            def J(x):
                return 2*x
            self.Jacobian = J
        
    def test_convergence(self):
        solver = NewtonRaphsonSolver(self.function, self.Jacobian, input_dim=1, output_dim=1)
        U0 = np.array([1000.0])
        root = solver.solve(U0)
        self.assertAlmostEqual(root[0], np.sqrt(2), places=5)
        U0 = np.array([10.0])
        root = solver.solve(U0)
        self.assertAlmostEqual(root[0], np.sqrt(2), places=5)


    def test_divergence(self):
        solver = NewtonRaphsonSolver(self.function, self.Jacobian, input_dim=1, output_dim=1)
        # Test de divergence

if __name__ == '__main__':
    unittest.main() 