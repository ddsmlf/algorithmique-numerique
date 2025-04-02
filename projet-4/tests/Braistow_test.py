import unittest
import numpy as np
from Bairstow import Bairstow

class TestBairstow(unittest.TestCase):

    def setUp(self):
        self.coefficients = np.array([1, -6, 11, -6]) # P(X) = X³ - -6x² + 11X - 6
        # P(X) = (X² - 3X + 2)(X - 3) + 0X + 0 ----> Q(X) = X - 3 | R = 0 | S = 0 | B = -3 | C = 2
        self.Q, self.R, self.B, self.C = np.array([1, -3]), np.array([0, 0]), -3, 2
        self.bairstow = Bairstow(np.array(self.coefficients)) 
        self.diviseur = np.array([1, self.B, self.C])
        self.U0 = np.array([-5,6]) # based on roots method b = -(r1 + r2 ) = -( 3 + 2 ) = -5 && c = r1*r2 = 3*2 = 6
        self.roots = np.array([3, 1, 2]) # Expected roots of the polynomial

    def test_residual(self):
        q, r, _ = self.bairstow.compute_factorization(self.B, self.C)
        self.assertTrue(np.allclose(r, self.R))
        self.assertTrue(np.allclose(q, self.Q))
        new_poly = np.polyadd(np.polymul(q, self.diviseur), r)
        self.assertTrue(np.allclose(new_poly, self.coefficients))

    
    def test_BC_solver(self):
        custom_U0 = np.array([-3,2])
        B, C = self.bairstow.BC_solver(custom_U0)
        self.assertTrue(np.allclose(B, self.B))
        self.assertTrue(np.allclose(C, self.C))

    
    def test_solver_with_U0(self):
        roots = self.bairstow.solve(self.U0)
        self.assertTrue(np.allclose(np.sort(roots), np.sort(self.roots)))


    def test_solver_without_U0(self):
        roots = self.bairstow.solve()
        self.assertTrue(np.allclose(np.sort(roots), np.sort(self.roots)))
        


if __name__ == "__main__":
    unittest.main()
