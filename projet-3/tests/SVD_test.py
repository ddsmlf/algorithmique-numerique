import unittest
import numpy as np
from SVD import SVD

class TestSVD(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
        # self.A = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36, 37, 38, 39, 40], [41, 42, 43, 44, 45, 46, 47, 48, 49, 50], [51, 52, 53, 54, 55, 56, 57, 58, 59, 60], [61, 62, 63, 64, 65, 66, 67, 68, 69, 70], [71, 72, 73, 74, 75, 76, 77, 78, 79, 80], [81, 82, 83, 84, 85, 86, 87, 88, 89, 90], [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]])
        self.svd = SVD(self.A, tol=1e-6, gpu=False)

    def test_svd_decomposition(self):
        """
        Tests the SVD decomposition and compares it with numpy's SVD.
        """
        U, S, V = self.svd.apply_SVD()
        _, np_S, _ = np.linalg.svd(self.A)
        if self.svd.gpu:
            if hasattr(U, 'get'):
                U = U.get()
            if hasattr(S, 'get'):
                S = S.get()
            if hasattr(V, 'get'):
                V = V.get()
        self.assertTrue(np.allclose(S, np_S, atol=1e-6))
        self.assertTrue(np.allclose(U @ U.T, np.eye(U.shape[0]), atol=1e-6))
        self.assertTrue(np.allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-6))


        self.assertTrue(np.allclose(self.A, U @ np.diag(S) @ V, atol=1e-6)) 


    def test_convergence_plot(self):
        """
        Tests the convergence of the SVD and plot the off-diagonal norms.
        """
        _, _, _, off_diag_norms, _ = self.svd.apply_SVD(plot_convergence=True)
        if self.svd.gpu:
            off_diag_norms = [norm.get() for norm in off_diag_norms]
        self.assertGreater(len(off_diag_norms), 0)
        self.assertLess(off_diag_norms[-1], self.svd.tol)

    def test_givens_rotation_identity(self):
        B = np.eye(3)
        Q, R = self.svd.givens_rotation(B)
        if self.svd.gpu:
            if hasattr(Q, 'get'):
                Q = Q.get()
            if hasattr(R, 'get'):
                R = R.get()
        self.assertTrue(np.allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-6))
        self.assertTrue(np.allclose(R, np.triu(R), atol=1e-6))
        self.assertTrue(np.allclose(Q @ R, B, atol=1e-6))

    def test_givens_rotation_zero_matrix(self):
        B = np.zeros((4, 4))
        Q, R = self.svd.givens_rotation(B)
        if self.svd.gpu:
            if hasattr(Q, 'get'):
                Q = Q.get()
            if hasattr(R, 'get'):
                R = R.get()
        self.assertTrue(np.allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-6))
        self.assertTrue(np.allclose(R, np.triu(R), atol=1e-6))
        self.assertTrue(np.allclose(Q @ R, B, atol=1e-6))

    def test_givens_rotation_random_matrix(self):
        from Bidiagonalization import Bidiagonalization
        np.random.seed(0)
        B = np.random.rand(5, 5)
        B = Bidiagonalization(B).compute()[1]
        Q, R = self.svd.givens_rotation(B)
        if self.svd.gpu:
            if hasattr(Q, 'get'):
                Q = Q.get()
            if hasattr(R, 'get'):
                R = R.get()
        self.assertTrue(np.allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-6))
        self.assertTrue(np.allclose(R, np.triu(R), atol=1e-6))
        self.assertTrue(np.allclose(Q @ R, B, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
