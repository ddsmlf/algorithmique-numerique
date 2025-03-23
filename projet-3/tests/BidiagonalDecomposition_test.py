import unittest
import numpy as np
from Bidiagonalization import Bidiagonalization

class TestBidiagonalization(unittest.TestCase):
    def setUp(self):
        """
        Initializes test setup.
        """
        self.gpu = False

    def test_bidiagonalization_square_matrix(self):
        """
        Tests bidiagonalization and assertions on a square 3x3 matrix.
        """
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        bidiag = Bidiagonalization(A, gpu=self.gpu)
        Qleft, BD, Qright = bidiag.compute()
        
        self.assertTrue(np.allclose(Qleft @ BD @ Qright, A))

    def test_bidiagonalization_rectangular_matrix(self):
        """
        Tests bidiagonalization and assertions on a rectangular 2x3 matrix.
        """
        A = np.array([[1, 2, 3], [4, 5, 6]])
        bidiag = Bidiagonalization(A, gpu=self.gpu)
        Qleft, BD, Qright = bidiag.compute()
        
        self.assertTrue(np.allclose(Qleft @ BD @ Qright, A))

    def test_bidiagonalization_identity_matrix(self):
        """
        Tests bidiagonalization and assertions on a identity matrix.
        """
        A = np.eye(3)
        bidiag = Bidiagonalization(A, gpu=self.gpu)
        Qleft, BD, Qright = bidiag.compute()
        
        self.assertTrue(np.allclose(Qleft @ BD @ Qright, A))

    def test_bidiagonalization_zero_matrix(self):
        """
        Tests bidiagonalization and assertions on a zero matrix.
        """
        A = np.zeros((3, 3))
        bidiag = Bidiagonalization(A, gpu=self.gpu)
        Qleft, BD, Qright = bidiag.compute()
        
        self.assertTrue(np.allclose(Qleft @ BD @ Qright, A))

    def test_bidiagonalization_single_value_matrix(self):
        """
        Tests bidiagonalization and assertions on a single-value matrix.
        """
        A = np.array([[5]])
        bidiag = Bidiagonalization(A, gpu=self.gpu)
        Qleft, BD, Qright = bidiag.compute()
        
        self.assertTrue(np.allclose(Qleft @ BD @ Qright, A))

if __name__ == "__main__":
    unittest.main()

