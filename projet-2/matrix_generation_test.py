import unittest
import numpy as np
from matrix_generation import generate_symmetric_positive_definite_matrix, generate_sparse_symmetric_positive_definite_matrix

def is_sdp(x):
    """
    Vérifie si une matrice est symétrique définie positive
    inputs:
    x (np.ndarray): Matrice.
    outputs:
    bool: True si la matrice est symétrique définie positive, False sinon.
    """
    return np.all(x == x.T) and np.all(np.linalg.eigvals(x) > 0)

class TestMatrixGeneration(unittest.TestCase):
    """
    TestMatrixGeneration is a test case class for testing matrix generation functions.

    Methods
    -------
    test_generate_symmetric_positive_definite_matrix():
        Tests the generate_symmetric_positive_definite_matrix function for different matrix sizes.
        Verifies that the generated matrix has the correct shape and is symmetric positive definite.

    test_generate_sparse_symmetric_positive_definite_matrix():
        Tests the generate_sparse_symmetric_positive_definite_matrix function for different matrix sizes and extra diagonal terms.
        Verifies that the generated matrix has the correct shape, is symmetric positive definite, and has the expected sparsity.
    """

    def test_generate_symmetric_positive_definite_matrix(self):
        for n in [5, 10, 50, 100]:
            with self.subTest(n=n):
                matrix = generate_symmetric_positive_definite_matrix(n)
                self.assertEqual(matrix.shape, (n, n))
                self.assertTrue(is_sdp(matrix))

    def test_generate_sparse_symmetric_positive_definite_matrix(self):
        #on choisit les
        for n, extra_diagonal_terms in [(5, 4), (10, 3), (50, 15), (100, 30)]:
            with self.subTest(n=n, extra_diagonal_terms=extra_diagonal_terms):
                matrix = generate_sparse_symmetric_positive_definite_matrix(n, extra_diagonal_terms)
                self.assertEqual(matrix.shape, (n, n))
                self.assertTrue(is_sdp(matrix))
                zero_elements = n*n-np.count_nonzero(matrix)
                self.assertLessEqual(zero_elements, extra_diagonal_terms)

if __name__ == '__main__':
    unittest.main()
