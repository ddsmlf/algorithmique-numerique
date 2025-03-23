import unittest
import numpy as np
from Householder import Householder
from tools import generate_random_UV

class TestHouseholderMatrix(unittest.TestCase):

    def setUp(self):
        """Initialisation des données de test"""
        self.U_valid = np.array([[3], [4], [0]]) 
        self.V_valid = np.array([[0], [0], [5]]) 
        self.U_invalid_shape = np.array([[1], [2]])  
        self.V_invalid_shape = np.array([[1], [2], [3], [4]])  
        self.U_invalid_norm = np.array([[1], [2], [3]]) 
        self.V_invalid_norm = np.array([[1], [1], [1]]) 
        self.gpu = False

    def test_initialization_valid(self):
        """Test de l'initialisation valide de la classe Householder"""
        householder = Householder(self.U_valid, self.V_valid, gpu=self.gpu)
        self.assertIsInstance(householder, Householder)
        U = householder.U.get() if self.gpu else householder.U
        V = householder.V.get() if self.gpu else householder.V
        np.testing.assert_array_equal(U, self.U_valid)
        np.testing.assert_array_equal(V, self.V_valid)

    def test_initialization_invalid_shape(self):
        """Test de l'initialisation avec des formes incorrectes pour U et V"""
        with self.assertRaises(ValueError):
            Householder(self.U_invalid_shape, self.V_valid, gpu=self.gpu)

        with self.assertRaises(ValueError):
            Householder(self.U_valid, self.V_invalid_shape, gpu=self.gpu)

    def test_initialization_invalid_norm(self):
        """Test de l'initialisation avec des normes différentes pour U et V"""
        with self.assertRaises(ValueError):
            Householder(self.U_invalid_norm, self.V_valid, gpu=self.gpu)

        with self.assertRaises(ValueError):
            Householder(self.U_valid, self.V_invalid_norm, gpu=self.gpu)

    def test_calculate_N(self):
        """Test du calcul de N"""
        householder = Householder(self.U_valid, self.V_valid, gpu=self.gpu)
        W = self.U_valid - self.V_valid
        norm_W = np.linalg.norm(W)
        expected_N = W / norm_W if not np.isclose(norm_W, 0) else None
        N = householder.N.get() if self.gpu else householder.N
        np.testing.assert_array_almost_equal(N, expected_N)

    def test_apply_transformation(self):
        """Test de la méthode d'application de la transformation de Householder"""
        householder = Householder(self.U_valid, self.V_valid, gpu=self.gpu)
        X = np.array([[1], [2], [3]]) 
        transformed_X = householder.apply_transformation(X)
        self.assertEqual(transformed_X.shape, X.shape)

    def test_apply_transformation_to_matrix(self):
        """Test de la méthode d'application de la transformation de Householder à une matrice"""
        householder = Householder(self.U_valid, self.V_valid, gpu=self.gpu)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
        transformed_X = householder.apply_transformation_to_matrix(X)
        transformed_X = transformed_X.get() if self.gpu else transformed_X
        expected_transformed_X = householder.H.get() @ X if self.gpu else householder.H @ X
        np.testing.assert_array_almost_equal(transformed_X, expected_transformed_X)

    def test_visualize_transformation_2d(self):
        """Test de la visualisation de la transformation pour un espace 2D"""
        U_2d, V_2d = generate_random_UV(2)
        householder = Householder(U_2d, V_2d, gpu=self.gpu)
        householder.visualize_transformation()

    def test_visualize_transformation_3d(self):
        """Test de la visualisation de la transformation pour un espace 3D"""
        U_3d, V_3d = generate_random_UV(3)
        householder = Householder(U_3d, V_3d, gpu=self.gpu)
        householder.visualize_transformation()

    def test_display_matrices(self):
        """Test de l'affichage des matrices de Householder"""
        householder = Householder(self.U_valid, self.V_valid, gpu=self.gpu)
        try:
            householder.display_matrices()
        except Exception as e:
            self.fail(f"display_matrices a échoué avec l'exception: {e}")

if __name__ == "__main__":
    unittest.main()
