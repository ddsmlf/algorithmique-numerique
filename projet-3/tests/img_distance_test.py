from tools import euclidian_distance, manhattan_distance
import math
import unittest
import numpy as np

class TestImageDistance(unittest.TestCase):

    def test_euclidian_distance_result(self):
        """
        Tests if the Euclidean distance function calculates the expected result.
        """
        A = np.zeros((3, 3, 3))
        B = np.array([[[1, 1, 1],[1, 1, 1],[1, 1, 1]],
                     [[1, 1, 1],[1, 1, 1],[1, 1, 1]],
                     [[1, 1, 1],[1, 1, 1],[1, 1, 1]]])
        
        self.assertTrue(euclidian_distance(A, B) == math.sqrt(27))

    def test_euclidian_distance_shape_error(self):
        """
        Tests if the Euclidean distance raises an error for mismatched shapes.
        """
        A = np.zeros((1, 3))
        B = np.zeros((2, 3))
        self.assertRaises(ValueError, euclidian_distance, A, B)

    def test_manhattan_distance(self):
        """
        Tests if the Manhattan distance function calculates the expected result.
        """
        A = np.zeros((3, 3, 3))
        B = np.array([[[1, 1, 1],[1, 1, 1],[1, 1, 1]],
                     [[1, 1, 1],[1, 1, 1],[1, 1, 1]],
                     [[1, 1, 1],[1, 1, 1],[1, 1, 1]]])
        self.assertTrue(manhattan_distance(A, B) == 3*3*3)

    def test_manhattan_distance_shape_error(self):
        """
        Tests if the Manhattan distance raises an error for mismatched shapes.
        """
        A = np.zeros((1, 3))
        B = np.zeros((2, 3))
        self.assertRaises(ValueError, manhattan_distance, A, B)


if __name__ == "__main__":
    unittest.main()
