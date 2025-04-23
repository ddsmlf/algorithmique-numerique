import unittest
import numpy as np
from AirfoilDataLoader import AirfoilDataLoader 

class TestAirfoilDataLoader(unittest.TestCase):
    def setUp(self):
        self.file_path = 'data/fx72150a.dat'
        self.loader = AirfoilDataLoader(self.file_path)

    def test_load_airfoil_data(self):
        self.assertIsNotNone(self.loader.dimension)
        self.assertIsNotNone(self.loader.outer_x)
        self.assertIsNotNone(self.loader.outer_y)
        self.assertIsNotNone(self.loader.inner_x)
        self.assertIsNotNone(self.loader.inner_y)

    def test_dimension(self):
        self.assertIsInstance(self.loader.dimension, np.ndarray)
        self.assertEqual(len(self.loader.dimension), 2)

    def test_outer_coordinates(self):
        self.assertIsInstance(self.loader.outer_x, np.ndarray)
        self.assertIsInstance(self.loader.outer_y, np.ndarray)
        self.assertEqual(len(self.loader.outer_x), len(self.loader.outer_y))

    def test_inner_coordinates(self):
        self.assertIsInstance(self.loader.inner_x, np.ndarray)
        self.assertIsInstance(self.loader.inner_y, np.ndarray)
        self.assertEqual(len(self.loader.inner_x), len(self.loader.inner_y))

    def test_outer_x_values(self):
        self.assertTrue(np.all(self.loader.outer_x >= 0))
        self.assertTrue(np.all(self.loader.outer_x <= 1))

    def test_outer_y_values(self):
        self.assertTrue(np.all(self.loader.outer_y >= -1))
        self.assertTrue(np.all(self.loader.outer_y <= 1))

    def test_inner_x_values(self):
        self.assertTrue(np.all(self.loader.inner_x >= 0))
        self.assertTrue(np.all(self.loader.inner_x <= 1))

    def test_inner_y_values(self):
        self.assertTrue(np.all(self.loader.inner_y >= -1))
        self.assertTrue(np.all(self.loader.inner_y <= 1))

if __name__ == '__main__':
    unittest.main()