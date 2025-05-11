import unittest
import numpy as np
from skimage import data
from ImageFilter import ImageFilter

class TestImageFilter(unittest.TestCase):
    def setUp(self):
        self.image = data.camera()
        self.filter = ImageFilter(self.image, dt_heat=0.1, dt_pm=0.2, max_iter=10, sigma=1.0)

    def test_heat_equation(self):
        filtered_image = self.filter.solve_heat_equation()
        self.assertEqual(filtered_image.shape, self.image.shape)
        self.assertTrue(np.all(filtered_image >= 0))

    def test_perona_malik(self):
        filtered_image = self.filter.solve_perona_malik()
        self.assertEqual(filtered_image.shape, self.image.shape)
        self.assertTrue(np.all(filtered_image >= 0))  

    def test_gaussian_convolution(self):
        convolved_image = self.filter.gaussian_convolution(self.image)
        self.assertEqual(convolved_image.shape, self.image.shape)
        self.assertTrue(np.all(convolved_image >= 0)) 

    def test_gradient(self):
        grad_x, grad_y = self.filter._gradient(self.image.flatten())
        self.assertEqual(grad_x.shape, self.image.shape)
        self.assertEqual(grad_y.shape, self.image.shape)

    def test_divergence(self):
        grad_x, grad_y = self.filter._gradient(self.image.flatten())
        divergence = self.filter._divergence((grad_x, grad_y))
        self.assertEqual(divergence.shape, self.image.flatten().shape)

    def test_perona_malik_function(self):
        grad_norm = np.random.rand(*self.image.shape)
        pm_function = self.filter._perona_malik_function(grad_norm)
        self.assertEqual(pm_function.shape, self.image.shape)
        self.assertTrue(np.all(pm_function >= 0))  

if __name__ == '__main__':
    unittest.main()