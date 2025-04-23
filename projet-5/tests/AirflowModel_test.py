import unittest
from AirflowModel import AirflowModel

class TestAirflowModel(unittest.TestCase):
    def setUp(self):
        self.airfoil_name = 'fx72150a'
        self.airfoil = AirflowModel(self.airfoil_name)

    def test_init(self):
        self.assertIsNotNone(self.airfoil.upper_surface_x)
        self.assertIsNotNone(self.airfoil.upper_surface_y)
        self.assertIsNotNone(self.airfoil.lower_surface_x)
        self.assertIsNotNone(self.airfoil.lower_surface_y)
        self.assertIsNotNone(self.airfoil.upper_surface_spline)
        self.assertIsNotNone(self.airfoil.lower_surface_spline)

    def test_get_airfoil_height(self):
        min_height, max_height = self.airfoil.get_airfoil_height()
        self.assertLess(min_height, max_height)

    def test_get_disturbance_curves(self):
        upper_curves, lower_curves = self.airfoil.get_disturbance_curves(num_curves_above=20, num_curves_below=20)
        self.assertEqual(len(upper_curves), 20)
        self.assertEqual(len(lower_curves), 20)

    def test_pressure_map(self):
        self.airfoil.pressure_map(method='midpoint', num_curves=[100,100])

    def test_plot_airfoil(self):
        self.airfoil.plot_airfoil(num_curves=[20, 5])

if __name__ == '__main__':
    unittest.main()