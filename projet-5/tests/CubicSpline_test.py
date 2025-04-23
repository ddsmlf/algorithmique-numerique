import unittest
import numpy as np
from CubicSpline import CubicSpline

class TestCubicSpline(unittest.TestCase):
  def setUp(self):
     # test with 3 polynomial functions with 15 original points and 1000 interpolated ones , with error limit 1e-10 
    
    self.epsilon = 1e-10
    self.test_cases = [
      {
        'name': "x -> x^2",
        'f': lambda x: x**2,
        'derivatives': (0, 2), 
        'domain': (0, 1)
      },
      {
        'name': "x -> x^3",
        'f': lambda x: x**3,
        'derivatives': (0,3),  
        'domain': (0, 1)
      },
      {
        'name': 'x -> x^3 - 2x^2 + 3x + 5',
        'f': lambda x: x**3 - 2*x**2 +3*x +5,
        'derivatives': (3, 2),
        'domain': (0, 1)
      }
    ]
    self.N = 15  
    self.N_test = 1000 
    

  def test_exact_interpolation_at_data_points(self):
      for case in self.test_cases:
          f = case['f']
          xa = np.linspace(*case['domain'], self.N)
          ya = f(xa)
          spline_correct = CubicSpline(xa, ya, *case['derivatives'])
          spline_natural = CubicSpline(xa, ya, natural_spline=True)

          for i, x in enumerate(xa):
              y_correct = spline_correct.interpolate(x)
              y_natural = spline_natural.interpolate(x)
              
              self.assertAlmostEqual(y_correct, ya[i],
                  msg=f"Failed at {case['name']} (correct spline) with x={x}, expected={ya[i]}, got={y_correct}, error={abs(ya[i]-y_correct)}")
              self.assertAlmostEqual(y_natural, ya[i],
                  msg=f"Failed at {case['name']} (natural spline) with x={x}, expected={ya[i]}, got={y_natural}, error={abs(ya[i]-y_natural)}")

if __name__ == '__main__':
    unittest.main()