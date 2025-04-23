import unittest
import numpy as np
import math as math
from NumericalIntegration import NumericalIntegration, calculate_curve_length

class TestNumericalIntegration(unittest.TestCase):

	def setUp(self):
		self.epsilon = 1e-4
		self.N = 10000000

		self.test_cases = [
			{
				'name': "x-> 2",
				'a': 0,
				'b': 10,
				'f': lambda x: 2,
				'expected': 20,
			},
			{
				'name': "x-> x",
				'a': 0,
				'b': 10,
				'f': lambda x: x,
				'expected': 50,
			},
			{
				'name': "x-> x**2+x",
				'a': 0,
				'b': 10,
				'f': lambda x: x**2 + x,
				'expected': 383.333333333333333333,
			},
			{
				'name': "x-> x**3+3x+16",
				'a': 0,
				'b': 10,
				'f': lambda x: x**2 + 3*x + 16,
				'expected': 643.33333333333333333,
			},
		]
		

		self.curve_test_cases = [
			{
				'name': "x-> x",
				'a': 0,
				'b': 1,
				'f': lambda x: x,
				'precision': 3,
				'expected': np.sqrt(2),
			},
			{
				'name': "x-> 2x",
				'a': 0,
				'b': 1,
				'f': lambda x: 2*x,
				'precision': 2,
				'expected':np.sqrt(5),
			}
		]


	def test_left_rectangle_rule(self):
		N=self.N
		precision = 4
		for i in range(2):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, 100)

			result_left_rectangle_rule = integrCase.left_rectangle_rule(N)
			np.testing.assert_almost_equal(result_left_rectangle_rule, expected, precision)

	def test_right_rectangle_rule(self):
		N=self.N
		precision = 4
		for i in range(2):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, 100)

			result_right_rectangle_rule = integrCase.right_rectangle_rule(N)
			np.testing.assert_almost_equal(result_right_rectangle_rule, expected, precision)

	def test_midpoint_rule(self):
		N=self.N
		precision = 7
		for i in range(2):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, 100)

			result_midpoint_rule = integrCase.midpoint_rule(N)
			np.testing.assert_almost_equal(result_midpoint_rule, expected, precision)


	def test_trapezoidal_rule(self):
		N=self.N
		precision = 7
		for i in range(3):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, 100)

			result_trapezoidal_rule = integrCase.trapezoidal_rule(N)
			np.testing.assert_almost_equal(result_trapezoidal_rule, expected, precision)


	def test_simpson_rule(self):
		N = self.N
		precision= 3
		for i in range(2):
			testedCase= self.test_cases[i+2]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, 100)

			result_simpson_rule = integrCase.simpson_rule(N)
			np.testing.assert_almost_equal(result_simpson_rule, expected, precision)

	def test_epsilon_left_rectangle_rule(self):
		epsilon = self.epsilon
		N=self.N
		precision = 4
		for i in range(2):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, N)

			result_left_rectangle_rule = integrCase.integrate_to_epsilon("left_rectangle", epsilon)
			np.testing.assert_almost_equal(result_left_rectangle_rule, expected, precision)

	def test_epsilon_right_rectangle_rule(self):
		epsilon = self.epsilon
		N=self.N
		precision = 4
		for i in range(2):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, N)

			result_right_rectangle_rule = integrCase.integrate_to_epsilon("right_rectangle", epsilon)
			np.testing.assert_almost_equal(result_right_rectangle_rule, expected, precision)

	def test_epsilon_midpoint_rule(self):
		epsilon = self.epsilon
		N=self.N
		precision = 4
		for i in range(2):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, N)

			result_midpoint_rule = integrCase.integrate_to_epsilon("midpoint", epsilon)
			np.testing.assert_almost_equal(result_midpoint_rule, expected, precision)

	def test_epsilon_trapezoidal_rule(self):
		epsilon = self.epsilon
		N=self.N
		precision = 4
		for i in range(3):
			testedCase= self.test_cases[i]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, N)

			result_trapezoidal_rule = integrCase.integrate_to_epsilon("trapezoidal", epsilon)
			np.testing.assert_almost_equal(result_trapezoidal_rule, expected, precision)

	def test_epsilon_simpson_rule(self):
		epsilon = 1e-10
		N = self.N
		precision= 8
		for i in range(2):
			testedCase= self.test_cases[i+2]
			f = testedCase['f']
			a = testedCase['a']
			b = testedCase['b']
			expected = testedCase['expected']
			integrCase = NumericalIntegration(f, a, b, N)

			result_simpson_rule = integrCase.integrate_to_epsilon("simpson", epsilon)
			np.testing.assert_almost_equal(result_simpson_rule, expected, precision)


	def test_calculate_curve_length(self):
		for test_case in self.curve_test_cases:
			result = calculate_curve_length(test_case['f'], test_case['b'], "simpson")
			self.assertAlmostEqual(result, test_case['expected'], places=test_case['precision'], msg=f"Test case: {test_case['name']}")


if __name__ == "__main__":
	unittest.main()
