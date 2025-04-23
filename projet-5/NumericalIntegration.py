import numpy as np
from math import *

class NumericalIntegration:
    """
    Class for numerical integration of a function.
    
    Attributes:
    - lower_bound (float): Lower bound of the integration interval.
    - upper_bound (float): Upper bound of the integration interval.
    - function (callable): Function to be integrated.
    - max_iterations (int): Maximum number of iterations for the integration.
    - plot (bool): Whether to plot the integral values.
    """

    def __init__(self, function, lower_bound, upper_bound, max_iterations, plot=False):
        """
        Initialize the NumericalIntegration object.
        
        Parameters:
        - function (callable): Function to be integrated.
        - lower_bound (float): Lower bound of the integration interval.
        - upper_bound (float): Upper bound of the integration interval.
        - max_iterations (int): Maximum number of iterations for the integration.
        - plot (bool): Whether to plot the integral values.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.function = function
        self.max_iterations = max_iterations
        self.plot = plot
        if plot:
            self.integral_values = []

    def left_rectangle_rule(self, num_subintervals):
        """
        Integrate the function using the left rectangle rule.
        
        Parameters:
        - num_subintervals (int): Number of subintervals for the integration.
        
        Returns:
        - integral (float): Approximate value of the integral.
        - integral_values (list): List of integral values for plotting (if plot=True).
        """
        if self.plot:
            self.integral_values = []
        subinterval_width = (self.upper_bound - self.lower_bound) / num_subintervals
        integral = 0
        for i in range(num_subintervals):
            integral += self.function(self.lower_bound + i * subinterval_width)
            if self.plot:
                self.integral_values.append(integral * subinterval_width)
        if self.plot:
            return integral * subinterval_width, self.integral_values
        return integral * subinterval_width

    def right_rectangle_rule(self, num_subintervals):
        """
        Integrate the function using the right rectangle rule.
        
        Parameters:
        - num_subintervals (int): Number of subintervals for the integration.
        
        Returns:
        - integral (float): Approximate value of the integral.
        - integral_values (list): List of integral values for plotting (if plot=True).
        """
        if self.plot:
            self.integral_values = []
        subinterval_width = (self.upper_bound - self.lower_bound) / num_subintervals
        integral = 0
        for i in range(1, num_subintervals + 1):
            integral += self.function(self.lower_bound + i * subinterval_width)
            if self.plot:
                self.integral_values.append(integral * subinterval_width)
        if self.plot:
            return integral * subinterval_width, self.integral_values
        return integral * subinterval_width

    def midpoint_rule(self, num_subintervals):
        """
        Integrate the function using the midpoint rule.
        
        Parameters:
        - num_subintervals (int): Number of subintervals for the integration.
        
        Returns:
        - integral (float): Approximate value of the integral.
        - integral_values (list): List of integral values for plotting (if plot=True).
        """
        if self.plot:
            self.integral_values = []
        subinterval_width = (self.upper_bound - self.lower_bound) / num_subintervals
        integral = 0
        for i in range(num_subintervals):
            integral += self.function(self.lower_bound + i * subinterval_width + subinterval_width / 2)
            if self.plot:
                self.integral_values.append(integral * subinterval_width)
        if self.plot:
            return integral * subinterval_width, self.integral_values
        return integral * subinterval_width

    def trapezoidal_rule(self, num_subintervals):
        """
        Integrate the function using the trapezoidal rule.
        
        Parameters:
        - num_subintervals (int): Number of subintervals for the integration.
        
        Returns:
        - integral (float): Approximate value of the integral.
        - integral_values (list): List of integral values for plotting (if plot=True).
        """
        if self.plot:
            self.integral_values = []
        subinterval_width = (self.upper_bound - self.lower_bound) / num_subintervals
        integral = (self.function(self.lower_bound) + self.function(self.upper_bound)) / 2
        for i in range(1, num_subintervals):
            integral += self.function(self.lower_bound + i * subinterval_width)
            if self.plot:
                self.integral_values.append(integral * subinterval_width)
        if self.plot:
            return integral * subinterval_width, self.integral_values
        return integral * subinterval_width

    def simpson_rule(self, num_subintervals):
        """
        Integrate the function using Simpson's rule.
        
        Parameters:
        - num_subintervals (int): Number of subintervals for the integration.
        
        Returns:
        - integral (float): Approximate value of the integral.
        - integral_values (list): List of integral values for plotting (if plot=True).
        """
        if self.plot:
            self.integral_values = []
        subinterval_width = (self.upper_bound - self.lower_bound) / num_subintervals
        integral = (self.function(self.lower_bound) + self.function(self.upper_bound)) / 6
        for i in range(1, num_subintervals):
            integral += (4 if i % 2 == 1 else 2) / 3 * self.function(self.lower_bound + i * subinterval_width)
            if self.plot:
                self.integral_values.append(integral * subinterval_width)
        if self.plot:
            return integral * subinterval_width, self.integral_values
        return integral * subinterval_width

    def integrate_to_epsilon(self, method, epsilon):
        """
        Integrate the function to a specified epsilon using the given method.
    
        Parameters:
        - method (str): Method to use for integration (e.g. 'left_rectangle','simpson', etc.).
        - epsilon (float): Desired precision for the integration.
    
        Returns:
        - integral (float): Approximate value of the integral.
        """
        n = 1
        h = (self.upper_bound - self.lower_bound)
        t = 3 if method == "midpoint" else 2
        I_n = getattr(self, f"{method}_rule")(n)

        if method == "simpson": f_a, f_b, S1 = self.function(self.lower_bound), self.function(self.upper_bound), 0
        
        while n <= self.max_iterations:
            h /= t
            n *= t
            if method == "simpson": S1 += sum(self.function(self.lower_bound + i * h) for i in range(1, n, t))
            new_points = (sum(self.function(self.lower_bound + (t * k + 1) * h) +(self.function(self.lower_bound + (3 * k + 2) * h) if t==3 else 0) for k in range(n // t)))  if method != "simpson" else (f_a + f_b + t * S1 + 4 * sum(self.function(self.lower_bound + (i + 1 / t) * h) for i in range(n)))
            I_tn = ((1 / t) * I_n + h * new_points) if method != "simpson" else ((h / 6) * new_points)

            if abs(I_tn - I_n) < epsilon:
                return I_tn
            
            I_n = I_tn
        return I_n

def calculate_curve_length(function, upper_limit, integration_method="simpson", max_iterations=1000):
    """
    Calculate the length of a curve defined by a function over a given interval.
    
    Parameters:
    - function (callable): Function defining the curve.
    - upper_limit (float): Upper limit of the interval.
    - integration_method (str): Method to use for integration (e.g.'simpson', 'trapezoidal', etc.).
    
    Returns:
    - curve_length (float): Length of the curve.
    """
    def numerical_derivative(function, x, lower_limit, upper_limit, h=1e-6):
        """
        Calculate the numerical derivative of a function at a point.
        
        Parameters:
        - function (callable): Function to differentiate.
        - x (float): Point at which to evaluate the derivative.
        - lower_limit (float): Lower limit of the interval.
        - upper_limit (float): Upper limit of the interval.
        - h (float): Step size for the numerical derivative.
        
        Returns:
        - derivative (float): Numerical derivative of the function at the point.
        """
        if x - h < lower_limit:
            # Forward difference near the left edge
            return (function(x + h) - function(x)) / h
        elif x + h > upper_limit:
            # Backward difference near the right edge
            return (function(x) - function(x - h)) / h
        else:
            # Central difference in the middle
            return (function(x + h) - function(x - h)) / (2 * h)

    def integrand(x):
        return sqrt(1 + numerical_derivative(function, x, 0, upper_limit)**2)

    integrator = NumericalIntegration(integrand, 0, upper_limit, max_iterations)
    curve_length = integrator.integrate_to_epsilon(integration_method, 1e-6)
    return curve_length
