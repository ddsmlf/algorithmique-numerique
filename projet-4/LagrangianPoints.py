import numpy as np
from NewtonRaphsonSolver import NewtonRaphsonSolver
import matplotlib.pyplot as plt
import matplotlib

class LagrangianPoints:
    def __init__(self, pos1, k1, pos2, k2, k_c=None, pos_e=np.array([0.0, 0.0]), k_e=0.0):
        """
        Initializes the Lagrangian points solver.

        Parameters:
        pos1 (np.array): Coordinates [x, y] of the first mass.
        k1 (float): Gravitational constant of the first mass.
        pos2 (np.array): Coordinates [x, y] of the second mass.
        k2 (float): Gravitational constant of the second mass.
        k_c (float): Centrifugal constant. Defaults to the sum of k1 and k2.
        pos_e (np.array): Coordinates [x, y] of the elastic mass. Defaults to [0.0, 0.0].
        k_e (float): Elastic constant. Defaults to 0.0.
        """
        self.eps = np.finfo(float).eps
        if k_c is None:
            self.k_c = k1 + k2
        else:
            self.k_c = k_c
        self.barycenter = self.__calc_barycenter(k1, k2, pos1, pos2)
        self.c_force, self.c_jac = self.__centrifugal_force(self.barycenter, self.k_c)
        self.g_force1, self.g_jac1 = self.__gravitational_force(pos1, k1)
        self.g_force2, self.g_jac2 = self.__gravitational_force(pos2, k2)
        self.e_force, self.e_jac = self.__elastic_force(pos_e, k_e) 
        self.pos1 = pos1
        self.pos2 = pos2
        self.total_force = self.calc_total_force
        self.total_jac = self.calc_total_jac

    def __calc_barycenter(self, k1, k2, pos1, pos2):
        """
        Calculates the barycenter of two masses.
        """
        return (k1 * pos1 + k2 * pos2) / (k1 + k2)

    def __centrifugal_force(self, pos, k):
        """
        Returns a centrifugal force function and its Jacobian.
        """
        def force(x):
            return k * (x - pos)
        
        def jac(x):
            return k * np.eye(2)

        return force, jac


    def __gravitational_force(self, pos, k):
        """
        Returns a gravitational force function and its Jacobian.
        """
        def force(x):
            r = np.linalg.norm(x - pos)
            if r < self.eps:
                return np.array([0.0, 0.0])
            return -k * (x - pos) / (r ** 3)
        def jacobian(x):
            r = np.linalg.norm(x - pos)
            if r < self.eps:
                return np.array([[0.0, 0.0], [0.0, 0.0]])
            return -k * (np.eye(2) / r ** 3 - 3 * np.outer(x - pos, x - pos) / r ** 5)
        return force, jacobian
    
    def __elastic_force(self, pos_eq, k):
        """
        Returns an elastic force function and its Jacobian.
        """
        def force(x):
            return -k * (x - pos_eq)
        def jacobian(x):
            return -k * np.eye(2)
        return force, jacobian
    
    def calc_total_force(self, x):
        """
        Calculates the total force at a given point.

        Parameters:
        x (np.array): Coordinates [x, y] of the point.

        Returns:
        np.array: The total force at the point.
        """
        return self.c_force(x) + self.g_force1(x) + self.g_force2(x) + self.e_force(x)

    def calc_total_jac(self, x):
        return self.c_jac(x) + self.g_jac1(x) + self.g_jac2(x) + self.e_jac(x)



    def solve(self, initial_guess, tolerance=1e-6, max_iterations=100):
        """
        Solves the system of equations to find the Lagrange points.

        Parameters:
        initial_guess (np.array): Initial guess for the Lagrange point.
        tolerance (float): Tolerance for the solver. Defaults to 1e-6.
        max_iterations (int): Maximum number of iterations. Defaults to 100.

        Returns:
        np.array: The Lagrange point.
        """
        if isinstance(initial_guess, str) and initial_guess == 'L1':
            initial_guess = np.array([0.8369154703225321, 0])
        elif isinstance(initial_guess, str) and initial_guess == 'L2':
            initial_guess = np.array([1.1556818961296604, 0])
        elif isinstance(initial_guess, str) and initial_guess == 'L3':
            initial_guess = np.array([-1.0050626166357435, 0])
        elif isinstance(initial_guess, str) and initial_guess == 'L4':
            initial_guess = np.array([0.5 - self.barycenter[0], np.sqrt(3) / 2])
        elif isinstance(initial_guess, str) and initial_guess == 'L5':
            initial_guess = np.array([0.5 - self.barycenter[0], -np.sqrt(3) / 2
            ])
        elif type(initial_guess) != np.ndarray:
            raise ValueError("Invalid initial guess")
        solver = NewtonRaphsonSolver(
            f=self.total_force, 
            J=self.total_jac, 
            input_dim=2, 
            output_dim=2, 
            N=max_iterations, 
            epsilon=tolerance
        )
        solution = solver.solve(initial_guess)
        return solution
    