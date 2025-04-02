import numpy as np
from NewtonRaphsonSolver import NewtonRaphsonSolver


class Bairstow:
    """
    Class to solve polynomial equations using Bairstow's method.
    
    Attributes:
        coefficients (list): Coefficients of the polynomial.
        n (int): Degree of the polynomial.
    """
    def __init__(self, coefficients):
        """
        Initialization of the Bairstow class.
        Parameters:
            coefficients (list): Coefficients of the polynomial.
        """
        self.coefficients = coefficients
        self.n = len(coefficients) - 1
    
    def __str__(self):
        """
        String representation of the Bairstow class.
        Returns:
            str: String representation of the polynomial.
        """
        string = "P(X) = "
        for i, coeff in enumerate(self.coefficients):
            exp = self.n - i
            if i == 0:
                if coeff == -1:
                    string += "-"
                elif coeff != 1:
                    string += f"{coeff}"
            else:
                if coeff == -1:
                    string += " - "
                elif coeff == 1:
                    string += " + "
                elif coeff < 0:
                    string += f" - {abs(coeff)}"
                elif coeff > 0:
                    string += f" + {coeff}"
            if coeff != 0:
                if exp == 0:
                    string += ""
                elif exp == 1:
                    string += "X"
                else:
                    string += f"X^{exp}"
        return string

    def compute_factorization(self, B, C):
        """
        Compute the factorization of the polynomial using Bairstow's method.
        
        Parameters:
            B (float): Coefficient B.
            C (float): Coefficient C.
        Returns:
            tuple: Quotient and remainder of the polynomial division.
        """
        divisor = np.array([1, B, C])
        
        quotient, remainder = np.polydiv(self.coefficients, divisor)

        return quotient, remainder, divisor
    
    def __F(self, U):
        """
        Compute the function F(U) = (R, S) where R and S are the remainder of the polynomial division.

        Parameters:
            U (np.array): Array containing the coefficients B and C.
        Returns:
            np.array: Residuals.
        """
        residuals = self.compute_factorization(U[0], U[1])[1]
        if residuals.shape[0] == 1:
            return np.array([0, residuals[0]])
        return residuals
    
    
    
    def __F_jacobian(self, U):
        """
        Compute the Jacobian matrix of the function F(U) = (R, S).
        Parameters:
            U (np.array): Initial guess for the coefficients B and C.
        Returns:
            np.array: Solution for the coefficients B and C.
        """
        B, C = U
        h = 1e-6 

        residuals = self.compute_factorization(B, C)[1]
        if residuals.shape[0] == 1:
            R, S = 0, residuals[0]
        else:
            R, S = residuals[0], residuals[1]

        residualsdB = self.compute_factorization(B + h, C)[1]
        if residualsdB.shape[0] == 1:
            dRdB, dSdB = 0, residualsdB[0] - S
        else:
            dRdB, dSdB = residualsdB[0] - R, residualsdB[1] - S
        dRdB /= h
        dSdB /= h

        residualsdC = self.compute_factorization(B, C + h)[1]
        if residualsdC.shape[0] == 1:
            dRdC, dSdC = 0, residualsdC[0] - S
        else:
            dRdC, dSdC = residualsdC[0] - R, residualsdC[1] - S
        dRdC /= h
        dSdC /= h

        return np.array([[dRdB, dRdC], [dSdB, dSdC]])
        

    
    def BC_solver(self, U0):
        """
        Solve the system of equations using the Newton-Raphson method.
        
        Parameters:
            U0 (np.array): Initial guess for the coefficients B and C.
        Returns:
            np.array: Solution for the coefficients B and C.
        """
        f = self.__F
        J = self.__F_jacobian
        solver = NewtonRaphsonSolver(f, J, input_dim=2, output_dim=2)
        return solver.solve(U0)
    
    def __determineU0(self):
        """
        Determine the initial guess for the coefficients B and C.
        
        Returns:
            np.array: Initial guess for the coefficients B and C.
        """
        a_n = self.coefficients[0]  
        a_n_1 = self.coefficients[1]  
        a_0 = self.coefficients[-1] 

        B = - a_n_1 /  a_n
        C = a_0 /  a_n
        return np.array([B, C])
    
    def solve(self, U0=None):
        """
        Solve the polynomial equation using Bairstow's method.
        
        Parameters:
            U0 (np.array): Initial guess for the coefficients B and C.
        
        Returns:
            tuple: Roots of the polynomial.
        """
        if U0 is None:
            U0 = self.__determineU0()

        B, C = self.BC_solver(U0)
        quotient, _, divisor = self.compute_factorization(B, C)
        roots_quotient = np.roots(quotient)
        
        roots_divisor = np.roots(divisor)
        
        roots = np.concatenate((roots_quotient, roots_divisor))
        
        return roots
        