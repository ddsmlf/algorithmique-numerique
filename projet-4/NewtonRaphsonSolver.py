import numpy as np

class NewtonRaphsonSolver:
    def __init__(self, f, J, input_dim=None, output_dim=None, N=100, epsilon=1e-12):
        """
        Initialization of the NewtonSolver class

        Parameters:
        f (function): Function to solve.
        J (function): Jacobian matrix of the function f.
        input_dim (int): Input dimension of the function f.
        output_dim (int): Output dimension of the function f.
        N (int): Maximum number of allowed iterations.
        epsilon (float): Convergence threshold of the algorithm.
        """
        if not callable(f) or not callable(J):
            raise ValueError("f and J must be functions")
        self.f = f
        self.J = J
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N
        self.epsilon = epsilon
        self.__cache = {}

    def __get_cache(self, U):
        """
        Retrieve cached values of f(U) and J(U).

        Parameters:
        U (numpy.array): Current point.

        Returns:
        tuple: (f(U), J(U)) if U is in the cache, otherwise None.
        """
        U_tuple = tuple(map(tuple, U)) if U.ndim > 1 else tuple(U)

        if U_tuple in self.__cache:
            return self.__cache[U_tuple]
        else:
            return None

    def __set_cache(self, U, f_U, J_U):
        """
        Store values of f(U) and J(U) in the cache.

        Parameters:
        U (numpy.array): Current point.
        f_U (numpy.array): Value of f(U).
        J_U (numpy.array): Value of J(U).
        """
        U_tuple = tuple(map(tuple, U)) if U.ndim > 1 else tuple(U)
        self.__cache[U_tuple] = (f_U, J_U)

    def compute_f_J(self, U):
        """
        Compute f(U) and J(U) with caching.

        Parameters:
        U (numpy.array): Current point.

        Returns:
        tuple: (f(U), J(U)).
        """
        cached = self.__get_cache(U)
        if cached is not None:
            return cached
        else:
            f_U = self.f(U)
            J_U = self.J(U)
            self.__set_cache(U, f_U, J_U)
            return f_U, J_U

    def __solve_linear_system(self, J_U, f_U, numpy=True):
        """
        Uses QR decomposition to solve the linear system H(U)V = -F(U).
        """
        if numpy:
            J_U = np.atleast_2d(J_U)
            f_U = np.atleast_2d(f_U).T
        Q, R = np.linalg.qr(J_U)
        V = np.linalg.lstsq(R, -(Q.T @ f_U), rcond=None)[0].flatten()  # Handles singular matrices
        return V

    def __backtracking(self, U, f_U, J_U, V):
        """
        Backtracking method to avoid divergence of the algorithm.

        Parameters:
        U (numpy.array): Current point.
        f_U (numpy.array): Value of f(U).
        J_U (numpy.array): Value of J(U).
        V (numpy.array): Descent direction.

        Returns:
        numpy.array: New point after backtracking.
        """
        alpha = 1.0
        while True:
            U_new = U + alpha * V
            f_U_new, _ = self.compute_f_J(U_new)
            if np.linalg.norm(f_U_new, ord=1) < np.linalg.norm(f_U, ord=1):
                return U_new
            alpha *= 0.5
            if alpha < self.epsilon:
                raise ValueError("Backtracking failed")

    def solve(self, U0, backtracking=True, graph=False):
        """
        Newton-Raphson method for solving nonlinear optimization problems.

        Parameters:
        U0 (numpy.array): Starting point of the algorithm.

        Returns:
        """
        if graph:
            convergence = []
        if self.input_dim is not None and U0.shape[0] != self.input_dim:
            raise ValueError("The input dimension of U0 is incorrect")
        U = U0
        for _ in range(self.N):
            f_U, J_U = self.compute_f_J(U)
            if self.output_dim is not None and f_U.shape[0] != self.output_dim:
                raise ValueError(f"The output dimension of f(U) is incorrect found {f_U.shape[0]} instead of {self.output_dim}")
            norm = np.linalg.norm(f_U, ord=1)
            if graph:
                convergence.append(norm)
            if norm < self.epsilon:
                if graph:
                    return U, convergence
                return U
            V = self.__solve_linear_system(J_U, f_U)
            if backtracking:
                U = self.__backtracking(U, f_U, J_U, V)
            else:
                U = U + V
        print("Newton's method did not converge")
        if graph:
            return U, convergence
        return U
