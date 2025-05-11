import numpy as np
import gc

class CauchySolver:
    """
    A class to solve Cauchy problems for ordinary differential equations (ODEs).
    """
    def __init__(self, f, eps=1e-2):
        """
        Initialize the CauchySolver with the ODE, initial conditions, and time span.

        :param f: The function representing the ODE (dy/dt = f(t, y)).
        :param eps: The tolerance for the numerical methods.
        """
        self.f = f
        self.eps = eps

    def step_euler(self, t, y, h):
        """
        Euler method for solving ODEs.
        """
        return np.array(y) + h * np.array(self.f(t, y))

    def step_runge_kutta_4(self, t, y, h):
        """
        Runge-Kutta method for solving ODEs.
        """
        y = np.array(y)
        k1 = self.f(t, y)
        k2 = self.f(t + 0.5 * h, y + 0.5 * h * np.array(k1))
        k3 = self.f(t + 0.5 * h, y + 0.5 * h * np.array(k2))
        k4 = self.f(t + h, y + h * np.array(k3))
        return np.array(y) + (h / 6.0) * (np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4))

    def step_midpoint(self, t, y, h):
        """
        Midpoint method for solving ODEs.
        """
        k1 = self.f(t, y)
        k2 = self.f(t + 0.5 * h, y + 0.5 * h * np.array(k1))
        return np.array(y) + h * np.array(k2)

    def step_heun(self, t, y, h):
        """
        Heun's method for solving ODEs.
        """
        k1 = self.f(t, y)
        k2 = self.f(t + h, y + h * k1)
        return np.array(y) + 0.5 * h * (np.array(k1) + np.array(k2))

    def meth_n_step(self, y0, t0, N, h, method='euler'):
        """
        Solve the ODE using the specified method.

        :param y0: Initial condition.
        :param t0: Initial time.
        :param N: Number of steps.
        :param h: Step size.
        :param method: Numerical method to use ('euler', 'runge_kutta_4', 'midpoint', 'heun').
        :return: Time points and corresponding solution values.
        """
        if method not in ['euler', 'runge_kutta_4', 'midpoint', 'heun']:
            raise ValueError("Invalid method. Choose from 'euler', 'runge_kutta_4', 'midpoint', 'heun'.")
        if N <= 0:
            raise ValueError("Number of steps must be positive.")
        if h <= 0:
            raise ValueError("Step size must be positive.")
        
        yn = np.zeros((N + 1,) + np.shape(y0))
        tn = np.zeros(N + 1)
        yn[0] = np.array(y0, dtype=yn.dtype)
        tn[0] = t0
        step_function = getattr(self, f'step_{method}')
        
        for i in range(1, N + 1):
            tn[i] = tn[i - 1] + h
            yn[i] = step_function(tn[i - 1], yn[i - 1], h)
            if np.any(np.isnan(yn[i])) or np.any(np.isinf(yn[i])):
                raise ValueError(f"Numerical instability detected: 'nan' or 'inf' values encountered. yn[i] = {yn[i]}, tn[i-1] = {tn[i-1]}, yn[i-1] = {yn[i-1]}, h = {h}")
        
        return tn, yn

    def meth_epsilon(self, y0, t0, t_end=10, method='euler', plot=False, plot_N_times=None):
        """
        Solve the ODE using the specified method with a given tolerance and total time.

        :param y0: Initial condition.
        :param t0: Initial time.
        :param t_end: Total time for the simulation.
        :param method: Numerical method to use ('euler', 'runge_kutta_4', 'midpoint', 'heun').
        :param plot: Whether to plot the results.
        :param plot_N_times: Number of times to plot the results.
        :return: Time points and corresponding solution values.
        """
        N = 100
        h = (t_end - t0) / N
        tn, y2n = self.meth_n_step(y0, t0, N, h, method)
        e = 100
        if plot:
            Ns = []
            errors = []
        while (e > self.eps and plot_N_times is None) or (plot_N_times is not None and (plot and plot_N_times > 0)):
            yn = y2n.copy()
            h /= 2
            N *= 2
            tn, y2n = self.meth_n_step(y0, t0, N, h, method)
            y2n_n = y2n[0::2]
            diff = yn - y2n_n
            if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
                raise ValueError("Numerical instability detected: 'nan' or 'inf' values encountered.")
            e = np.linalg.norm(diff)
            if plot:
                errors.append(e)
                Ns.append(N)
                if plot_N_times is not None:
                    plot_N_times -= 1
                    if plot_N_times == 0:
                        break
            print(f"Step size: {h}, Error: {e}, self.eps: {self.eps}")
            del yn
            del y2n_n
            gc.collect()
        if plot:
            return tn, y2n, errors, Ns
        return tn, y2n
