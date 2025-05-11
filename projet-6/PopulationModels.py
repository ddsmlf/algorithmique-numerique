import numpy as np
from CauchySolver import CauchySolver

class PopulationModels:
    def __init__(self, gamma, kappa=None, b=None, c=None, d=None, eps=1e-2):
        """
        Initialize PopulationModels with parameters for Malthus and Verhulst and Lotka-Volterra models.

        :param gamma: prey growth rate (equivalent to a for Lotka-Volterra)
        :param kappa: carrying capacity (for Verhulst model)
        :param b: prey death rate (for Lotka-Volterra)
        :param c: predators reproduction rate (for Lotka-Volterra)
        :param d: predators death rate (for Lotka-Volterra)
        :param eps: tolerance for numerical methods
        """
        self.gamma = gamma
        self.kappa = kappa
        self.b = b
        self.c = c
        self.d = d
        self.b = b
        self.c = c
        self.d = d
        self.eps = eps
        if kappa is not None and (b is not None or c is not None or d is not None):
            raise ValueError("Either kappa or Lotka-Volterra parameters (b, c, d) should be provided, not both.")
        if (b is None or c is None or d is None) and (b is not None or c is not None or d is not None):
            raise ValueError("All Lotka-Volterra parameters (b, c, d) must be provided together.")
        if kappa is None and b is None and c is None and d is None:
            print("Malthus model will be used as no parameters for Verhulst or Lotka-Volterra are provided.")
            self.model = lambda t, N: self.malthus(t, N)
        if kappa is not None :
            print("Verhulst model will be used as kappa is provided.")
            self.model = lambda t, N: self.verhulst(t, N)
        if b is not None and c is not None and d is not None:
            print("Lotka-Volterra model will be used as b, c, d are provided.")
            self.model = self.lotka_volterra
            self.a = gamma
        self.solver = CauchySolver(self.model, eps=self.eps)

    def malthus(self, t, N):
        """
        Malthusian growth model.
        :param t: time
        :param N: population
        :return: rate of change of population
        """
        return self.gamma * N

    def verhulst(self, t, N):
        """
        Verhulst (logistic) growth model.
        :param t: time
        :param N: population
        :return: rate of change of population
        """
        if self.kappa is None:
            raise ValueError("kappa (carrying capacity) must be provided for the Verhulst model.")
        return self.gamma * N * (1 - N / self.kappa)
    
    def lotka_volterra(self, t, y):
        """
        Lotka-Volterra equations.
        :param t: time
        :param y: population vector [N, P]
        :return: derivatives [dN/dt, dP/dt]
        """
        N, P = y
        dN_dt = N * (self.a - self.b * P)
        dP_dt = P * (self.c * N - self.d)

        return [dN_dt, dP_dt]
    
    def solve(self, N0, t0, t_end=10, method='runge_kutta_4', plot=False, plot_N_times=None):
        """
        Solve the specified population model.
        :param N0: initial population vector [N0] for Malthus and Verhulst or [N0, P0] for Lotka-Volterra
        :param t0: initial time
        :param method: numerical method to use ('euler', 'runge_kutta_4', 'midpoint', 'heun')
        :param plot: whether to plot the results
        :param plot_N_times: number of times to plot the results
        :return: time points and population values

        """
        return self.solver.meth_epsilon(N0, t0, t_end=t_end, method=method, plot=plot, plot_N_times=plot_N_times)

    def find_period(self, t, y, threshold=1e-3, max_steps=10000):
        """
        Find the period of the Lotka-Volterra model by checking when the population returns to its initial state.
        :param t: time points
        :param y: result of the ODE solver (population values)
        :param threshold: threshold for determining when the populations are close enough to be considered the same
        :param max_steps: maximum number of steps to take
        :return: period, time points, population values
        """

        N, P = zip(*y)
        y0 = np.array(y[0])
        periods = [] 

        for i in range(1, len(y)):
            distance = np.linalg.norm(np.array(y[i]) - y0)
            if distance < threshold:
                periods.append(t[i]) 

        if periods:
           
            if len(periods) > 1:
                period = periods[1] - periods[0]  
            else:
                period = t[0] - periods[0]
            return period, periods, t, N, P
        else:
            print("No period found within the given threshold.")
            return None, [], t, N, P

    def find_singular_points(self):
        """
        Find the singular points of the Lotka-Volterra model.
        :return: list of singular points
        """
        if self.model != self.lotka_volterra:
            raise ValueError("This method is only applicable to the Lotka-Volterra model.")

        singular_points = [
            (0, 0),  
            (self.d / self.c, self.a / self.b)  
        ]

        return singular_points
