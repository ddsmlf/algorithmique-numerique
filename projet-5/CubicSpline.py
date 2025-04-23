import numpy as np

class CubicSpline:
    """
    A class representing a cubic spline interpolation.

    Attributes:
        x (numpy array): The x-values of the data points.
        y (numpy array): The y-values of the data points.
        yp1 (float): The derivative at the first point (default is estimated).
        ypn (float): The derivative at the last point (default is estimated).
        max (float): The maximum value for the derivative (used for natural spline conditions).
    """

    def __init__(self, x, y, yp1=None, ypn=None, natural_spline=False):
        """
        Initialize the CubicSpline object.

        Parameters:
            x (list or numpy array): The x-values of the data points.
            y (list or numpy array): The y-values of the data points.
            yp1 (float, optional): The derivative at the first point (default is estimated).
            ypn (float, optional): The derivative at the last point (default is estimated).
        """
        self.x = np.array(x)
        self.y = np.array(y)
        if yp1 is None:
            self.yp1 = self._estimate_derivative(x, y, 0)
        else:
            self.yp1 = yp1
        if ypn is None:
            self.ypn = self._estimate_derivative(x, y, -1)
        else:
            self.ypn = ypn
        self.max = np.finfo(float).max
        if len(self.x)!= len(self.y):
            raise ValueError("The x and y arrays must have the same length")
        self._spline()
        if natural_spline:
            self.y2[0] = 0.0
            self.y2[-1] = 0.0

    def _estimate_derivative(self, x, y, index):
        """
        Estimate the derivative at a given point.

        Parameters:
            x (list or numpy array): The x-values of the data points.
            y (list or numpy array): The y-values of the data points.
            index (int): The index of the point (0 for the first point, -1 for the last point).

        Returns:
            float: The estimated derivative at the given point.
        """
        if index == 0:
            return (y[1] - y[0]) / (x[1] - x[0])
        elif index == -1:
            return (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            raise ValueError("Index must be 0 or -1")

    def _spline(self):
        """
        Compute the cubic spline interpolation coefficients.

        This method computes the coefficients of the cubic spline interpolation
        based on the x and y values, and the derivatives at the first and last points.
        """
        n = len(self.x)
        self.y2 = np.zeros(n)
        u = np.zeros(n-1)

        # Check if the first derivative is large enough to use natural spline conditions
        if self.yp1 > self.max:
            self.y2[0] = u[0] = 0.0
        else:
            dx0 = self.x[1] - self.x[0]
            dy0 = self.y[1] - self.y[0]
            self.y2[0] = -0.5
            u[0] = (3.0 / dx0) * (dy0 / dx0 - self.yp1)

        for i in range(1, n-1):
            dx_prev = self.x[i] - self.x[i-1]
            dx_next = self.x[i+1] - self.x[i]
            sig = dx_prev / (dx_prev + dx_next)
            p = sig * self.y2[i-1] + 2.0
            self.y2[i] = (sig - 1.0) / p
            u[i] = (self.y[i+1] - self.y[i]) / dx_next - (self.y[i] - self.y[i-1]) / dx_prev
            u[i] = (6.0 * u[i] / (dx_prev + dx_next) - sig * u[i-1]) / p

        # Check if the last derivative is large enough to use natural spline conditions
        if self.ypn > self.max:
            qn = un = 0.0
        else:
            dx_last = self.x[n-1] - self.x[n-2]
            dy_last = self.y[n-1] - self.y[n-2]
            qn = 0.5
            un = (3.0 / (dx_last)) * (self.ypn - (dy_last) / (dx_last))

        self.y2[n-1] = (un - qn * u[n-2]) / (qn * self.y2[n-2] + 1.0)

        for k in range(n-2, -1, -1):
            self.y2[k] = self.y2[k] * self.y2[k+1] + u[k]

    def interpolate(self, x_val):
        """
        Evaluate the cubic spline interpolation at a given x-value.

        Parameters:
            x_val (float): The x-value at which to evaluate the interpolation.

        Returns:
            float: The interpolated y-value at the given x-value.
        """
        n = len(self.x)

        klo = 0
        khi = n - 1
        while khi - klo > 1:
            k = (khi + klo) // 2
            if self.x[k] > x_val:
                khi = k
            else:
                klo = k

        h = self.x[khi] - self.x[klo]
        if h == 0.0:
            raise ValueError("The points x[klo] and x[khi] are coincident")
        if x_val < self.x[0] or x_val > self.x[-1]:
            raise ValueError("The x-value is out of bounds")

        a = (self.x[khi] - x_val) / h
        b = (x_val - self.x[klo]) / h
        y = (a * self.y[klo] + b * self.y[khi] +
              ((a**3 - a) * self.y2[klo] + (b**3 - b) * self.y2[khi]) * (h**2) / 6.0)
        return y

    def get_bounds(self):
        """
        Get the bounds of the x-values.

        Returns:
            tuple: The lower and upper bounds of the x-values.
        """
        return self.x[0], self.x[-1]

    def derivative(self, x_val):
        """
        Evaluate the derivative of the cubic spline interpolation at a given x-value.

        Parameters:
            x_val (float): The x-value at which to evaluate the derivative.

        Returns:
            float: The derivative of the interpolation at the given x-value.
        """
        n = len(self.x)
        klo = 0
        khi = n - 1
        while khi - klo > 1:
            k = (khi + klo) // 2
            if self.x[k] > x_val:
                khi = k
            else:
                klo = k

        h = self.x[khi] - self.x[klo]
        a = (self.x[khi] - x_val) / h
        b = (x_val - self.x[klo]) / h

        dy = (self.y[khi] - self.y[klo]) / h - (3 * a**2 - 1) * h * self.y2[klo] / 6 + (3 * b**2 - 1) * h * self.y2[khi] / 6
        return dy