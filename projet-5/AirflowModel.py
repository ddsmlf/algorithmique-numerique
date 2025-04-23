from CubicSpline import CubicSpline
from AirfoilDataLoader import AirfoilDataLoader  
from NumericalIntegration import calculate_curve_length
import matplotlib.pyplot as plt
import numpy as np

class AirflowModel:
    """
    Class to model airflow around an airfoil.
    """

    def __init__(self, airfoil_name, vertical_interval=3):
        """
        Initialize the AirflowModel class.

        Parameters:
        airfoil_name (str): Name of the airfoil.
        vertical_interval (int): Vertical interval.
        """
        _, self.upper_surface_x, self.upper_surface_y, self.lower_surface_x, self.lower_surface_y = AirfoilDataLoader.load(f'./data/{airfoil_name}.dat')
        self.upper_surface_spline = CubicSpline(self.upper_surface_x, self.upper_surface_y)
        self.lower_surface_spline = CubicSpline(self.lower_surface_x, self.lower_surface_y)
        self.vertical_interval = vertical_interval
        self.min_height, self.max_height = self.get_airfoil_height()
        self.min_x = min(self.upper_surface_x.min(), self.lower_surface_x.min())
        self.max_x = max(self.upper_surface_x.max(), self.lower_surface_x.max())

    def get_airfoil_height(self):
        """
        Calculate the minimum and maximum heights of the airfoil.

        Returns:
        tuple: Minimum and maximum heights.
        """
        upper_surface_y_values = np.array([self.upper_surface_spline.interpolate(x) for x in self.upper_surface_x])
        lower_surface_y_values = np.array([self.lower_surface_spline.interpolate(x) for x in self.lower_surface_x])
        min_height = np.min(lower_surface_y_values)
        max_height = np.max(upper_surface_y_values)
        return min_height, max_height

    def _curve_function(self, lambda_value, spline, height):
        """
        Calculate a curve function.

        Parameters:
        lambda_value (float): Lambda value.
        spline (CubicSpline): Spline object.
        height (float): Height value.

        Returns:
        function: Curve function.
        """
        def curve(x):
            return (1 - lambda_value) * spline.interpolate(x) + lambda_value * self.vertical_interval * height
        return curve

    def get_disturbance_curves(self, num_curves_above=20, num_curves_below=5):
        """
        Calculate disturbance curves.

        Returns:
        tuple: Upper and lower disturbance curves.
        """
        lambda_values_above = np.linspace(0, 1, num_curves_above)
        lambda_values_below = np.linspace(0, 1, num_curves_below)
        upper_curves = []
        lower_curves = []
        for i in range(num_curves_above):
            upper_curves.append(self._curve_function(lambda_values_above[i], self.upper_surface_spline, self.max_height))
        for i in range(num_curves_below):
            lower_curves.append(self._curve_function(lambda_values_below[i], self.lower_surface_spline, self.min_height))
        return upper_curves, lower_curves

    def plot_airfoil(self, num_curves=[20, 20]):
        """
        Plot the airfoil.
        """
        plt.figure(figsize=(10, 6))

        upper_curves, lower_curves = self.get_disturbance_curves(num_curves[0], num_curves[1])
        x_range = np.linspace(self.min_x, self.max_x, 500)
        for i, curve in enumerate(upper_curves):
            y_values = [curve(x) for x in x_range]
            plt.plot(x_range, y_values, 'gray')  
        for i, curve in enumerate(lower_curves):
            y_values = [curve(x) for x in x_range]
            plt.plot(x_range, y_values, 'gray', label='Courbes de perturbation' if i == 0 else "")  
        plt.plot(self.upper_surface_x, self.upper_surface_y, 'yo', label='Points de la surface supérieure')
        plt.plot(self.lower_surface_x, self.lower_surface_y, 'ro', label='Points de la surface inférieure')
        plt.plot(self.upper_surface_x, [self.upper_surface_spline.interpolate(x) for x in self.upper_surface_x], 'b-', label='Spline de la surface supérieure')
        plt.plot(self.lower_surface_x, [self.lower_surface_spline.interpolate(x) for x in self.lower_surface_x], 'g-', label='Spline de la surface inférieure')

        plt.title("Profil aérodynamique")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        plt.show()

    def pressure_map(self, method="simpson", num_curves=[20, 20]):
        """
        Generate and display a pressure map.
        """
        num_curves_above = num_curves[0]
        num_curves_below = num_curves[1]
        density = 1.0 
        a, b = self.min_x, self.max_x
        x_range = np.linspace(a, b, 500)

        upper_curves, lower_curves = self.get_disturbance_curves(num_curves_above, num_curves_below)

        upper_lengths = [calculate_curve_length(curve, 1, method) for curve in upper_curves]
        lower_lengths = [calculate_curve_length(curve, 1, method) for curve in lower_curves]

        upper_pressures = [0.5 * density * length**2 for length in upper_lengths]
        lower_pressures = [0.5 * density * length**2 for length in lower_lengths]

        height = num_curves_above + num_curves_below
        width = len(x_range)
        pressure_map = np.full((height, width),  0.5 * density)  
        y_map = np.zeros((height, width))

        for i, curve in enumerate(reversed(upper_curves)):
            y_values = np.array([curve(x) for x in x_range])
            pressure_map[i, :] = upper_pressures[len(upper_curves) - 1 - i]
            y_map[i, :] = y_values

        for i, curve in enumerate(lower_curves):
            y_values = np.array([curve(x) for x in x_range])
            pressure_map[num_curves_above + i, :] = lower_pressures[i]
            y_map[num_curves_above + i, :] = y_values

        upper_surface_y_values = np.array([self.upper_surface_spline.interpolate(x) for x in x_range])
        lower_surface_y_values = np.array([self.lower_surface_spline.interpolate(x) for x in x_range])

        mask_inside = (y_map <= np.tile(upper_surface_y_values, (y_map.shape[0], 1))) & \
                    (y_map >= np.tile(lower_surface_y_values, (y_map.shape[0], 1)))

        pressure_map[mask_inside] =  0.5 * density

        plt.figure(figsize=(10, 6))
        X = np.tile(x_range, (y_map.shape[0], 1)) 

        ax = plt.gca()
        ax.set_facecolor('black') 

        cmap = plt.get_cmap('hot')
        cmap.set_bad(color='black')

        plt.contourf(X, y_map, pressure_map, levels=100, cmap=cmap, extend='both')
        plt.colorbar(label='Pression')
        plt.plot(x_range, upper_surface_y_values, 'b-', label='Surface supérieure')
        plt.plot(x_range, lower_surface_y_values, 'g-', label='Surface inférieure')

        plt.ylim(3 * self.min_height, 3 * self.max_height)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Carte de Pression (Simplifiée Bernoulli)')
        plt.axis('equal')
        plt.legend()
        plt.grid(False) 
        plt.show()

if __name__ == "__main__": 
    airfoil_name = 'fx72150a'
    
    airfoil = AirflowModel(airfoil_name)
    airfoil.plot_airfoil()
    airfoil.pressure_map()