import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage import data
from CauchySolver import CauchySolver

class ImageFilter:
    def __init__(self, image, dt_heat=0.1, dt_pm=0.2, max_iter=200, sigma=1.0):
        """
        Initialize the ImageFilter with the image and parameters.

        :param image: The input image (2D numpy array).
        :param dt_heat: Time step for the heat equation.
        :param dt_pm: Time step for the Perona-Malik equation.
        :param max_iter: Maximum number of iterations.
        :param k: Parameter for the Perona-Malik function.
        :param sigma: Standard deviation for the Gaussian kernel.
        """
        self.image = image.astype(np.float64)
        self.dt_heat = dt_heat
        self.dt_pm = dt_pm
        self.max_iter = max_iter
        self.sigma = sigma
        self.shape = image.shape

    def _gradient(self, u):
        """
        Compute the gradient of the image.

        :param u: Input image (2D array).
        :return: Gradient of the image (2D arrays).
        """
        grad_x = np.zeros_like(u)
        grad_y = np.zeros_like(u)
        grad_x[:-1, :] = u[1:, :] - u[:-1, :]
        grad_y[:, :-1] = u[:, 1:] - u[:, :-1]
        return grad_x, grad_y

    def _divergence(self, p):
        """
        Compute the divergence of the vector field p, taking into account boundary conditions
        as per the discrete definition.

        :param p: Tuple (p_x, p_y) of 2D arrays representing the vector field components.
        :return: 2D array of divergence values.
        """
        p_x, p_y = p
        N, M = p_x.shape  

        div = np.zeros_like(p_x)

        div[1:-1, :] += p_x[1:-1, :] - p_x[0:-2, :]  # 1 < k < N
        div[0, :] += p_x[0, :]                      # k = 1
        div[-1, :] += -p_x[-2, :]                   # k = N

        div[:, 1:-1] += p_y[:, 1:-1] - p_y[:, 0:-2]  # 1 < l < M
        div[:, 0] += p_y[:, 0]                      # l = 1
        div[:, -1] += -p_y[:, -2]                   # l = M

        return div


    def _heat_equation(self, u):
        """
        Compute the heat equation.
        :param u: Input image (2D array).
        :return: Laplacian of the image (2D array).
        """
        grad_x, grad_y = self._gradient(u)
        laplacian = self._divergence((grad_x, grad_y))
        return laplacian

    def _perona_malik_function(self, grad_norm):
        """
        Compute the Perona-Malik function.
        :param grad_norm: Norm of the gradient.
        :return: Perona-Malik function value.
        """
        return np.exp(-(grad_norm**2))

    def _perona_malik_equation(self, u, f):
        """
        Compute the Perona-Malik equation.
        :param u: Input image (2D array).
        :param f: Perona-Malik function (2D array).
        :return: Divergence of the Perona-Malik function (2D array).
        """
        grad_x, grad_y = self._gradient(u)
        div_f_grad = self._divergence((f * grad_x, f * grad_y))
        return div_f_grad

    def gaussian_convolution(self, image):
        """
        Convolve the image with a Gaussian kernel.
        :param image: Input image (2D array).
        :return: Convolved image (2D array).
        """
        kernel_size = int(4 * self.sigma + 1)
        x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
        kernel = np.exp(-((x**2 + y**2) / (2.0 * self.sigma**2)))
        kernel /= kernel.sum()
        convolved_image = convolve2d(image, kernel, mode='same', boundary='symm')
        return convolved_image

    def solve_heat_equation(self, method='euler'):
        """
        Solve the heat equation using the specified numerical method.
        :param method: Numerical method to use ('euler', 'runge_kutta_4', 'midpoint', 'heun').
        :return: Filtered image (2D array).
        """
        def heat_ode(t, u):
            u_2d = u.reshape(self.shape)
            laplacian = self._heat_equation(u_2d)
            return laplacian.ravel()

        solver = CauchySolver(heat_ode)
        t0 = 0
        y0 = self.image.copy().ravel()
        N = self.max_iter
        h = self.dt_heat

        t_points, y_points = solver.meth_n_step(y0, t0, N, h, method)
        u = y_points[-1].reshape(self.shape)
        return u

    def solve_perona_malik(self, method='euler'):
        """
        Solve the Perona-Malik equation using the specified numerical method.
        :param method: Numerical method to use ('euler', 'runge_kutta_4', 'midpoint', 'heun').
        :return: Filtered image (2D array).
        """
        convolved_image = self.gaussian_convolution(self.image)
        grad_x, grad_y = self._gradient(convolved_image)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        f = self._perona_malik_function(grad_norm)

        def perona_malik_ode(t, u):
            u_2d = u.reshape(self.shape)
            div_f_grad = self._perona_malik_equation(u_2d, f)
            return div_f_grad.ravel()

        solver = CauchySolver(perona_malik_ode)
        t0 = 0
        y0 = self.image.copy().ravel()
        N = self.max_iter
        h = self.dt_pm

        t_points, y_points = solver.meth_n_step(y0, t0, N, h, method)
        u = y_points[-1].reshape(self.shape)
        return u

    def visualize(self, original, filtered, title):
        """
        Visualize the original and filtered images.
        :param original: Original image (2D array).
        :param filtered: Filtered image (2D array).
        :param title: Title for the filtered image.
        """
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(filtered, cmap='gray')
        ax[1].set_title(title)
        ax[1].axis('off')
        plt.show()

    def visualize_perona_malik_function(self):
        """
        Visualize the Perona-Malik function f with and without Gaussian convolution.
        """
        grad_x, grad_y = self._gradient(self.image)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        f_no_gaussian = self._perona_malik_function(grad_norm)

        convolved_image = self.gaussian_convolution(self.image)
        grad_x_conv, grad_y_conv = self._gradient(convolved_image)
        grad_norm_conv = np.sqrt(grad_x_conv**2 + grad_y_conv**2)
        f_with_gaussian = self._perona_malik_function(grad_norm_conv)

        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(f_no_gaussian, cmap='gray')
        ax[0].set_title('Fonction Perona-Malik (Sans Gaussienne)')
        ax[0].axis('off')

        ax[1].imshow(f_with_gaussian, cmap='gray')
        ax[1].set_title('Fonction Perona-Malik (Avec Gaussienne)')
        ax[1].axis('off')

        plt.savefig('img/perona_malik_function_comparison.png')
        plt.show()


if __name__ == "__main__":
    image = data.camera()

    filter = ImageFilter(image, sigma=1.0)

    heat_filtered = filter.solve_heat_equation(method='euler')
    filter.visualize(image, heat_filtered, 'Heat Equation Filtered')

    gaussian_filtered = filter.gaussian_convolution(image)
    filter.visualize(image, gaussian_filtered, 'Gaussian Convolution Filtered')

    filter.visualize_perona_malik_function()

    pm_filtered = filter.solve_perona_malik(method='euler')
    filter.visualize(image, pm_filtered, 'Perona-Malik Filtered')