import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
from CauchySolver import CauchySolver
from PopulationModels import PopulationModels
from ImageFilter import ImageFilter

def f_1(t, y):
    return np.array(y) / (1 + np.array(t)**2)


def f_2(_, y):
    return np.array([-y[1], y[0]])
def solve_1(t):
    return np.exp(np.arctan(t))

def solve_2(t):
    return np.array([np.cos(t), np.sin(t)])

test_functions = [
    (f_1, -2, 1, solve_1),
    (f_2, 0, np.array([1, 0]), solve_2)
]

methods = ['euler', 'runge_kutta_4', 'midpoint', 'heun']



def plot_cauchy_on_tan_field(f, t0, expected, method='euler', eps=1e-4):
    """
    Tracer la solution du problème de Cauchy sur le champ tangent.
    :param f:
    :param t0: Temps initial.
    :param y0: Condition initiale.
    :param expected: Solution attendue pour comparaison (fonction pour créer la solution attendue).
    :param method: Méthode à utiliser ('euler', 'runge_kutta_4', 'midpoint', 'heun').
    :param eps: Tolérance pour la solution.
    """
    y1_lim = (-2, 2) 
    y2_lim = (-2, 2) 
    n = 20
    t_vals = np.linspace(y1_lim[0], y1_lim[1], n)
    y_vals = np.linspace(y2_lim[0], y2_lim[1], n)
    T, Y = np.meshgrid(t_vals, y_vals)
    if f.__name__ == 'f_2':
        DT = np.zeros_like(T)
        DY = np.zeros_like(Y)

        for i in range(n):
            for j in range(n):
                y_point = np.array([T[i, j], Y[i, j]])
                derivee = f(0, y_point)
                DT[i, j] = derivee[0]
                DY[i, j] = derivee[1]

        norm = np.sqrt(DT**2 + DY**2)
        DT_norm = np.where(norm != 0, DT / norm, 0)
        DY_norm = np.where(norm != 0, DY / norm, 0)

    else:
        DY = f(T, Y)
        DT = np.ones_like(DY)

        norm = np.sqrt(DT**2 + DY**2)
        DY_norm = DY / norm
        DT_norm = DT / norm
        
    plt.quiver(T, Y, DT_norm, DY_norm, color='gray', alpha=0.8)
    solver = CauchySolver(f, eps=eps)
    if f.__name__ == 'f_1':
        initial_ys = np.linspace(-0.6, 0.6, 9)
    else:
        radius_min = 0.5
        radius_max = 1.8
        num_radii = 7   
        num_angles = 8   

        initial_ys = [
            np.array([r * np.cos(a), r * np.sin(a)])
            for r in np.linspace(radius_min, radius_max, num_radii)
            for a in np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        ]



    for y0_i in initial_ys:
        t, y = solver.meth_epsilon(y0_i, t0, method=method)
        if f.__name__ == 'f_2': 
            plt.plot([yi[0] for yi in y], [yi[1] for yi in y], zorder=1)
        else:
            plt.plot(t, y, zorder=1)

    plt.xlim(y1_lim)
    plt.ylim(y2_lim)
    plt.xlabel('t' if f.__name__ == 'f_1' else 'y1')
    plt.ylabel('y' if f.__name__ == 'f_1' else 'y2')
    plt.savefig(f'img/cauchy_on_tan_field_{f.__name__}_{method}.png')
    plt.show()
    plt.close()

def plot_precision(test_functions, methods):
    """
    Tracer la précision des différentes méthodes pour résoudre les problèmes de Cauchy.
    :param test_functions: Liste de tuples contenant la fonction, le temps initial, la condition initiale et la solution attendue.
    :param methods: Liste des méthodes à comparer.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 

    for idx, (f, t0, y0, _) in enumerate(test_functions):
        for method in methods:
            solver = CauchySolver(f)
            solver.eps = 1e-4 
            if method == 'euler': 
                _, _, errors, Ns = solver.meth_epsilon(y0, t0, method=method, plot=True)
                N_times = len(Ns)
            else :
                _, _, errors, Ns = solver.meth_epsilon(y0, t0, method=method, plot=True, plot_N_times=N_times)
            if method == 'midpoint':
                axs[idx].plot(Ns, errors, label=f'{method}', linestyle='dotted', zorder=0)
            else:
                axs[idx].plot(Ns, errors, label=f'{method}', linestyle='solid', zorder=1)

        axs[idx].set_xscale('log')
        axs[idx].set_yscale('log')
        axs[idx].set_xlabel('Nombre d\'itérations (N)')
        axs[idx].set_ylabel('Erreur (échelle logarithmique)') 
        axs[idx].set_title(f'Précision pour f_{idx + 1}') 
        axs[idx].legend(title="Méthodes")

    plt.tight_layout()
    plt.savefig('img/precision.png')
    plt.show()
    plt.close()


def plot_population(gamma, kappa, N0, t0):
    """
    Tracer la dynamique de la population en utilisant les modèles de Malthus et de Verhulst.
    :param gamma: Taux de croissance de la population.
    :param kappa: Capacité de charge de l'environnement.
    :param N0: Taille initiale de la population.
    :param t0: Temps initial.
    """
    population_models_malthus = PopulationModels(gamma)
    population_models_velhulst = PopulationModels(gamma, kappa)
    t_m, N_m = population_models_malthus.solve(N0, t0)
    t_v, N_v = population_models_velhulst.solve(N0, t0)
    plt.figure(figsize=(8, 6))

    plt.plot(t_m, N_m, label='Modèle Malthusien')
    plt.plot(t_v, N_v, label='Modèle de Verhulst')
    plt.xlabel('Temps')
    plt.ylabel('Population (échelle logarithmique)')
    plt.yscale('log')  
    plt.legend()

    plt.tight_layout()
    plt.savefig('img/population_models.png')
    plt.show()

def plot_population_dynamics(a, b, c, d, y0, t0, method='runge_kutta_4', plot_N_times=None):
    model = PopulationModels(a, b=b, c=c, d=d)
    t, y, errors, Ns = model.solve(y0, t0, t_end=40, method=method, plot=True, plot_N_times=plot_N_times)
    period, periods, t_period, N_period, P_period = model.find_period(t, y)

    N, P = zip(*y)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(t, N, label='Proies (N(t))')
    plt.plot(t, P, label='Prédateurs (P(t))')
    test=0
    for period_time in periods:
        if test == 0:
            plt.axvline(x=period_time, color='gray', linestyle='--', alpha=0.5, label=f'Périodicité ({period:.2f})')
            test = 1
        plt.axvline(x=period_time, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Temps')
    plt.ylabel('Population')
    plt.title('Évolution des populations proies et prédateurs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(N, P, label='Trajectoire')
    singular_points = model.find_singular_points()
    plt.scatter(*zip(*singular_points), color='red', label='Points singuliers')
    plt.xlabel('Proies (N(t))')
    plt.ylabel('Prédateurs (P(t))')
    plt.title('Portrait de phase')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('img/population_dynamics.png')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.loglog(Ns, errors, marker='o')
    plt.xlabel('Nombre de pas (N)')
    plt.ylabel('Erreur')
    plt.title('Erreur en fonction du nombre de pas (échelle logarithmique)')
    plt.grid(True)
    plt.savefig('img/population_dynamics_error.png')
    plt.show()

def plot_solutions_around_initial(a, b, c, d, y0, t0, method='runge_kutta_4', delta=0.1, num_points=9):
    model = PopulationModels(a, b=b, c=c, d=d)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].plot([], [], label='Proies', color='yellow')
    axs[0].plot([], [], label='Prédateurs', color='grey')

    for i in range(num_points):
        for j in range(num_points):
            y0_perturbed = [y0[0] + (i - num_points // 2) * delta, y0[1] + (j - num_points // 2) * delta]
            t, y = model.solve(y0_perturbed, t0, method=method, end_time=10)
            N, P = zip(*y)
            axs[0].plot(t, N, alpha=0.07 , color='yellow')
            axs[0].plot(t, P, alpha=0.07, color='grey')
            axs[1].plot(N, P, alpha=0.07, color='gray')
    t_original, y0_original = model.solve(y0, t0, method=method)
    N0, P0 = zip(*y0_original)
    axs[0].plot(t_original, N0, label='Proies (Condition initiale)', color='red')
    axs[0].plot(t_original, P0, label='Prédateurs (Condition initiale)', color='blue')
    axs[0].set_xlabel('Temps')
    axs[0].set_ylabel('Population')
    axs[0].set_title(f'Évolution des populations proies et prédateurs avec perturbations de {delta}')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(N0, P0, label='Condition initiale', color='red')
    axs[1].set_xlabel('Proies (N(t))')
    axs[1].set_ylabel('Prédateurs (P(t))')
    axs[1].set_title(f'Portrait de phase avec perturbations de {delta}')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    # plt.savefig('img/solutions_around_initial.png')
    plt.show()

def plot_all_filters(image_filter):
    """
    Compute and visualize the original image alongside the three filtered images.
    :param image_filter: Instance of ImageFilter class.
    """
    heat_filtered = image_filter.solve_heat_equation()
    pm_filtered = image_filter.solve_perona_malik()
    gaussian_filtered = image_filter.gaussian_convolution(image_filter.image)

    _, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image_filter.image, cmap='gray')
    axs[0].set_title('\nImage Originale', fontsize=18)
    axs[0].axis('off')

    axs[1].imshow(heat_filtered, cmap='gray')
    axs[1].set_title('\nÉquation de la Chaleur', fontsize=18)
    axs[1].axis('off')

    axs[2].imshow(gaussian_filtered, cmap='gray')
    axs[2].set_title('\nConvolution Gaussienne', fontsize=18)
    axs[2].axis('off')

    axs[3].imshow(pm_filtered, cmap='gray')
    axs[3].set_title('\nPerona-Malik', fontsize=18)
    axs[3].axis('off')


    plt.tight_layout()
    plt.savefig('img/all_filters.png')
    plt.show()

if __name__ == "__main__":
    import argparse
    import os

    os.makedirs("graphs", exist_ok=True)

    arglist = ['cauchy_on_tan_field', 'precision', 'population', 'population_dyn', 'plot_solutions_around_initial', 'plot_all_filters']
    parser = argparse.ArgumentParser(description="Tracer des courbes avec spline cubique.")
    parser.add_argument('-a', '--algo', type=str, nargs='+', choices=arglist,
                        help=f"Choisissez parmi : {arglist}")
    args = parser.parse_args()

    algo_selected = args.algo if args.algo else arglist
    if 'cauchy_on_tan_field' in algo_selected:
        for f, t0, y0, expected in test_functions:
            plot_cauchy_on_tan_field(f, t0, expected, method='runge_kutta_4')
    if 'precision' in algo_selected:
        plot_precision(test_functions, methods)
    if 'population' in algo_selected:
        gamma = 0.85
        kappa = 9000
        N0 = [100]
        t0 = 0
        plot_population(gamma, kappa, N0, t0)
    if 'population_dyn' in algo_selected:
        a, b, c, d = 1.1, 0.08, 0.08, 0.89
        y0 = [1.66, 0.12]
        t0 = 0
        plot_population_dynamics(a, b, c, d, y0, t0, method='runge_kutta_4', plot_N_times=10)
    if 'plot_solutions_around_initial' in algo_selected:
        a, b, c, d = 1.1, 0.08, 0.08, 0.89
        y0 = [1.66, 0.12]
        t0 = 0
        plot_solutions_around_initial(a, b, c, d, y0, t0, method='runge_kutta_4', delta=0.01)
    if 'plot_all_filters' in algo_selected:
        from skimage import data
        image = data.camera()
        image_filter = ImageFilter(image, dt_heat=0.1, dt_pm=0.2, max_iter=10, sigma=1.0)
        plot_all_filters(image_filter)