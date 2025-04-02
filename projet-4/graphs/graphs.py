import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

from graphs.Plotter import Plotter
from NewtonRaphsonSolver import NewtonRaphsonSolver
from LagrangianPoints import LagrangianPoints
from Bairstow import Bairstow




root = "/home/melissa/Documents/ENSEIRB-MATMECA/1A/S6/algo-num/is104-p4-25627/img/"


def convergence_newton_solver():
    # Define a sample nonlinear function and its Jacobian
    def f(x):
        return x**2 - 2

    def J(x):
        return 2*x

    # Initial guess
    U0 = np.array([0.001])

    # Create solver instances
    solver_with_bt = NewtonRaphsonSolver(f, J, N=50)
    solver_without_bt = NewtonRaphsonSolver(f, J, N=50)

    _, convergence_bt = solver_with_bt.solve(U0, backtracking=True, graph=True)

    _, convergence_no_bt = solver_without_bt.solve(U0, backtracking=False, graph=True)

    plt.figure()
    plt.plot(convergence_bt, label="Avec backtracking")
    plt.plot(convergence_no_bt, label="Sans backtracking")
    plt.xlabel("Iterations")
    plt.ylabel("Norme de f(U)")
    plt.yscale('log')
    plt.title("Convergence de Newton Raphson")
    plt.legend()
    plt.grid()
    # plt.savefig(root + "convergence_comparison.png")
    plt.show()

def precision_vs_time():
    def data_generator(size):
        A = np.random.rand(size, size)
        b = np.random.rand(size)

        def f(x):
            return np.dot(A, x) - b

        def J(x):
            return A

        return f, J, np.random.rand(size)

    def test_function(algo_func, data):
        f, J, U0 = data
        start_time = time.time()
        U = algo_func(f, J, U0)
        elapsed_time = time.time() - start_time
        precision = np.linalg.norm(f(U))
        return precision, elapsed_time

    algorithms = {
        "Backtracking": lambda f, J, U0: NewtonRaphsonSolver(f, J, N=50).solve(U0, backtracking=True, graph=False),
        "Sans Backtracking": lambda f, J, U0: NewtonRaphsonSolver(f, J, N=50).solve(U0, backtracking=False, graph=False),
        "Numpy": lambda f, J, U0: np.linalg.solve(J(U0), -f(U0)) + U0
    }

    legends = ["Backtracking", "Sans Backtracking", "Numpy"]

    plotter.plot_time_precision(
        algorithms=algorithms,
        data_generator=data_generator,
        test_function=test_function,
        save_path=root + "time_and_precision_vs_size.png",
        legends=legends,
        log_y_time=True,
        log_y_precision=True,
        x_label="Taille de la matrice",
        y_label_time="Temps (s)",
        y_label_precision="Erreur relative",
        title_time="Temps en fonction de la taille de la matrice",
        title_precision="Précision en fonction de la taille de la matrice"
    )

def plot_force_field():
    lg = LagrangianPoints(
            pos1 = np.array([0.0, 0.0]), k1 = 1.0, 
            pos2 = np.array([1.0, 0.0]), k2 = 0.01,
            k_c = 1.0
        )
    initial_guess = np.array([1.5, 0.0])
    dec = 5
    resolution=1000
    x_min = y_min = - initial_guess[0] 
    x_max = y_max = initial_guess[0]
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    F_c = np.zeros((resolution, resolution, 2))
    F_g1 = np.zeros((resolution, resolution, 2))
    F_g2 = np.zeros((resolution, resolution, 2))

    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            F_c[i, j] = lg.c_force(point)
            F_g1[i, j] = lg.g_force1(point)
            F_g2[i, j] = lg.g_force2(point)

    F_tot = F_c + F_g1 + F_g2

    magnitude_tot = np.linalg.norm(F_tot, axis=2)

    plt.imshow(magnitude_tot, cmap='plasma', extent=[x_min, x_max, y_min, y_max], origin='lower', norm=matplotlib.colors.LogNorm(vmin=1e-4, vmax=100))

    plt.plot(lg.pos1[0], lg.pos1[1], 'o', color='b', markersize=12, label='Objet massique 1')
    plt.plot(lg.pos2[0], lg.pos2[1], 'o', color='g', markersize=12, label='Objet massique 2')

    lagrange_points = [lg.solve(initial_guess + i) for i in [[0, 0], [dec, 0], [-dec, 0], [0, dec], [0, -dec]]]

    print(len(lagrange_points))
    for i, point in enumerate(lagrange_points):
        print(point)
        if i == 0:
            plt.plot(point[0], point[1], 'x', color='black', markersize=10, label='Points de Lagrange')
        else:
            plt.plot(point[0], point[1], 'x', color='black', markersize=10)

    plt.title('Magnitude de la force totale')
    plt.colorbar()
    plt.legend()
    plt.show()
        
def plot_lagrange_points_planetary_system():
    lg = LagrangianPoints(
        pos1 = np.array([0.0, 0.0]), k1 = 1.0, 
        pos2 = np.array([1.0, 0.0]), k2 = 0.01,
        k_c = 1.0
    )
    pi_2 = lg.barycenter[0]
    L_1, L_2, L_3, L_4, L_5 = [lg.solve(initial_guess) for initial_guess in ['L1', 'L2', 'L3', 'L4', 'L5']]

    theta = np.linspace(0, np.pi, 100)
    x_2 = (1 - pi_2) * np.cos(theta)
    y_2 = (1 - pi_2) * np.sin(theta)
    x_1 = (-pi_2) * np.cos(theta)
    y_1 = (-pi_2) * np.sin(theta)

    plt.figure(figsize=(5, 5), dpi=96)
    plt.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])), color='k')
    plt.plot(np.hstack((x_1, x_1[::-1])), np.hstack((y_1, -y_1[::-1])), color='k')
    plt.plot([-pi_2, 0.5 - pi_2, 1 - pi_2, 0.5 - pi_2, -pi_2], [0, np.sqrt(3) / 2, 0, -np.sqrt(3) / 2, 0], 'k--')

    plt.plot(L_1[0], L_1[1], 'rv', label='L1')
    plt.plot(L_2[0], L_2[1], 'r^', label='L2')
    plt.plot(L_3[0], L_3[1], 'rp', label='L3')
    plt.plot(L_4[0], L_4[1], 'rX', label='L4')
    plt.plot(L_5[0], L_5[1], 'rs', label='L5')

    plt.plot(0, 0, 'ko', markersize=10, label='Terre') 
    plt.plot(1 - pi_2, 0, 'go', label='Lune')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.xlabel('$x^*$')
    plt.ylabel('$y^*$')
    plt.title('Système planétaire')
    plt.legend()
    plt.axis('equal')
    plt.show()


def plot_function_and_roots():
    def polynomial_function(x, coefficients):
        return sum(c * x**i for i, c in enumerate(reversed(coefficients)))

    coefficients = [1, -6, 11, -6]  # Example polynomial: x^3 - 6x^2 + 11x - 6
    bairstow_solver = Bairstow(coefficients)
    roots = bairstow_solver.solve()

    x = np.linspace(min(roots) - 1, max(roots) + 1, 1000)
    y = polynomial_function(x, coefficients)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=bairstow_solver, color="blue")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.scatter(roots, [0] * len(roots), color="red", label="Racines", zorder=5)
    plt.title("Graphique de la fonction et des racines")
    plt.xlabel("x")
    plt.ylabel(bairstow_solver)
    plt.legend()
    plt.grid()
    plt.show()


def compare_newton_bairstow_roots():
    coefficients = [
        [1, -6, 11, -6],  # x^3 - 6x^2 + 11x - 6
        [1, 0, -4],        # x^2 - 4
        [1, 0, -4, 0, 0],     # x^3 - 4x
    ]

    algorithms = {
        "Bairstow": lambda coeffs: Bairstow(coeffs).solve(),
        "Newton-Raphson": lambda coeffs: NewtonRaphsonSolver(
        lambda x: np.polyval(coeffs, x),
        lambda x: np.polyval(np.polyder(coeffs), x)
        ).solve(np.array([0.5]), backtracking=True, graph=False)[0],
    }

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.set_xlabel("Nombre de racines correctement trouvées")
    ax1.set_ylabel("Temps (s)", color="tab:blue")
    ax2.set_ylabel("Nombre de racines correctement trouvées", color="tab:orange")

    for i, coefficients in enumerate(coefficients):
        degree = len(coefficients) - 1
        for j, (name, func) in enumerate(algorithms.items()):
            try:
                start_time = time.time()
                roots = func(coefficients)
                elapsed_time = time.time() - start_time
                if not isinstance(roots, (list, np.ndarray)):
                    roots = [roots] 
                correct_roots = sum(
                1 for root in roots if np.abs(np.polyval(coefficients, root)) <= 1e-6
                )
                ax1.scatter(correct_roots, elapsed_time, label=f"{name} (deg {degree})", alpha=0.7)
                ax2.scatter(correct_roots, correct_roots, color="tab:orange", alpha=0.7)
            except ValueError as e:
                print(f"Error with {name} on test case {i + 1}: {e}")

    ax1.set_title("Comparaison des méthodes de Bairstow et de Newton-Raphson")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig("bairstow_plot_comparison.png")

if __name__ == "__main__":
    import argparse
    arglist = ['convergence',             # fait le 19/03/2025
               'time_vs_precision',
                'force_field',
                'lagrange_points_planetary_system',
                'plot_function_and_roots', 
                'compare_newton_bairstow_roots'
               ]
    parser = argparse.ArgumentParser(description='Choisissez les algorithmes à tracer.')
    parser.add_argument('-a',
    '--algo', type=str, nargs='+', choices=arglist, help=f'Algorithmes à tracer parmi {arglist}')
    args = parser.parse_args()

    plotter = Plotter(iterations_range=(4, 1000), step=1, repetitions=10)

    algorithms_to_run = args.algo if args.algo else arglist

    if 'convergence' in algorithms_to_run:
        convergence_newton_solver()
    if 'time_vs_precision' :
        precision_vs_time()
    if 'force_field' in algorithms_to_run:
        plot_force_field()
    if 'lagrange_points_planetary_system' in algorithms_to_run:
        plot_lagrange_points_planetary_system()
    if 'plot_function_and_roots' in algorithms_to_run:
        plot_function_and_roots()
    if 'compare_newton_bairstow_roots' in algorithms_to_run:
        compare_newton_bairstow_roots()

 
