import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
from scipy import integrate
from CubicSpline import CubicSpline
from AirfoilDataLoader import AirfoilDataLoader
from NumericalIntegration import NumericalIntegration

def plot_polynomials():
    a, b = 0, 1
    N = 10  # Spécifiez le nombre de points de données

    poly_funcs = [
        lambda x: x**2,
        lambda x: x**3,
        lambda x: np.polyval([1, -2, 3, 5], x)
    ]
    poly_names = ["x^2", "x^3", "x^3 - 2x^2 + 3x + 5"]
    poly_derivs = [(2*a, 2*b), (3*(a**2), 3*(b**2)), (3 * a**2 - 4 * a + 3, 3 * b**2 - 4 * b + 3)]

    for f, name, deriv in zip(poly_funcs, poly_names, poly_derivs):
        xa = np.linspace(a, b, N)
        ya = f(xa)

        spline_nat = CubicSpline(xa, ya)
        spline_der = CubicSpline(xa, ya, yp1=deriv[0], ypn=deriv[1])

        x_interp = np.linspace(a, b, 10000)
        y_interp_nat = [spline_nat.interpolate(x) for x in x_interp]
        y_interp_der = [spline_der.interpolate(x) for x in x_interp]
        y_true = f(x_interp)

        plt.figure()
        plt.plot(x_interp, y_true, 'b-', label='Original')
        plt.plot(xa, ya, 'ko', label='Data points')

        # Dégradé de couleur pour les courbes
        for i, alpha in enumerate(np.linspace(0.1, 1, 10)):
            plt.plot(x_interp, y_interp_nat, color=(0, 1, 0, alpha), label=f'Natural spline (step {i+1})' if i == 9 else "")
            plt.plot(x_interp, y_interp_der, color=(1, 0, 0, alpha), label=f'Clamped spline (step {i+1})' if i == 9 else "")

        plt.title(f"Spline {name} (N={N})")
        plt.legend()
        plt.grid()
        plt.savefig(f"img/spline_{name.replace(' ', '').replace('^','')}_{N}.png")
        plt.close()

def plot_trigonometrics():
    N = 10  # Spécifiez le nombre de points de données
    a,b=0,1
    trig_funcs = [np.sin, np.cos, lambda x: x + 2 * np.sin(x)]
    trig_names = ["sin(x)", "cos(x)", "x + 2sin(x)"]
    trig_derivs = [(np.cos(a), np.cos(b)), (-np.sin(a), -np.sin(b)), (1 + 2*np.cos(a), 1 + 2*np.cos(b))]

    for f, name, deriv in zip(trig_funcs, trig_names, trig_derivs):
        xa = np.linspace(a,b, N)
        ya = f(xa)

        spline_nat = CubicSpline(xa, ya)
        spline_der = CubicSpline(xa, ya, yp1=deriv[0], ypn=deriv[1])

        x_interp = np.linspace(a,b, 10000)
        y_interp_nat = [spline_nat.interpolate(x) for x in x_interp]
        y_interp_der = [spline_der.interpolate(x) for x in x_interp]
        y_true = f(x_interp)

        plt.figure()
        plt.plot(xa, ya, 'ko', label='Data points')
        plt.plot(x_interp, y_interp_nat, 'g--', label='Natural spline')
        plt.plot(x_interp, y_interp_der, 'r--', label='Clamped spline')
        plt.plot(x_interp, y_true, 'b-', label='Original')
        plt.title(f"Spline {name} (N={N})")
        plt.legend()
        plt.grid()
        plt.savefig(f"img/spline_{name.replace(' ', '')}_{N}.png")
        plt.close()

def plot_errors():
    N = [50]

    trig_funcs = [np.sin, np.cos, lambda x: x + 2 * np.sin(x)]
    trig_names = ["sin(x)", "cos(x)", "x + 2sin(x)"]
    trig_derivs = [(1, np.cos(1)), (0, -np.sin(1)), (1 + 2*np.cos(0), 1 + 2*np.cos(1))]
    x_interp = np.linspace(0, 1, 10000)

    colors = ['g','b','m','c']
    
    for f, name, deriv in zip(trig_funcs, trig_names, trig_derivs):
        plt.figure()
        plt.title(f"Spline errors for {name}")
        error = []
        for i,n in enumerate(N):
            xa = np.linspace(0, 1, n)
            ya = f(xa)

            spline_der = CubicSpline(xa, ya, yp1=deriv[0], ypn=deriv[1])

            y_error = [abs(spline_der.interpolate(x) - f(x)) for x in x_interp]
            error.append(np.linalg.norm(y_error,ord=np.inf))
            
            plt.plot(x_interp, y_error, f'{colors[i]}-', label=f'Spline error N={n}')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.savefig(f"img/spline_errors_{name.replace(' ', '')}.png")
        plt.close()
        
def plot_norm_errors(funcs, begin_N=10, end_N=150, step_N=1):
    """
    Trace les erreurs de spline pour différentes fonctions et valeurs de N.
    
    Parameters:
    - funcs (list): Liste de dictionnaires contenant les fonctions, leurs noms et dérivées.
    - begin_N (int): Valeur de départ pour N.
    - end_N (int): Valeur de fin pour N.
    - step_N (int): Pas d'incrémentation pour N.
    """
    x_interp = np.linspace(0, 1, 10000)

    plt.figure()
    plt.title("Évolution de l'erreur en fonction de N pour toutes les fonctions")
    plt.xlabel("N")
    plt.ylabel("Erreur (échelle logarithmique)")
    plt.yscale('log')
    plt.grid()

    for func in funcs:
        f = func['function']
        name = func['name']
        deriv = func['derivative']
        degree = func['degree']
        error = []
        for n in range(begin_N, end_N + 1, step_N):
            print(f"Traitement de {name} avec N={n}")
            xa = np.linspace(0, 1, n)
            ya = f(xa)
            spline_der = CubicSpline(xa, ya, yp1=deriv[0], ypn=deriv[1])
            y_error = [abs(spline_der.interpolate(x) - f(x)) for x in x_interp]
            error.append(np.linalg.norm(y_error, ord=np.inf) - np.finfo(float).eps)

        # Définir la couleur en fonction du degré
        color_map = {
            'circular': (0.5, 0, 0.5),
            1: (0, 1, 0),
            2: (0, 0, 1),
            3: (1, 0, 0)
        }
        base_color = color_map.get(degree, (0, 0, 0))  # Default to black if degree is not in the map
        variation = funcs.index(func) * 0.2 / len(funcs)  # Increase variation factor for more distinct colors
        varied_color = tuple(min(1, max(0, c + variation)) for c in base_color)  # Ensure RGB values stay in [0, 1]
        alpha = 0.5 + 0.5 * (funcs.index(func) / len(funcs))  # Vary alpha from 0.5 to 1
        color = (*varied_color, alpha)

        plt.plot(range(begin_N, end_N + 1, step_N), error, label=name, color=color)

    plt.legend(title="Fonctions")
    plt.show()
    plt.close()

def plot_airfoil():
    airfoil_name = 'fx72150a'
    _, ex, ey, ix, iy = AirfoilDataLoader.load(f'./data/{airfoil_name}.dat')
    
    extrados = CubicSpline(ex, ey)
    intrados = CubicSpline(ix, iy)

    x_vals = np.linspace(0, 1, 100000)
    ey_interp = [extrados.interpolate(x) for x in x_vals]
    print(ey_interp)
    iy_interp = [intrados.interpolate(x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(ex, ey, 'yo', label='Extrados points')
    plt.plot(ix, iy, 'ro', label='Intrados points')
    plt.plot(x_vals, ey_interp, 'b-', label='Spline Extrados')
    plt.plot(x_vals, iy_interp, 'b-', label='Spline Intrados')
    plt.title(f"Airfoil: {airfoil_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

def evaluate_convergence(functions, lower_limit, upper_limit, max_iterations):
    """
    Évaluer la vitesse de convergence de chaque méthode d'intégration numérique pour une liste de fonctions.
    
    Parameters:
    - functions (list): Liste de dictionnaires contenant les fonctions et leurs intégrales exactes.
    - lower_limit (float): Limite inférieure de l'intervalle.
    - upper_limit (float): Limite supérieure de l'intervalle.
    - max_iterations (int): Nombre maximum d'itérations pour l'intégration.
    
    Returns:
    - results (list): Liste de résultats contenant les données de convergence pour chaque fonction.
    """
    methods = ['left_rectangle', 'right_rectangle', 'midpoint', 'trapezoidal', 'simpson']
    results = []

    for func in functions:
        function = func['function']
        exact_integral = func['exact_integral']
        func_name = func['name']
        convergence_data = {}
        execution_times = {}

        for method in methods:
            integrator = NumericalIntegration(function, lower_limit, upper_limit, max_iterations)
            errors = []
            execution_time = []
            num_subintervals = 1
            while num_subintervals < max_iterations:
                start_time = time.time()
                integral = getattr(integrator, f'{method}_rule')(num_subintervals)
                end_time = time.time()
                error = abs(integral - exact_integral)
                errors.append((num_subintervals, error))
                execution_time.append((num_subintervals, end_time - start_time))
                num_subintervals *= 2
            convergence_data[method] = errors
            execution_times[method] = execution_time

        results.append({
            'function_name': func_name,
            'convergence_data': convergence_data,
            'execution_times': execution_times
        })

    return results
def plot_all_convergences(results):
    """
    Trace les courbes de convergence pour toutes les fonctions sur le même graphique.
    
    Parameters:
    - results (list): Liste contenant les données de convergence pour chaque fonction.
    """
    plt.figure(figsize=(12, 8))
    plt.title("Convergence des méthodes d'intégration numérique")
    plt.xlabel("Nombre de sous-intervalles")
    plt.ylabel("Erreur")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()

    line_styles = {
        'left_rectangle': '-',
        'right_rectangle': '--',
        'midpoint': '-.',
        'trapezoidal': ':',
        'simpson': (0, (3, 5, 1, 5))  # Custom dashed style
    }
    # Assign unique colors to each function
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    i = 0
    # Plot data
    for idx, result in enumerate(results):
        func_name = result['function_name']
        convergence_data = result['convergence_data']
        a =0
        for method, errors in convergence_data.items():
            num_subintervals, error_values = zip(*errors)
            if i==0:
                plt.plot(
                    num_subintervals, error_values, 
                    label=f"{method}", 
                    color='black',  
                    linestyle=line_styles.get(method, '-')
                )
                plt.plot(
                    num_subintervals, error_values, 
                    color=colors[idx], 
                    linestyle=line_styles.get(method, '-')
                )
                if a == 0:
                    tpm_num_subintervals, tmp_error_values = num_subintervals, error_values
                    tmp_method = method
                    tmp_color = colors[idx]
                    tmp_line_style = line_styles.get(method, '-')
                a += 1
            if a == 0:
                plt.plot(
                    num_subintervals, error_values, 
                    label=f"{func_name}", 
                    color=colors[idx], 
                    linestyle=line_styles.get(method, '-')
                )
                a += 1
            else:
                plt.plot(
                    num_subintervals, error_values,
                    color=colors[idx], 
                    linestyle=line_styles.get(method, '-')
                )
        if i == 0:
            plt.plot(
                tpm_num_subintervals, tmp_error_values, 
                label=f"{func_name}", 
                color=tmp_color, 
                linestyle=tmp_line_style
            )
        i += 1

    # Add legend
    plt.legend(title="Fonctions et Méthodes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    import os

    os.makedirs("graphs", exist_ok=True)

    arglist = ['poly', 'trigo', 'airfoil','error','norm_error', 'integration_convergence']
    parser = argparse.ArgumentParser(description="Tracer des courbes avec spline cubique.")
    parser.add_argument('-a', '--algo', type=str, nargs='+', choices=arglist,
                        help=f"Choisissez parmi : {arglist}")
    args = parser.parse_args()

    algo_selected = args.algo if args.algo else arglist

    if 'poly' in algo_selected:
        plot_polynomials()
    if 'trigo' in algo_selected:
        plot_trigonometrics()
    if 'airfoil' in algo_selected:
        plot_airfoil()
    if 'error' in algo_selected:
        plot_errors()
    if 'norm_error' in algo_selected:
        funcs = [
            {
                'function': lambda x: x**3,
                'name': 'x^3',
                'derivative': (3, 3),
                'degree': 3
            },
            {
                'function': lambda x: np.polyval([1, -2, 3, 5], x),
                'name': 'x^3 - 2x^2 + 3x + 5',
                'derivative': (3 * 1**2 - 4 * 1 + 3, 3 * 2**2 - 4 * 2 + 3),
                'degree': 3
            },
            {
                'function': lambda x: np.sin(x),
                'name': 'sin(x)',
                'derivative': (np.cos(0), np.cos(1)),
                'degree': 'circular'
            },
            {
                'function': lambda x: np.cos(x),
                'name': 'cos(x)',
                'derivative': (-np.sin(0), -np.sin(1)),
                'degree': 'circular'
            },
            {
                'function': lambda x: x**2 + 2*x + 1,
                'name': 'x^2 + 2x + 1',
                'derivative': (2*0 + 2, 2*1 + 2),
                'degree': 2
            },
            {
                'function': lambda x: x**2 -6*x + 8,
                'name': 'x^2 - 6x + 8',
                'derivative': (2*0 - 6, 2*1 - 6),
                'degree': 2
            },
            {
                'function': lambda x: x,
                'name': 'x',
                'derivative': (1, 1),
                'degree': 1
            },
            {
                'function': lambda x: 9*x - 3,
                'name': '9x - 3',
                'derivative': (9, 9),
                'degree': 1
            }
        ]
        plot_norm_errors(funcs, begin_N=2, end_N=500, step_N=1)

    if 'integration_convergence' in algo_selected:
        functions = [
            {
            'function': lambda x: x,
            'name': 'x',
            'degree': 1,
            'exact_integral': 200
            },
            {
            'function': lambda x: x**2,
            'name': 'x^2',
            'degree': 2,
            'exact_integral': 8000/3
            },
            {
            'function': lambda x: x**3,
            'name': 'x^3',
            'degree': 3,
            'exact_integral': 40000
            },
            {
            'function': lambda x: np.sin(x),
            'name': 'sin(x)',
            'degree': 'circular',
            'exact_integral': 2*np.sin(10)**2
            },
            {
            'function': lambda x: -0.02*(x+2)**2 + x,
            'name': '-0.02*(x+2)**2 + x',
            'degree': 2,
            'exact_integral': 129.067
            }
        ]

        lower_limit = 0
        upper_limit = 20
        max_iterations = 100000

        results = evaluate_convergence(functions, lower_limit, upper_limit, max_iterations)
        plot_all_convergences(results)