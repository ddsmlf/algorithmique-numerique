import numpy as np
import time
from equation_chaleur import simulation_grad, simulation_cholesky
import numpy as np
import matplotlib.pyplot as plt
import matrix_generation as mg

import matplotlib.pyplot as plt

def show_chaleur(F, T, N):
    """
    Affiche le flux de chaleur initial et la température après diffusion.

    Parameters:
    F (numpy.ndarray): Le flux de chaleur initial, de taille (N, N).
    T (numpy.ndarray): La température après diffusion, de taille (N, N).
    N (int): La dimension de la grille (NxN).

    Returns:
    None
    """
    F = F.reshape(N, N)
    T = T.reshape(N, N)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.matshow(F, fignum=False)
    plt.title("Flux de chaleur initial")

    plt.subplot(1, 2, 2)
    plt.matshow(T, fignum=False)
    plt.title("Température après diffusion")

    plt.show()

def show_evolutions(tabx, N):
    """
    Affiche l'évolution de la température au cours du temps.

    Parameters:
    tabx (list): La liste des températures à chaque itération.
    N (int): La dimension de la grille (NxN).

    Returns:
    None
    """
    plt.figure(figsize=(10, 10))
    for i in range(40):
        plt.subplot(4, 10, i+1)
        plt.matshow(tabx[i].reshape(N, N), fignum=False)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def compare(N, F):
    """
    Plot the difference between different methods.

    Parameters:
    N (int): The dimension of the grid (NxN).
    F (numpy.ndarray): The initial heat flux, of size (N, N).

    Returns:
    None
    """
    T_grad, tab, tabx, iteration = simulation_grad(N, F)
    T_cholesky = simulation_cholesky(N, F)
    T_cholesky_incomplet = simulation_cholesky(N, F, False)

    T_grad = T_grad.reshape(N, N)
    T_cholesky = T_cholesky.reshape(N, N)
    T_cholesky_incomplet = T_cholesky_incomplet.reshape(N, N)

    T_numpy = np.linalg.solve(mg.M_c(N), F.flatten()).reshape(N, N)
    # plot_all(T_grad, T_cholesky, T_cholesky_incomplet, T_numpy, N, F)
    # plot_difference_all(T_grad, T_cholesky, T_cholesky_incomplet, T_numpy)
    compare_cholesky_methods(4, 500, 1)
def plot_all(T_grad, T_cholesky, T_cholesky_incomplet, T_numpy, N, F):
    """
    Plot the results of different methods.

    Parameters:
    T_grad (numpy.ndarray): The temperature after diffusion with the gradient method.
    T_cholesky (numpy.ndarray): The temperature after diffusion with the Cholesky method.
    T_cholesky_incomplet (numpy.ndarray): The temperature after diffusion with the incomplete Cholesky method.
    T_numpy (numpy.ndarray): The temperature after diffusion with the numpy method.

    Returns:
    None
    """

    F = F.reshape(N, N)
    T_grad = T_grad.reshape(N, N)
    T_cholesky = T_cholesky.reshape(N, N)
    T_cholesky_incomplet = T_cholesky_incomplet.reshape(N, N)
    T_numpy = T_numpy.reshape(N, N)

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.matshow(F, fignum=False)
    plt.title("Flux de chaleur initial")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 2)
    plt.matshow(T_grad, fignum=False)
    plt.title("Gradient conjugué")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 3)
    plt.matshow(T_cholesky, fignum=False)
    plt.title("Cholesky complet")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 4)
    plt.matshow(T_cholesky_incomplet, fignum=False)
    plt.title("Cholesky incomplet")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 5)
    plt.matshow(T_numpy, fignum=False)
    plt.title("Numpy")
    plt.xticks([])
    plt.yticks([])

    plt.show()

def compare_cholesky_methods(start, end, step):
    precisions_grad = []
    precisions_cholesky = []
    precisions_cholesky_incomplet = []
    times_grad = []
    times_cholesky = []
    times_cholesky_incomplet = []
    time_numpy = []

    for i in range(start, end+1, step):
        print(i)
        A = mg.M_c(i)
        F = np.zeros((i, i))
        F[int(i/2)][int(i/2)] = 10

        times_grad_a = time.time()
        T_grad, tab, tabx, iteration = simulation_grad(i, F)
        times_grad_b = time.time()
        times_grad.append(times_grad_b - times_grad_a)

        times_cholesky_a = time.time()
        T_cholesky = simulation_cholesky(i, F)
        times_cholesky_b = time.time()
        times_cholesky.append(times_cholesky_b - times_cholesky_a)

        times_cholesky_incomplet_a = time.time()
        T_cholesky_incomplet = simulation_cholesky(i, F, False)
        times_cholesky_incomplet_b = time.time()
        times_cholesky_incomplet.append(times_cholesky_incomplet_b - times_cholesky_incomplet_a)

        time_numpy_a = time.time()
        T_numpy = np.linalg.solve(A, F.flatten()).reshape(i, i)
        time_numpy_b = time.time()
        time_numpy.append(time_numpy_b - time_numpy_a)

        T_grad = T_grad.reshape(i, i)
        T_cholesky = T_cholesky.reshape(i, i)
        T_cholesky_incomplet = T_cholesky_incomplet.reshape(i, i)

        precisions_grad.append(np.linalg.norm(T_numpy - T_grad, 1) / (np.linalg.norm(T_numpy, 1) * i * np.finfo(float).eps))
        precisions_cholesky.append(np.linalg.norm(T_numpy - T_cholesky, 1) / (np.linalg.norm(T_numpy, 1) * i * np.finfo(float).eps))
        precisions_cholesky_incomplet.append(np.linalg.norm(T_numpy - T_cholesky_incomplet, 1) / (np.linalg.norm(T_numpy, 1) * i * np.finfo(float).eps))
    
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(range(start, end+1, step), times_grad, label="Gradient conjugué")
    plt.plot(range(start, end+1, step), times_cholesky, label="Cholesky complet")
    plt.plot(range(start, end+1, step), times_cholesky_incomplet, label="Cholesky incomplet")
    plt.plot(range(start, end+1, step), time_numpy, label="Numpy")
    plt.xlabel('Taille de la matrice')
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Comparaison des temps")
    plt.legend()
    plt.yscale('log')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.plot(range(start, end+1, step), precisions_grad, label="Gradient conjugué")
    plt.plot(range(start, end+1, step), precisions_cholesky, label="Cholesky complet")
    plt.plot(range(start, end+1, step), precisions_cholesky_incomplet, label="Cholesky incomplet")
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Précision')
    plt.title('Comparaison des précisions')
    plt.legend()
    plt.yscale('log')
    plt.xticks([])
    plt.yticks([])

    plt.show()

def compare_cholesky_methods(start, end, step):
    precisions_grad = []
    precisions_cholesky = []
    precisions_cholesky_incomplet = []
    times_grad = []
    times_cholesky = []
    times_cholesky_incomplet = []
    time_numpy = []

    img_path = '/home/melissa/Documents/ENSEIRB-MATMECA/1A/S6/algo-num/is104-p2-25617/img/temps.png'

    for i in range(start, end+1, step):
        print(i)
        A = mg.M_c(i)
        F = np.zeros((i, i))
        F[int(i/2)][int(i/2)] = 10

        times_grad_a = time.time()
        T_grad, tab, tabx, iteration = simulation_grad(i, F)
        times_grad_b = time.time()
        times_grad.append(times_grad_b - times_grad_a)

        times_cholesky_a = time.time()
        T_cholesky = simulation_cholesky(i, F)
        times_cholesky_b = time.time()
        times_cholesky.append(times_cholesky_b - times_cholesky_a)

        times_cholesky_incomplet_a = time.time()
        T_cholesky_incomplet = simulation_cholesky(i, F, False)
        times_cholesky_incomplet_b = time.time()
        times_cholesky_incomplet.append(times_cholesky_incomplet_b - times_cholesky_incomplet_a)

        time_numpy_a = time.time()
        T_numpy = np.linalg.solve(A, F.flatten()).reshape(i, i)
        time_numpy_b = time.time()
        time_numpy.append(time_numpy_b - time_numpy_a)

        T_grad = T_grad.reshape(i, i)
        T_cholesky = T_cholesky.reshape(i, i)
        T_cholesky_incomplet = T_cholesky_incomplet.reshape(i, i)

        precisions_grad.append(np.linalg.norm(T_numpy - T_grad, 1) / (np.linalg.norm(T_numpy, 1) * i * np.finfo(float).eps))
        precisions_cholesky.append(np.linalg.norm(T_numpy - T_cholesky, 1) / (np.linalg.norm(T_numpy, 1) * i * np.finfo(float).eps))
        precisions_cholesky_incomplet.append(np.linalg.norm(T_numpy - T_cholesky_incomplet, 1) / (np.linalg.norm(T_numpy, 1) * i * np.finfo(float).eps))
    
        # Création du graphique
        plt.figure(figsize=(12, 12))

        plt.subplot(2, 2, 1)
        plt.plot(range(start, i+1, step), times_grad, label="Gradient conjugué")
        plt.plot(range(start, i+1, step), times_cholesky, label="Cholesky complet")
        plt.plot(range(start, i+1, step), times_cholesky_incomplet, label="Cholesky incomplet")
        plt.plot(range(start, i+1, step), time_numpy, label="Numpy")
        plt.xlabel('Taille de la matrice')
        plt.ylabel("Temps d'exécution (s)")
        plt.title("Comparaison des temps")
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(range(start, i+1, step), precisions_grad, label="Gradient conjugué")
        plt.plot(range(start, i+1, step), precisions_cholesky, label="Cholesky complet")
        plt.plot(range(start, i+1, step), precisions_cholesky_incomplet, label="Cholesky incomplet")
        plt.xlabel('Taille de la matrice')
        plt.ylabel('Précision')
        plt.title('Comparaison des précisions')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        # Sauvegarde du graphique
        plt.savefig(img_path)
        plt.close()

        print(f"Graphique mis à jour et sauvegardé dans {img_path}")

if __name__ == "__main__":
    # Matrice de flux de chaleur
    N = 25
    F = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j or i == N - j - 1:
                F[i, j] = 1

    A = mg.M_c(60)

    # Test sur différence
    compare(N, F)
