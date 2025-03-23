import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp

from tools import generate_random_UV, matrix_multiply, euclidian_distance
from Householder import Householder
from SVD import SVD
from plot_scripts.Plotter import Plotter
from Bidiagonalization import Bidiagonalization
from ImageCompressor import ImageCompressor


root = "/home/melissa/Documents/ENSEIRB-MATMECA/1A/S6/algo-num/is104-p3-25619/img/"

def plot_householder_comparison(plotter, gpu):
    def generate_householder_data(size):
        U, V = generate_random_UV(size)
        X = np.random.rand(size, size)
        householder = Householder(U, V)
        return householder, X
    
    def evaluate_householder_algorithm(algo_func, data):
        householder, X = data
        start = time.time()
        result = algo_func(householder, X)
        elapsed_time = time.time() - start

        result_reference = np.dot(householder.H, X)
        abs_error = np.linalg.norm(result_reference - result)
        rel_error = abs_error / (np.linalg.norm(result_reference) + np.finfo(float).eps)
        
        return rel_error, elapsed_time
    
    def householder_transformation(householder, X):
        return householder.apply_transformation_to_matrix(X)

    def usual_matrix_multiplication(householder, X):
        return matrix_multiply(householder.H, X)

    algorithms = {
        "Householder": householder_transformation,
        "Produit Usuel": usual_matrix_multiplication
    }

    plotter.plot_time_precision(
        algorithms, 
        generate_householder_data, 
        evaluate_householder_algorithm, 
        legends=["Householder", "Produit Usuel"], 
        log_y_time=True, 
        x_label='Taille des matrices', 
        y_label_time='Temps (secondes)', 
        y_label_precision='Erreur relative', 
        title_precision='Comparaison des erreurs relatives', 
        title_time='Temps de calcul en fonction de la taille des matrices', 
        save_path=f"{root}householder_comparaison.png"
    )

def analyze_bidiagonalization_stability(gpu=False):
    condition_numbers = np.logspace(1, 8, num=20) 
    errors = []

    for cond in condition_numbers:
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        A += np.random.rand(3, 3) * 1e-6 
        A = A / np.linalg.cond(A) * cond

        bidiag = Bidiagonalization(A, gpu=gpu)
        Qleft, BD, Qright = bidiag.compute()

        error = np.linalg.norm(Qleft @ BD @ Qright - A) / np.linalg.norm(A)
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.plot(condition_numbers, errors, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Nombre de Conditionnement')
    plt.ylabel('Erreur Relative')
    plt.title('Stabilité Numérique de l\'Algorithme de Bidiagonalisation')
    plt.grid(True)
    plt.show()


def plot_svd_convergence_speed(plotter, gpu=False):
    def generate_random_matrix(size):
        return np.random.rand(size, size)

    def evaluate_svd_convergence(algo_func, data):
        try:
            start = time.time()
            U, S, V = algo_func(data)
            elapsed_time = time.time() - start

            original_matrix = data
            
            relative_error = np.linalg.norm(original_matrix - U @ np.diag(S) @ V) / np.linalg.norm(original_matrix)
            return relative_error, elapsed_time
        except ValueError as e:
            print(f"Erreur pour la taille {data.shape[0]}:", e)
            return None, None

    algorithms = {
        'SVD Personnalisé': lambda A: SVD(A, gpu=gpu).apply_SVD(),
        'SVD NumPy': lambda A: np.linalg.svd(A)
    }

    plotter.plot_time_precision(
        algorithms=algorithms,
        data_generator=generate_random_matrix,
        test_function=evaluate_svd_convergence,
        legends=['SVD Personnalisé'],
        save_path=f"{root}svd_convergence_speed.png",
        x_label='Taille des matrices',
        y_label_time='Nombre d\'itérations',
        y_label_precision='Erreur relative par rapport à NumPy SVD',
        title_time='Temps de calcul en fonction de la taille des matrices',
        title_precision='Erreur relative par rapport à NumPy SVD en fonction de la taille des matrices',
    )

def plot_qr_algorithm_convergence(matrix, gpu=False):
    svd = SVD(matrix, gpu=gpu)
    try:
        _, _, _, off_diag_norms, non_negligible_off_diag_counts = svd.apply_SVD(plot_convergence=True)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(off_diag_norms)
        plt.xlabel("Itérations")
        plt.ylabel("Norme des éléments hors diagonale de S")
        plt.title("Convergence de l'algorithme QR")
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(non_negligible_off_diag_counts)
        plt.xlabel("Itérations")
        plt.ylabel("Nombre d'éléments hors diagonale non négligeables")
        plt.title("Évolution des éléments hors diagonale non négligeables")
        plt.grid()

        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print("Erreur :", e)
        
def compare_qr_methods(plotter, initial_size=4, final_size=100, step=1, repetitions=10, gpu=False):
    def generate_random_matrix(size):
        return Bidiagonalization(np.random.rand(size, size)).compute()[1]

    def evaluate_qr_methods(algo_func, data):
        try:
            temps = time.time()
            result = algo_func(data)
            elapsed_time = time.time() - temps
            Q, R = result
            if isinstance(Q, cp.ndarray):
                data = cp.asarray(data)
            error = np.linalg.norm(data - Q @ R) / (np.linalg.norm(data) + np.finfo(float).eps)
            return error, elapsed_time
        except ValueError as e:
            print(f"Erreur pour la taille {data.shape[0]}:", e)
            return None, None

    algorithms = {
        'QR Bidiagonal': lambda A: SVD(A, gpu=gpu).givens_rotation(A),
        'QR Standard': lambda A: np.linalg.qr(A),
        'QR Householder': lambda A: SVD(A, gpu=gpu).householder_qr(A)
    }

    plotter.plot_time_precision(
        algorithms=algorithms,
        data_generator=generate_random_matrix,
        test_function=evaluate_qr_methods,
        legends=['QR Bidiagonal', 'QR Standard', 'QR Householder'],
        save_path=f"{root}qr_comparaison.png",
        x_label='Taille des matrices',
        y_label_time='Temps moyen (s)',
        y_label_precision='Erreur moyenne',
        # log_y_precision=True,
        title_time='Comparaison des temps de décomposition QR',
        title_precision='Comparaison des erreurs de décomposition QR'
    )

def plot_qr_comparison(sizes, times_bidiagonal, times_standard, errors_bidiagonal, errors_standard, save_path=None):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times_bidiagonal, label="QR Bidiagonal")
    plt.plot(sizes, times_standard, label="QR Standard")
    plt.xlabel("Taille des matrices")
    plt.ylabel("Temps moyen (s)")
    plt.legend()
    plt.title("Comparaison des temps de décomposition QR")

    plt.subplot(1, 2, 2)
    plt.plot(sizes, errors_bidiagonal, label="QR Bidiagonal")
    plt.plot(sizes, errors_standard, label="QR Standard")
    plt.xlabel("Taille des matrices")
    plt.ylabel("Erreur moyenne")
    plt.legend()
    plt.title("Comparaison des erreurs de décomposition QR")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def time_vs_size(k):
    compressor = ImageCompressor("exemple.png")
    time_curve_separate_color = []
    time_curve_merged_color = []
    size_curve = [x for x in range(1, 100, 10)]
    for size in size_curve:
        time_curve_separate_color_moy = []
        time_curve_merged_color_moy = []
        for _ in range(10):
            compressor.update_image(np.random.rand(size, size, 3))
            tps = time.time()
            compressor.compress_image_3_channels(k)
            time_curve_separate_color_moy.append(time.time() - tps)
            tps = time.time()
            compressor.compress_image_3_channels_merged_channels(k)
            time_curve_merged_color_moy.append(time.time() - tps)
        time_curve_separate_color.append(np.mean(time_curve_separate_color_moy))
        time_curve_merged_color.append(np.mean(time_curve_merged_color_moy))

        plt.plot(size_curve[:len(time_curve_separate_color)], time_curve_separate_color, label=f"Canaux séparés")
        plt.plot(size_curve[:len(time_curve_merged_color)], time_curve_merged_color, label=f"Canaux fusionnés")
        plt.legend()
        plt.title("Temps de compression en fonction de la taille de l'image")
        plt.ylabel("Temps de compression (s)")
        plt.xlabel("Taille de l'image (pixels)")
        plt.savefig(f"{root}time_plot.png")
        plt.clf()
        


def gain_error():
    compressor = ImageCompressor("exemple.png")

    errors = []
    gains = []
    for k in range(1, min(compressor.img.shape[0], compressor.img.shape[1]), 10):
        img_compressed_manual = compressor.compress_image_3_channels(k, method="manual")
        img_compressed_merged = compressor.compress_image_3_channels_merged_channels(k, method="manual")

        error_manual = np.linalg.norm(compressor.img - img_compressed_manual) / np.linalg.norm(compressor.img)
        error_merged = np.linalg.norm(compressor.img - img_compressed_merged) / np.linalg.norm(compressor.img)
        errors.append((error_manual, error_merged))

        original_size = compressor.img.nbytes  # taille de l'image originale en octets
        compressed_size_manual = k * (compressor.img.shape[0] + compressor.img.shape[1] + 1) * compressor.img.itemsize  # taille de l'image compressée en octets
        gain_manual = (original_size - compressed_size_manual) / original_size

        compressed_size_merged = k * (compressor.img.shape[0] + compressor.img.shape[1] + 1) * compressor.img.itemsize  # taille de l'image compressée en octets
        gain_merged = (original_size - compressed_size_merged) / original_size

        gains.append((gain_manual, gain_merged))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ks = range(1, min(compressor.img.shape[0], compressor.img.shape[1]), 10)
    errors_manual, errors_merged = zip(*errors)
    gains_manual, gains_merged = zip(*gains)

    plt.subplot(1, 2, 1)
    plt.plot(ks, errors_manual, label='Canaux séparés')
    plt.plot(ks, errors_merged, label='Canaux fusionnés')
    plt.xlabel("Rang k")
    plt.ylabel("Erreur de reconstruction")
    plt.title("Évolution de l'erreur en fonction de k")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ks, gains_manual, label='Canaux séparés')
    plt.plot(ks, gains_merged, 'r--', label='Canaux fusionnés')
    plt.xlabel("Rang k")
    plt.ylabel("Gain de compression")
    plt.title("Évolution du gain de compression en fonction de k")
    plt.grid()
    plt.legend()

    plt.show()


def plot_estimation_gain():
    """Le nombre total de valeurs stockées après compression est donc de k(m + n + 1) ce qui donne un gain de
    place de mn − k(m + n + 1). Pour que la compression soit efficace, il faut que ce gain soit strictement positif,
    mn
    .
    c’est-à-dire que mn − k(m + n + 1) > 0 ou encore k < m+n+1"""
    gain_ratio = []
    n = 100
    m = 100
    for k in range(1, 100):
        gain_ratio.append(((m * n) / (m * n - k * (m + n + 1)))/100)
    plt.plot(range(1, 100), gain_ratio)
    plt.xlabel("Rang k")
    plt.ylabel("Gain de compression (ratio)")
    plt.title("Estimation du gain de compression en fonction de k")
    plt.grid()
    plt.show()

def difference_images():
    """
    affiche une acrte de chaleure montrant les zones de l'iamge qui sont les plus différentes entre l'image compressé poru la méthode par fusion des canaux et celle par canaux séparés
    """
    compressor = ImageCompressor("exemple.png")
    k = 5
    img_compressed_manual = compressor.compress_image_3_channels(k, method="manual")
    img_compressed_merged = compressor.compress_image_3_channels_merged_channels(k, method="manual")

    diff = np.abs(img_compressed_manual - img_compressed_merged)
    plt.imshow(diff)
    plt.title("Différence entre les images compressées")
    plt.show()
        
if __name__ == "__main__":
    import argparse
    arglist = ['hh',                        # fait le 08/03/2025 à 14h20
               'qr',                        # fait le 08/03/2025 à 14h20
               'bd_stability',              # fait le 08/03/2025 à 14h20
               'svd_convergence_speed',     # fait le 08/03/2025 à 14h20  
               'qr_convergence',            # fait le 08/03/2025 à 17h00
               'gain_error',                # fait le 08/03/2025 à 17h00
               'compression_time',          # fait le 08/03/2025 à 17h00 A REFAIRE
               'visual_comparison',         # fait le 09/03/2025 à 12h00
               'estimation_gain',           # fait le 09/03/2025 à 12h30
               'difference_images',         # 
               ]
    parser = argparse.ArgumentParser(description='Choisissez les algorithmes à tracer.')
    parser.add_argument('-a',
    '--algo', type=str, nargs='+', choices=arglist, help=f'Algorithmes à tracer parmi {arglist}')
    parser.add_argument('-g', '--gpu', action='store_true', help='Utiliser l\'accélération GPU')
    args = parser.parse_args()

    plotter = Plotter(iterations_range=(4, 1000), step=1, repetitions=10)

    algorithms_to_run = args.algo if args.algo else arglist

    if 'hh' in algorithms_to_run: 
        plot_householder_comparison(plotter, args.gpu)
    if 'qr' in algorithms_to_run:
        compare_qr_methods(plotter)
    if 'bd_stability' in algorithms_to_run:
        analyze_bidiagonalization_stability()
    if 'qr_convergence' in algorithms_to_run:
        plot_qr_algorithm_convergence(np.random.rand(250, 250))
    if 'compression_time' in algorithms_to_run:
        time_vs_size(5)
    if 'gain_error' in algorithms_to_run:
        gain_error()
    if 'svd_convergence_speed' in algorithms_to_run:
        plot_svd_convergence_speed(plotter)
    if 'estimation_gain' in algorithms_to_run:
        plot_estimation_gain()
    if 'difference_images' in algorithms_to_run:
        difference_images()