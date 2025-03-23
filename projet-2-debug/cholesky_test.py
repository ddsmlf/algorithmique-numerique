import unittest
import numpy as np
from cholesky import cholesky_complet, cholesky_incomplet
from numpy.linalg import cholesky
from matrix_generation import generate_symmetric_positive_definite_matrix, generate_sparse_symmetric_positive_definite_matrix
from equation_chaleur import precond
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class TestCholesky(unittest.TestCase):

    def test_cholesky_complet(self):
        A = generate_symmetric_positive_definite_matrix(50)
        L = cholesky_complet(A)
        np.testing.assert_array_almost_equal(np.dot(L, L.T), A, decimal=5)

    def test_invalid_matrix(self):
        A = np.array([[1, 2], [2, 1]])
        with self.assertRaises(ValueError):
            cholesky_complet(A)

    def test_incomplete_cholesky(self):
        A = generate_sparse_symmetric_positive_definite_matrix(50, 40)
        L = cholesky_incomplet(A)
        assert precond(A,L)==1
       
def compare_cholesky_methods(min_size, max_size, step, *funcs, labels=None):
    """
    Compare the performance and precision of different Cholesky decomposition methods.
    Parameters:
    min_size (int): The minimum size of the matrices to test.
    max_size (int): The maximum size of the matrices to test.
    step (int): The step size for the range of matrix sizes.
    *funcs (callable): Variable number of functions that perform Cholesky decomposition.
    labels (list of str, optional): Labels for the functions. If None, default labels will be used.
    Returns:
    None
    This function generates symmetric positive definite matrices of varying sizes and applies
    the provided Cholesky decomposition functions to them. It measures the execution time and
    precision of each method and plots the results.
    The precision is calculated as the relative error in the 1-norm between the original matrix
    and the product of the Cholesky factor with its transpose, normalized by the machine epsilon.
    The results are displayed in two subplots:
    1. Execution time vs. matrix size.
    2. Precision vs. matrix size.
    """
    if labels is None:
        labels = [f'func{i+1}' for i in range(len(funcs))]
    
    sizes = list(range(min_size, max_size + 1, step))
    times = {label: [] for label in labels}
    precisions = {label: [] for label in labels}
    times_sparse = {label: [] for label in labels}
    precisions_sparse = {label: [] for label in labels}

    for size in tqdm(sizes, desc="Matrix sizes"):
        times_list = {label: [] for label in labels}
        times_sparse_list = {label: [] for label in labels}
        precisions_list = {label: [] for label in labels}
        precisions_sparse_list = {label: [] for label in labels}

        for _ in range(10):
            B = generate_sparse_symmetric_positive_definite_matrix(size, 70)
            A = generate_symmetric_positive_definite_matrix(size)       
            for func, label in zip(funcs, labels):
                start_time = time.time()
                L = func(A)
                times_list[label].append(time.time() - start_time)
                start_time_sparse = time.time()
                L_sparse = func(B)
                times_sparse_list[label].append(time.time() - start_time_sparse)

                ex = np.dot(L, L.T)
                ex_sparse = np.dot(L_sparse, L_sparse.T)
                relative_error_sparse = np.linalg.norm(ex_sparse - B, 1) / (np.linalg.norm(B, 1) * size * np.finfo(float).eps)
                precisions_sparse_list[label].append(relative_error_sparse)
                relative_error = np.linalg.norm(ex - A, 1) / (np.linalg.norm(A, 1) * size * np.finfo(float).eps)
                precisions_list[label].append(relative_error)

        for label in labels:
            times[label].append(np.mean(times_list[label]))
            precisions[label].append(np.mean(precisions_list[label]))
            times_sparse[label].append(np.mean(times_sparse_list[label]))
            precisions_sparse[label].append(np.mean(precisions_sparse_list[label]))

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    for label in labels:
        plt.plot(sizes, times[label], label=label)
    plt.xlabel('Taille de la matrice')
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Comparaison des temps d'exécution (Dense)")
    plt.legend()

    plt.subplot(2, 2, 2)
    for label in labels:
        plt.plot(sizes, precisions[label], label=label)
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Précision ')
    plt.title('Comparaison des précisions (Dense)')
    plt.legend()

    plt.subplot(2, 2, 3)
    for label in labels:
        plt.plot(sizes, times_sparse[label], label=label)
    plt.xlabel('Taille de la matrice')
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Comparaison des temps d'exécution (Sparse)")
    plt.legend()

    plt.subplot(2, 2, 4)
    for label in labels:
        plt.plot(sizes, precisions_sparse[label], label=label)
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Précision echelle logarithmique')
    plt.title('Comparaison des précisions (Sparse)')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_cholesky_methods(4, 500, 1, cholesky, cholesky_complet,cholesky_incomplet, labels=['numpy', 'complet', 'incomplet'])
    unittest.main()

