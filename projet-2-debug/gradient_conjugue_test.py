import unittest
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import time
from gradient_conjugue import conjgrad
from matrix_generation import generate_symmetric_positive_definite_matrix   
from scipy.sparse.linalg import cg


class TestConjGrad(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)  # For reproducibility

    def test_small_matrix(self):
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        x0 = np.zeros(2)
        x, _,_,_ = conjgrad(A, b, x0)
        np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_large_matrix(self):
        n = 10
        A = generate_symmetric_positive_definite_matrix(n)
        b = np.random.randint(0, 10, n)
        x0 = np.zeros(n)
        x, _,_,_ = conjgrad(A, b, x0)
        np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_random_matrix(self):
        n = 5
        A = generate_symmetric_positive_definite_matrix(n)
        b = np.random.randint(0, 10, n)
        x0 = np.zeros(n)
        x, _,_,_ = conjgrad(A, b, x0)
        np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_identity_matrix(self):
        n = 5
        A = np.eye(n)
        b = np.random.randint(0, 10, n)
        x0 = np.zeros(n)
        x, _,_,_ = conjgrad(A, b, x0)
        np.testing.assert_array_almost_equal(x, b, decimal=5)

def test_performance(min_size=4, max_size=500, step=1):
    """
    Tests the performance of the conjugate gradient method on symmetric positive definite matrices of varying sizes.

    Parameters:
    min_size (int): The minimum size of the matrix to test. Default is 10.
    max_size (int): The maximum size of the matrix to test. Default is 500.
    step (int): The step size for increasing the matrix size. Default is 1.

    This function generates symmetric positive definite matrices of sizes ranging from min_size to max_size, 
    with increments of step. For each matrix size, it runs the conjugate gradient method 10 times, 
    recording the execution time and relative error for each trial. 
    It then calculates the average execution time and relative error for each matrix size and plots these values.

    The function produces two plots:
    1. Execution Time vs Matrix Size
    2. Relative Error vs Matrix Size

    The plots are displayed in a single figure with two subplots.
    """

    sizes = list(range(min_size, max_size + 1, step))
    times_ours = []
    relative_errors_ours = []
    times_scipy = []
    relative_errors_scipy = []

    for n in sizes:
        print(f"Testing matrix size {n}...")
        total_time_ours = 0
        total_relative_error_ours = 0
        total_time_scipy = 0
        total_relative_error_scipy = 0
        num_trials = 10

        for _ in range(num_trials):
            A = generate_symmetric_positive_definite_matrix(n)
            b = np.random.randint(0, 10, n)
            x0 = np.zeros(n)

            # Our implementation
            start_time = time.time()
            x, _,_,_ = conjgrad(A, b, x0)
            end_time = time.time()

            execution_time_ours = end_time - start_time
            relative_error_ours = np.linalg.norm(np.dot(A, x) - b) / (np.linalg.norm(b) + np.finfo(float).eps)

            total_time_ours += execution_time_ours
            total_relative_error_ours += relative_error_ours

            # SciPy implementation
            start_time = time.time()
            x_scipy, _ = cg(A, b, x0=x0)
            end_time = time.time()

            execution_time_scipy = end_time - start_time
            relative_error_scipy = np.linalg.norm(np.dot(A, x_scipy) - b) / (np.linalg.norm(b) + np.finfo(float).eps)

            total_time_scipy += execution_time_scipy
            total_relative_error_scipy += relative_error_scipy

        avg_time_ours = total_time_ours / num_trials
        avg_relative_error_ours = total_relative_error_ours / num_trials
        avg_time_scipy = total_time_scipy / num_trials
        avg_relative_error_scipy = total_relative_error_scipy / num_trials

        times_ours.append(avg_time_ours)
        relative_errors_ours.append(avg_relative_error_ours)
        times_scipy.append(avg_time_scipy)
        relative_errors_scipy.append(avg_relative_error_scipy)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times_ours, label='Our Implementation')
    plt.plot(sizes, times_scipy, label='SciPy Implementation')
    plt.title('Execution Time vs Matrix Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sizes, relative_errors_ours, label='Our Implementation')
    plt.plot(sizes, relative_errors_scipy, label='SciPy Implementation')
    plt.yscale('log')
    plt.title('Relative Error vs Matrix Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Relative Error (log scale)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_residue(N):
    for i in range(1, 10):
        A = generate_symmetric_positive_definite_matrix(N)
        b = np.random.randint(0, 10, N)
        _, tab, _, iteration = conjgrad(A,b,np.array(np.zeros(N)))
        plt.yscale("log")
        plt.plot(range(iteration), tab[:iteration])
        plt.xlabel('Iteration')
        plt.ylabel('Residue echelle logarithmique')
        plt.title('Residue Gradient Conjugue')
    plt.show()

if __name__ == '__main__':
    test_performance()
    print_residue(10)
    unittest.main()
