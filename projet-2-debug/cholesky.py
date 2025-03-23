import numpy as np
from math import sqrt
from matrix_generation_test import is_sdp, generate_sparse_symmetric_positive_definite_matrix, generate_symmetric_positive_definite_matrix
import math
import time



def cholesky_complet(A): 
    """
    Performs the Cholesky decomposition of a symmetric positive definite matrix.

    Parameters:
    A (numpy.ndarray): A symmetric positive definite matrix of shape (n, n).

    Returns:
    numpy.ndarray: A lower triangular matrix T such that A = T @ T.T.

    Raises:
    ValueError: If the input matrix A is not symmetric positive definite.

    Notes:
    - The input matrix A must be symmetric and positive definite.
    - The function does not check if A is symmetric positive definite. Uncomment the check if needed.
    """
    # if not is_sdp(A):
    #     raise ValueError("Matrix is not symmetric positive definite.")
    
    n, n = A.shape
    T = np.zeros((n, n))
    #i plays the role of the row    
    for i in range(n):
        #j plays the role of column. We always have j ≤ i < n
        for j in range(i + 1):
        #Thus, we compute L[i][j] in the formula, which justifies that L is a lower triangular matrix.
            value = A[j, i] - sum(T[i, k] * T[j, k] for k in range(j))
            if i==j:
                T[i, j] = math.sqrt(value)
            else:
                T[i, j] = value / T[j, j]
    return T



def cholesky_incomplet(A):
    """
    Performs an incomplete Cholesky decomposition of a symmetric positive definite matrix A.

    The incomplete Cholesky decomposition is a factorization of the form A = L * L.T,
    where L is a lower triangular matrix. This function does not perform a full decomposition
    and may be used as a preconditioner in iterative methods.

    Parameters:
    A (numpy.ndarray): A symmetric positive definite matrix of shape (n, n).

    Returns:
    numpy.ndarray: A lower triangular matrix L such that A ≈ L * L.T.

    Raises:
    ValueError: If the input matrix A is not symmetric positive definite.
    """
    # if not is_sdp(A):
    #     raise ValueError("Matrix is not symmetric positive definite.")
    n, n = A.shape
    L = np.zeros((n, n))
    D = np.zeros(n)

    for i in range(n):
        for j in range(i, n):
        
            #We only compute L[i][j] where A[i][j]!=0 ; this is the characterization of incomplete Cholesky.
            if(A[j][i]!=0):
                msum = A[i, j]
                for k in range(0,i):
                    msum-= L[i, k] * L[j, k]
                if i==j:
                    D[i] = sqrt(msum)
                    L[i, i] = D[i]
                if D[i] != 0:
                    L[j, i] = msum / D[i]
                else:
                    L[j, i] = 0
    return L



if __name__ == "__main__":
    print("Generating a random symmetric positive definite matrix...")
    A = generate_symmetric_positive_definite_matrix(1349)
    print(A)
    # print("Performing Cholesky decomposition...")
    # time1 = time.time()
    # L = cholesky_complet(A)
    # time2 = time.time()
    # print(f"Execution time: {time2 - time1:.4f} seconds")
