import numpy as np
import math

def euclidian_distance(M1: np.matrix, M2: np.matrix) -> float:
    """
    Computes the euclidian distance between `M1` and `M2`.
    Parameters : 
    `M1` a matrix.
    `M2` a matrix with the same shape as `M1`.
    Returns :
    A float representing the distance between matrix M1 and M2.
    POSTCOND : distance is a positive number.
    """
    if np.shape(M1) != np.shape(M2):
        raise ValueError("Matrices' shapes do not match.")
    h, w, d = np.shape(M1)
    distance = 0
    for color_ch in range(d):
        for x in range(h):
            for y in range(w):
                distance += abs(M1[x, y, color_ch]**2 - M2[x, y, color_ch]**2)

    return math.sqrt(distance)

def manhattan_distance(M1: np.matrix, M2: np.matrix) -> float:
    """
    Computes the manhattan distance between `M1` and `M2`.
    Parameters : 
    `M1` a matrix.
    `M2` a matrix with the same shape as `M1`.
    Returns :
    A float representing the distance between matrix M1 and M2.
    POSTCOND : distance is a positive number.
    """
    h, w, d = np.shape(M1)
    distance = 0
    for color_ch in range(d):
        for x in range(h):
            for y in range(w):
                distance += abs(M1[x, y, color_ch] - M2[x, y, color_ch])

    return distance 

def generate_random_UV(size):
    """
    Generates two random vectors U and V of size `size`, having the same random norm.
    
    Args:
        size (int): Size of the vectors U and V.
    
    Returns:
        tuple: Two vectors U and V of size `size` having the same random norm.
    """
    U = np.random.randn(size, 1)
    norm_U = np.random.uniform(1, 10)
    U = U / np.linalg.norm(U) * norm_U
    V = np.random.randn(size, 1)
    V = V / np.linalg.norm(V) * norm_U

    return U, V


def matrix_multiply(A, B):
    """
    Performs a usual matrix-matrix product of A and B.

    Args:
        A (ndarray): Matrix of size (n, p)
        B (ndarray): Matrix of size (p, m)

    Returns:
        ndarray: Resulting matrix of size (n, m)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions are not compatible for multiplication.")
    
    n, p = A.shape
    p2, m = B.shape
    result = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i, j] += A[i, k] * B[k, j]
    
    return result
