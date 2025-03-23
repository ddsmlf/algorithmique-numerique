import numpy as np
import matplotlib.pyplot as plt
import matrix_generation as mg
import gradient_conjugue as gc
from matrix_generation import M_c
from cholesky import cholesky_incomplet, cholesky_complet
import time



def pivot_gausse_inf(L,b):
    #on
    n=len(L)
    X=np.zeros(n)
    for i in range(n):
        y=b[i]
        for j in range(i):
            y-=L[i][j]*X[j]
        if L[i][i] == 0:
            raise ValueError("Zero diagonal element encountered in pivot_gausse_inf")
        X[i]=y/L[i][i]
    A=np.dot(L,X)
    j=0
    maximum=0
    for i in range(len(A)):
        if(abs(b[i]-A[i])>1e-2):
            j+=1
            if(maximum<abs(b[i]-A[i])):
                maximum=abs(b[i]-A[i])
    print(f"la valeur d'erreur est {j} sachant que N={len(A)} et le grand ecart est {maximum}")
    return X
 
    


def pivot_gausse_sup(L,b):
    #on
    n=len(L)
    X=np.zeros(n)
    for i in range(n-1,-1,-1):
        y=b[i]
        for j in range(i+1,n):
            y-=L[i][j]*X[j]
        X[i]=y/L[i][i]
    A=np.dot(L,X)
    j=0
    maximum=0
    for i in range(len(A)):
        if(abs(b[i]-A[i])>1e-2):
            j+=1
            if(maximum<abs(b[i]-A[i])):
                maximum=abs(b[i]-A[i])
    print(f"la valeur d'erreur est {j} sachant que N={len(A)} et le grand ecart est {maximum}")
    return X


def precond(A, T):
    """
    Evaluate the quality of the preconditioner

    Parameters:
    A (numpy.ndarray): A symmetric positive definite matrix of shape (n, n).
    T (numpy.ndarray): T a lower triangular matrix of shape (n, n).

    Returns:
    bool: True if the preconditioner is of good quality, False otherwise.
    """
    a = np.linalg.cond(A)
    T = T.reshape(A.shape)
    M_1=np.linalg.inv(T @ T.T)
    b = np.linalg.cond(np.dot(M_1,A))
    return b < a

def simulation_cholesky(N, F, complet=True):
    """
    Simule la diffusion de chaleur dans une grille NxN.

    Parameters:
    N (int): La dimension de la grille (NxN).
    F (numpy.ndarray): Le flux de chaleur initial, de taille (N, N).

    Returns:
    T (numpy.ndarray): La température après diffusion, de taille
    """
    F = F.flatten()
    T = np.zeros(N*N)
    A = mg.M_c(N)
    if complet:
        L = cholesky_complet(A)
    else:
        #print("Incomplet")
        L = cholesky_incomplet(A)
        
        # Ensure the preconditioner is applied correctly
        # if not precond(A, L):
        #     raise ValueError("The preconditioner is not of good quality.")
    Y = pivot_gausse_inf(L, F)
    T = pivot_gausse_sup(L.T, Y)

    return T

def simulation_grad(N, F):
    """
    Simule la diffusion de chaleur dans une grille NxN.

    Parameters:
    N (int): La dimension de la grille (NxN).
    F (numpy.ndarray): Le flux de chaleur initial, de taille (N, N).

    Returns:
    T (numpy.ndarray): La température après diffusion, de taille
    tab (list): La liste des températures à chaque itération.
    tabx (list): La liste des températures à chaque itération.
    iteration (int): Le nombre d'itérations.
    """
    F.shape = (N*N)
    T = np.zeros(N*N)
    A = mg.M_c(N)
    T, tab, tabx, iteration = gc.conjgrad(A, F, T)

    return T, tab, tabx, iteration
