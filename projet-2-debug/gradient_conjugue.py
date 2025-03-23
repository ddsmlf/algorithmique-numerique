import numpy as np

def conjgrad(A,b,x):
    """
    Solve the system of linear equations Ax = b using the Conjugate Gradient method.

    Parameters:
    A (numpy.ndarray): The symmetric positive-definite matrix A.
    b (numpy.ndarray): The right-hand side vector b.
    x (numpy.ndarray): The initial guess for the solution vector x.

    Returns:
    tuple: A tuple containing:
        - x (numpy.ndarray): The solution vector x.
        - tab (numpy.ndarray): An array containing the residuals at each iteration.

    Notes:
    The function iteratively improves the solution vector x until the residual is less than 1e-8 or the maximum number of iterations (1000) is reached.
    """
    r = b - np.dot(A,x)
    tab = np.zeros((1000,1))
    tabx = np.zeros((1000,len(x)))
    iteration = 0
    p = r
    rsold = np.dot(r.T,r)
    for i in range(1000):
        Ap = np.dot(A,p)
        alpha = rsold / np.dot(p.T,Ap)
        x = x + alpha*p
        tabx[i] = x
        iteration = iteration+1
        r = r - alpha*Ap
        rsnew = np.dot(r.T,r)
        tab[i] = rsnew
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x,tab,tabx,iteration


    
