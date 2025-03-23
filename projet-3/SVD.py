import numpy as np
import cupy as cp
import tqdm

from Householder import Householder
from Bidiagonalization import Bidiagonalization

class SVD:
    def __init__(self, A, max_iter=1000, tol=1e-6, invariant_tol=1e-5, gpu=False):
        """
        Initialise la classe pour calculer la décomposition SVD d'une matrice A.

        Parameters:
        -----------
        A : ndarray
            Matrice à décomposer.
        max_iter : int, optional
            Nombre maximal d'itérations pour la convergence. Default is 3000.
        tol : float, optional
            Seuil de tolérance pour la convergence de S. Default is 1e-6.
        invariant_tol : float, optional
            Seuil de tolérance pour || U S V^T - BD ||. Default is 1e-5.
        gpu : bool, optional
            Si True, utilise le GPU (CuPy), sinon utilise le CPU (NumPy). Default is False.
        """
        self.gpu = gpu
        self.xp = cp if self.gpu else np
        self.A = self.xp.asarray(A)
        self.max_iter = max_iter
        self.tol = tol
        self.invariant_tol = invariant_tol

        bidiag = Bidiagonalization(self.A, tol=self.tol, gpu=self.gpu)
        self.U = self.xp.eye(self.A.shape[0])
        self.V = self.xp.eye(self.A.shape[1])
        self.Qleft, self.s, self.Qright = bidiag.compute()
        self.BD = self.xp.asarray(self.s).copy()

    
    def householder_qr(self, B):
        """
        Applique la décomposition QR optimisée pour une matrice bidiagonale
        en utilisant des transformations de Householder.

        Parameters:
        -----------
        B : ndarray
            Matrice bidiagonale.

        Returns:
        --------
        Q : ndarray
            Matrice orthogonale.
        R : ndarray
            Matrice triangulaire supérieure.
        """
        m, n = B.shape
        Q = self.xp.eye(m)
        R = self.xp.array(B, copy=True)

        for i in range(min(m, n)):
            x = R[i:, i]
            norm_x = self.xp.linalg.norm(x)
            if norm_x == 0:
                continue

            e1 = self.xp.zeros_like(x)
            e1[0] = norm_x

            householder = Householder(x.reshape(-1, 1), e1.reshape(-1, 1), gpu=self.gpu)
            R[i:, i:] = householder.apply_transformation_to_matrix(R[i:, i:])
            Q[:, i:] = householder.apply_transformation_to_matrix(Q[:, i:], True)

        return Q, R

    def givens_rotation(self, B):
        """
        QR decomposition for a bidiagonal matrix using Givens rotations.
        
        Parameters:
        -----------
        B : ndarray
            Bidiagonal matrix (lower or upper).
            
        Returns:
        --------
        Q : ndarray
            Orthogonal matrix.
        R : ndarray
            Upper triangular matrix.
        """
        m, n = B.shape
        R = self.xp.array(B, copy=True)
        Q = self.xp.eye(m, dtype=R.dtype)

        for i in range(min(m, n) - 1):
            if self.xp.abs(R[i+1, i]) < 1e-15:
                continue

            a = R[i, i].item()
            b = R[i+1, i].item()
            norm = self.xp.sqrt(a**2 + b**2)
            c = a / norm
            s = b / norm

            G = self.xp.array([[c, s], [-s, c]], dtype=R.dtype)
            rows = R[[i, i+1], i:]  
            rotated_rows = G @ rows
            R[i, i:] = rotated_rows[0]
            R[i+1, i:] = rotated_rows[1]

            cols = Q[:, [i, i+1]]
            rotated_cols = cols @ G.T
            Q[:, i] = rotated_cols[:, 0]
            Q[:, i+1] = rotated_cols[:, 1]

        return Q, R
    
    def apply_SVD(self, plot_convergence=False, qr_method="givens_rotation"):
        """
        Applies the Singular Value Decomposition (SVD) algorithm to the matrix.

        Parameters:
        -----------
        plot_convergence : bool, optional
            If True, returns the convergence plot data. Default is False.
        qr_method : str, optional
            The QR decomposition method to use. Options are "givens_rotation" or any method supported by xp.linalg.qr. Default is "givens_rotation".

        Returns:
        --------
        U : ndarray
            The left singular vectors.
        S : ndarray
            The singular values in a diagonal matrix.
        V : ndarray
            The right singular vectors.
        off_diag_norms : list, optional
            The norms of the off-diagonal elements during the iterations, returned if plot_convergence is True.
        non_negligible_off_diag_counts : list, optional
            The counts of non-negligible off-diagonal elements during the iterations, returned if plot_convergence is True.

        Raises:
        -------
        ValueError
            If the invariant USV - BD is not respected.
        """
        off_diag_norms = []
        non_negligible_off_diag_counts = []
        for _ in tqdm.tqdm(range(self.max_iter), desc="SVD Convergence"):
            if qr_method == "givens_rotation":
                Q1, R1 = self.givens_rotation(self.s.T)
                Q2, R2 = self.givens_rotation(R1.T)
            else:
                Q1, R1 = self.xp.linalg.qr(self.s.T)
                Q2, R2 = self.xp.linalg.qr(R1.T)

            self.s = R2
            self.U = self.U @ Q2
            self.V = Q1.T @ self.V

            off_diag_norm = self.xp.linalg.norm(self.s - self.xp.diag(self.xp.diag(self.s)))
            off_diag_norms.append(off_diag_norm)

            non_negligible_off_diag_count = self.xp.sum(self.xp.abs(self.s - self.xp.diag(self.xp.diag(self.s))) > self.tol)
            non_negligible_off_diag_counts.append(non_negligible_off_diag_count)

            if not np.allclose(np.dot(self.U, np.dot(self.s, self.V)), self.BD, atol=self.tol):
                raise ValueError(f"Invariant non respecté USV - BD = {np.linalg.norm(np.dot(self.U, np.dot(self.s, self.V)) - self.BD)}")

            if off_diag_norm < self.tol:
                break

        self.U = self.Qleft @ self.U
        self.V = self.V @ self.Qright
        self.S = self.xp.diag(self.s)
        
        singular_values = self.xp.abs(self.S)
        sorted_indices = self.xp.argsort(singular_values)[::-1]

        self.S = singular_values[sorted_indices]
        self.U = self.U[:, sorted_indices]
        self.V = self.V[sorted_indices, :]

        if plot_convergence:
            return self.U, self.s, self.V, off_diag_norms, non_negligible_off_diag_counts
        return self.U, self.S, self.V