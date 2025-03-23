import numpy as np
import cupy as cp

from Householder import Householder

class Bidiagonalization:
    def __init__(self, A, tol=1e-6, gpu=False):
        """
        Initializes the bidiagonalization algorithm.

        Args:
            A (ndarray): Matrix to bidiagonalize (n x m).
            tol (float): Reconstruction tolerance.
            gpu (bool): If True, uses the GPU (CuPy), else uses the CPU (NumPy).
        """
        self.gpu = gpu
        self.xp = cp if self.gpu else np
        self.A = self.xp.asarray(A)
        self.n, self.m = self.A.shape
        self.Qleft = self.xp.eye(self.n)
        self.Qright = self.xp.eye(self.m)
        self.BD = self.A.copy()
        self.tol = tol

    def compute(self):
        """
        Applies the bidiagonalization algorithm.
        Returns the  matrixes Qleft, BD, Qright where Qleft and Qright are orthogonal and BD is bidiagonal.
        """
        for i in range(min(self.n, self.m)):
            # ------------ Calculation of Q1 ------------
            U = self.xp.zeros((min(self.n, self.m), 1))
            U[i:] = self.BD[i:, i].copy().reshape(-1, 1)
            norm_U = self.xp.linalg.norm(U)
            
            V = self.xp.zeros_like(U)
            V[i] = norm_U
            Q1 = Householder(U, V, gpu=self.gpu)
            # ---------------------------------------

            self.Qleft = Q1.apply_transformation_to_matrix(self.Qleft, True) 
            self.BD = Q1.apply_transformation_to_matrix(self.BD)

            if i != max(self.n, self.m) - 2 :
                # ------------ Calculation of Q2 ------------
                U_row = self.xp.zeros((max(self.n, self.m), 1))
                U_row[i+1:] = self.BD[i, i+1:].copy().reshape(-1, 1)
                norm_U_row = self.xp.linalg.norm(U_row)
                
                if not self.xp.isclose(norm_U_row, 0):
                    V_row = self.xp.zeros_like(U_row)
                    V_row[i+1] = norm_U_row
                    Q2 = Householder(U_row, V_row, gpu=self.gpu)
                    # -----------------------------------

                    self.Qright = Q2.apply_transformation_to_matrix(self.Qright)  
                    self.BD = Q2.apply_transformation_to_matrix(self.BD, True)

            reconstruction = self.xp.dot(self.Qleft, self.xp.dot(self.BD, self.Qright))
            tol = self.tol * self.xp.linalg.norm(self.A)
            if not self.xp.allclose(reconstruction, self.A, atol=tol):
                diff = self.xp.linalg.norm(reconstruction - self.A)
                raise ValueError(f"Erreur {diff:.2e} > tolérance {tol:.2e}")

        return self.Qleft, self.BD, self.Qright
