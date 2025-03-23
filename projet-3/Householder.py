import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

class Householder:
    def __init__(self, U, V, gpu=False):
        """
        Initializes the Householder transformation.

        Args:
            U (ndarray): Starting vector.
            V (ndarray): Destination vector.
            gpu (bool): If True, uses CuPy for the GPU calculations, else uses NumPy (False by default).
        
        Raises:
            ValueError: If U and V are not valid to build a Householder matrix.
        """
        self.gpu = gpu
        self.xp = cp if self.gpu and cp else np
        self.U = self.xp.asarray(U)
        self.V = self.xp.asarray(V)

        self._norm_U = self.xp.linalg.norm(self.U)
        self._norm_V = self.xp.linalg.norm(self.V)
        if not self.__verification_on_UV()[0]:
            raise ValueError("U and V are not valid to build a Householder matrix: " + self.__verification_on_UV()[1])

        self.N = self.__calculate_N()
        self._NT = self.N.T if self.N is not None else None
        self.H = self.__calculate_householder_matrix()

    def __asnumpy(self, X):
        """Converts an array to NumPy if it is on the GPU."""
        if self.gpu:
            return cp.asnumpy(X)
        else:
            return X

    def __verification_on_UV(self):
        """
        Verifies if U and V have the same dimensions and if their norms are equal.
        """
        if (self.U.shape != self.V.shape):
            return False, "U and V should have the same dimension."
        if not self.xp.isclose(self._norm_U, self._norm_V): 
            return False, "U and V vectors should have the same norm."
        return True, ""

    def __calculate_N(self):
        """
        Computes the vector N, which is the normalized difference between U and V.
        """
        W = self.U - self.V
        norm_W = self.xp.linalg.norm(W) 
        if self.xp.isclose(norm_W, 0):  
            return None
        return W / norm_W

    def __calculate_householder_matrix(self):
        """
        Computes the Householder matrix used for orthogonal transformations.
        """
        I = self.xp.eye(len(self.U))
        if self.N is None:
            return I
        N_outer = self.N @ self.N.T
        return I - 2 * N_outer

    def apply_transformation(self, X):
        """
        Applies the Householder transformation to a vector X in an optimized way.

        Args:
            X (ndarray): Vector to transform.

        Returns:
            ndarray: Transformed vector.
        """
        if self.N is None:
            return X
        Z = self.xp.asarray(X)
        return Z - 2 * self.N @ (self.N.T @ Z)

    def apply_transformation_to_matrix(self, X, row=False):
        """
        Applies the Householder transformation to a matrix X in an optimized way.

        Args:
            X (ndarray): Matrix to transform.
            T (bool): If True, applies the transformation to rows of X, else to columns.

        Returns:
            ndarray: Transformed matrix.
        """
        Z = self.xp.asarray(X)
        if self.N is None:
            return Z
            
        N_outer = self.N @ self.N.T
        if not row:
            return Z - 2 * N_outer @ Z  # Applies H to columns of X
        return Z - 2 * Z @ N_outer  # Applies H to rows of X

    def display_matrices(self):
        """
        Prints U, V et N vectors, and the Householder matrix H.
        """
        print("\nVector U :\n", self.__asnumpy(self.U)) 
        print("Vector V :\n", self.__asnumpy(self.V)) 
        if self.N is not None:
            print("Vector N :\n", self.__asnumpy(self.N)) 
        else:
            print("N not defined because U = V.")
        print("Householder matrix H :\n", self.__asnumpy(self.H)) 

    def visualize_transformation(self):
        """
        Prints a graph showing U,V and H(U).
        Only works in 2D or 3D.
        """
        dim = len(self.U)
        if dim > 3:
            print("Visualisation only available in 2D or 3D.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if dim == 3 else fig.add_subplot(111)
        
        Ux, Uy, Uz = (self.__asnumpy(self.U.flatten()).tolist() + [0])[:3] 
        Vx, Vy, Vz = (self.__asnumpy(self.V.flatten()).tolist() + [0])[:3] 
        HUx, HUy, HUz = (self.__asnumpy(self.apply_transformation(self.U).flatten()).tolist() + [0])[:3] 

        origin = [0, 0, 0]
        
        if dim == 3:
            ax.quiver(*origin[:dim], Ux, Uy, Uz, color='b', label="U", linewidth=2)
            ax.quiver(*origin[:dim], Vx, Vy, Vz, color='r', label="V", linewidth=2)
            ax.quiver(*origin[:dim], HUx, HUy, HUz, color='g', linestyle='dashed', label="H(U)", linewidth=2)
        else:
            ax.quiver(origin[0], origin[1], Ux, Uy, angles='xy', scale_units='xy', scale=1, color='b', label="U", linewidth=2)
            ax.quiver(origin[0], origin[1], Vx, Vy, angles='xy', scale_units='xy', scale=1, color='r', label="V", linewidth=2)
            ax.quiver(origin[0], origin[1], HUx, HUy, angles='xy', scale_units='xy', scale=1, color='g', linestyle='dashed', label="H(U)", linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if dim == 3:
            ax.set_zlabel('Z')

        ax.legend()
        plt.title("Transformation de Householder : U → V")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid()
        plt.show()