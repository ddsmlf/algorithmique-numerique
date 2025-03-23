import numpy as np
import matplotlib.pyplot as plt
from SVD import SVD
import cupy as cp

class ImageCompressor:
    def __init__(self, img_path, gpu=False):
        """
        Initialize the ImageCompressor with an image.

        Parameters:
        img_path (str): Path to the image file.
        """
        self.gpu = gpu
        self.xp = cp if gpu else np
        self.img = plt.imread(img_path)
        self.n, self.m, self.colors = self.img.shape
        self.img_resized = self._resize_image()
        self.k = min(self.n, self.m)


        self.cache = {
            "manual_separate_channel_0": {},
            "manual_separate_channel_1": {},
            "manual_separate_channel_2": {},
            "numpy_separate_channel_0": {},
            "numpy_separate_channel_1": {},
            "numpy_separate_channel_2": {},
            "manual_merged": {},
            "numpy_merged": {}
        }

    def __asnumpy(self, X):
        """Converts an array to NumPy if it is on the GPU."""
        if self.gpu:
            return cp.asnumpy(X)
        else:
            return X
        
    def update_image(self, img_array):
        """
        Update the image with a new image array and clear the cache.

        Parameters:
        img_array (numpy.ndarray): New image array.
        """
        self.img = img_array
        self.n, self.m, self.colors = self.img.shape
        self.img_resized = self._resize_image()
        self.k = min(self.n, self.m)

        self.cache = {
            "manual_separate_channel_0": {},
            "manual_separate_channel_1": {},
            "manual_separate_channel_2": {},
            "numpy_separate_channel_0": {},
            "numpy_separate_channel_1": {},
            "numpy_separate_channel_2": {},
            "manual_merged": {},
            "numpy_merged": {}
        }

    def display_comparison(self, r):
        """
        Display the original and compressed images with all methods.

        Parameters:
        r (int): Rank for compression.
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Compression au rang {r}")

        axs[0, 0].imshow(self.img)
        axs[0, 0].set_title("Image originale")

        img_compressed_manual = self.compress_image_3_channels(r, method="manual")
        axs[0, 1].imshow(np.clip(self.__asnumpy(img_compressed_manual), 0, 1))
        axs[0, 1].set_title("Compression manuelle")

        img_compressed_numpy = self.compress_image_3_channels(r, method="numpy")
        axs[0, 2].imshow(np.clip(self.__asnumpy(img_compressed_numpy), 0, 1))
        axs[0, 2].set_title("Compression numpy")

        img_compressed_merged_manual = self.compress_image_3_channels_merged_channels(r, method="manual")
        axs[1, 0].imshow(np.clip(self.__asnumpy(img_compressed_merged_manual), 0, 1))
        axs[1, 0].set_title("Compression manuelle (canaux fusionnés)")

        img_compressed_merged_numpy = self.compress_image_3_channels_merged_channels(r, method="numpy")
        axs[1, 1].imshow(np.clip(self.__asnumpy(img_compressed_merged_numpy), 0, 1))
        axs[1, 1].set_title("Compression numpy (canaux fusionnés)")

        axs[1, 2].axis('off')

        plt.show()

    def _resize_image(self):
        """
        Resize the image to a square shape with black padding.

        Returns:
        numpy.ndarray: Resized image array.
        """
        max_size = max(self.n, self.m)
        img_resized = self.xp.zeros((max_size, max_size, self.colors))
        img_resized[:self.n, :self.m, :] = self.xp.asarray(self.img)
        return img_resized
    
    def _get_cached_s(self, method, channel=None):
        """
        Retrieve the cached singular values matrix for a given method, rank and channel.

        Parameters:
        method (str): Compression method ("manual" or "numpy").
        channel (int): Channel number (0, 1 or 2 for RGB).

        Returns:
        
        """
        if channel is None:
            cache_key = f"{method}_merged"
        else:
            cache_key = f"{method}_separate_channel_{channel}"
        
        if cache_key not in self.cache:
            return None
        s = self.cache[cache_key].get("s", None)
        U = self.cache[cache_key].get("U", None)
        V = self.cache[cache_key].get("V", None)
        return s, U, V

    def _cache_s(self, method,s , U, V, channel=None):
        """
        Cache the singular values matrix for future use.

        Parameters:
        method (str): Compression method ("manual" or "numpy").
        s (numpy.ndarray): Singular values matrix.
        U (numpy.ndarray): Left singular vectors.
        V (numpy.ndarray): Right singular vectors.
        channel (int): Channel number (0, 1 or 2 for RGB).
        """
        if channel is None:
            cache_key = f"{method}_merged"
            self.cache[cache_key]["s"] = s
            self.cache[cache_key]["U"] = U
            self.cache[cache_key]["V"] = V
            return
        cache_key = f"{method}_separate_channel_{channel}"
        self.cache[cache_key]["s"] = s
        self.cache[cache_key]["U"] = U
        self.cache[cache_key]["V"] = V

    def compress_image_3_channels(self, r, method="manual"):
        img_compressed = self.xp.zeros_like(self.img_resized)
        r = min(r, self.k)

        for i in range(self.colors):
            s, U, V = self._get_cached_s(method, channel=i)

            if s is None: 
                if method == "manual":
                    svd = SVD(self.xp.asarray(self.img_resized[:, :, i]), gpu=self.gpu)
                    U, s, V = svd.apply_SVD()  
                else:
                    U, s, V = self.xp.linalg.svd(self.xp.asarray(self.img_resized[:, :, i]), full_matrices=False)

                self._cache_s(method, channel=i, s=s, U=U, V=V)

            S_reduced = self.xp.diag(s[:r])
            U_reduced = U[:, :r]
            V_reduced = V.T[:, :r]  

            img_compressed[:, :, i] = U_reduced @ S_reduced @ V_reduced.T

        return self._recrop_image(img_compressed)
    

    def compress_image_3_channels_merged_channels(self, r, method="manual"):
        """
        Compress the image using r singular values for all RGB channels merged.
        """
        img_merged = self.xp.vstack([self.img_resized[:, :, i] for i in range(self.colors)])

        h, w = img_merged.shape
        r = min(r, self.k)

        if h != w:
            size = max(h, w)
            img_merged_square = self.xp.zeros((size, size))
            img_merged_square[:h, :w] = img_merged 
        else:
            img_merged_square = img_merged

        s, U, V = self._get_cached_s(method)

        if s is None:
            if method == "manual":
                svd = SVD(img_merged_square, gpu=self.gpu)
                U, s, V = svd.apply_SVD()
            else:
                U, s, V = self.xp.linalg.svd(img_merged_square, full_matrices=False)

            self._cache_s(method, s=s, U=U, V=V)

        s = s[:r]
        S = self.xp.zeros((r, r))
        self.xp.fill_diagonal(S, s)

        img_compressed_merged = self.xp.dot(self.xp.asarray(U[:, :r]), self.xp.dot(S, self.xp.asarray(V[:r, :])))
        img_compressed = self.xp.zeros_like(self.img_resized)
        img_compressed[:, :, 0] = img_compressed_merged[:self.img_resized.shape[0], :self.img_resized.shape[1]]
        img_compressed[:, :, 1] = img_compressed_merged[self.img_resized.shape[0]:2*self.img_resized.shape[0], :self.img_resized.shape[1]]
        img_compressed[:, :, 2] = img_compressed_merged[2*self.img_resized.shape[0]:3*self.img_resized.shape[0], :self.img_resized.shape[1]]

        img_compressed = img_compressed[:self.n, :self.m, :]

        return self.__asnumpy(img_compressed)

    def _recrop_image(self, img):
        """
        Recrop the image to its original size.

        Parameters:
        img (numpy.ndarray): Image array to recrop.

        Returns:
        numpy.ndarray: Recropped image array.
        """
        return img[:self.n, :self.m, :]
    
