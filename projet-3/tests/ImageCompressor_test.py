import unittest
import numpy as np
from ImageCompressor import ImageCompressor

class TestImageCompressor(unittest.TestCase):
    def setUp(self):
        """
        Initializes ImageCompressor with example image.
        """
        self.compressor = ImageCompressor("exemple.png")
        self.img = self.compressor.img
        self.method = "numpy"
        
    def test_compress_image_3_channels(self):
        """
        Tests image compression while preserving all three color channels separately.
        """
        k = 20
        img_compressed = self.compressor.compress_image_3_channels(k, method=self.method)
        self.assertEqual(img_compressed.shape, self.img.shape)

    def test_compress_image_3_channels_with_color_channel_marged(self):
        """
        Tests image compression with merged color channels.
        """
        k = 20
        img_compressed = self.compressor.compress_image_3_channels_merged_channels(k, method=self.method)
        self.assertEqual(img_compressed.shape, self.img.shape)
        
if __name__ == "__main__":
    unittest.main()
