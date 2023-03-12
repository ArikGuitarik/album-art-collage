import numpy as np
from skimage.transform import resize


class Image:
    """Image with RGB pixels of the shape (height, width, 3) with values in the range [0, 1]"""
    def __init__(self, pixels: np.ndarray):
        self._reject_invalid_params(pixels)
        self._pixels = pixels

    @staticmethod
    def _reject_invalid_params(pixels: np.ndarray):
        if len(pixels.shape) != 3 or pixels.shape[2] != 3:
            raise ValueError(f"Image should have shape (height, width, 3), but is {pixels.shape}.")
        min_pixel_value, max_pixel_value = np.min(pixels), np.max(pixels)
        if min_pixel_value < 0 or max_pixel_value > 1:
            raise ValueError(f"Pixel values should be in [0, 1], but are in [{min_pixel_value, max_pixel_value}]")

    def get_pixels(self, shape: tuple[int, int] = None) -> np.ndarray:
        """Get pixels of the image, resized to shape=(height, width) if shape is not None"""
        if shape is None:
            return self._pixels
        return resize(self._pixels, shape)
