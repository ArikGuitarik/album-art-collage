from functools import lru_cache
import numpy as np
from skimage.transform import resize
from grid import SquareGrid


class Image:
    """Image with RGB pixels of the shape (height, width, 3) with uint8 values in the range [0, 255]"""

    def __init__(self, pixels: np.ndarray):
        self._reject_invalid_params(pixels)
        self._pixels = pixels

    @staticmethod
    def _reject_invalid_params(pixels: np.ndarray):
        if len(pixels.shape) != 3 or pixels.shape[2] != 3:
            raise ValueError(f"Image should have shape (height, width, 3), but is {pixels.shape}.")
        if pixels.dtype != np.uint8:
            raise TypeError(f"Sub-pixel values should be encoded as uint8 with values in [0, 255], not {pixels.dtype}.")

    @lru_cache(maxsize=None)
    def get_pixels(self, shape: tuple[int, int] = None) -> np.ndarray:
        """Get pixels of the image, resized to shape=(height, width) if shape is not None"""
        if shape is None:
            return self._pixels
        resized = resize(self._pixels, shape)
        return (255 * resized).astype(np.uint8)


class Collage:
    def __init__(self, img_grid: SquareGrid[Image], img_shape: tuple[int, int]):
        self.img_grid = img_grid
        self.img_shape = img_shape
        self._pixels = None

    def render(self) -> Image:
        self._init_pixels()
        for row, col in self.img_grid.get_all_coordinates():
            self._place_image(row, col)
        return Image(self._pixels)

    def _init_pixels(self):
        grid_height, grid_width = self.img_grid.shape
        img_height, img_width = self.img_shape
        collage_height, collage_width = grid_height * img_height, grid_width * img_width
        self._pixels = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

    def _place_image(self, row: int, col: int):
        img_height, img_width = self.img_shape
        img_pixels = self.img_grid.get(row, col).get_pixels(self.img_shape)
        x, y = self.get_top_left_pixel_coordinates(row, col)
        self._pixels[y: y + img_height, x:x + img_width] = img_pixels

    def get_top_left_pixel_coordinates(self, row: int, col: int) -> tuple[int, int]:
        self._reject_if_out_of_bounds(row, col)
        img_height, img_width = self.img_shape
        return col * img_width, row * img_height

    def get_grid_coordinates(self, x: int, y: int) -> tuple[int, int]:
        img_height, img_width = self.img_shape
        row, col = y // img_height, x // img_width
        self._reject_if_out_of_bounds(row, col)
        return row, col

    def _reject_if_out_of_bounds(self, row: int, col: int):
        if not (0 <= row < self.img_grid.shape[0] and 0 <= col < self.img_grid.shape[1]):
            raise IndexError(f"Invalid coordinates {row, col} for grid of shape {self.img_grid.shape}")
