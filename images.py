import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
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

    @property
    def pixels(self) -> np.ndarray:
        return self._pixels

    def resize(self, shape: tuple[int, int]):
        self._pixels = resize(self._pixels, shape, preserve_range=True)


class ImageFromFile(Image):
    """Image that is loaded from a file in image_path and can restore its original resolution even after downsizing

    Purpose:
    - Handles importing an image from a file stored at location given by image_path
    - You can resize it for faster processing, then reload the original size if you need it for high-res rendering.
    """

    def __init__(self, image_path: str):
        self._image_path = image_path
        pixels = self._load_image_file(image_path)
        super().__init__(pixels)

    @staticmethod
    def _load_image_file(image_path: str):
        return imread(image_path)

    def reload_original(self):
        pixels = self._load_image_file(self._image_path)
        self._reject_invalid_params(pixels)
        self._pixels = pixels


class Collage:
    """A Collage merges individual Images into one, aligned as in the grid it comes in.

    All images will be resized to the same size in such a way that the shape of the resulting collage will be as close
    as possible to desired_shape.
    """

    def __init__(self, img_grid: SquareGrid[Image], desired_shape: tuple[int, int]):
        self.img_grid = img_grid
        self.img_shape = self._infer_img_shape(desired_shape)
        self._resize_images()
        self._pixels = None

    def _infer_img_shape(self, desired_collage_shape: tuple[int, int]):
        collage_height, collage_width = desired_collage_shape
        grid_height, grid_width = self.img_grid.shape
        img_height = int(np.round(collage_height / grid_height))
        img_width = int(np.round(collage_width / grid_height))
        return img_height, img_width

    def _resize_images(self):
        for img in tqdm(self.img_grid.to_list(), desc='Resizing'):
            img.resize(self.img_shape)

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
        img_pixels = self.img_grid.get(row, col).pixels
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
