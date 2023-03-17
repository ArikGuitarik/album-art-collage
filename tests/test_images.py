import pytest
import numpy as np

from images import Image, ImageFromFile, Collage
from grid import SquareGrid


def test_image_creation():
    example_value = 42
    pixels = np.ones((64, 64, 3), dtype=np.uint8) * example_value
    img = Image(pixels)
    assert img.pixels[0, 0, 0] == example_value
    assert img.pixels.shape == (64, 64, 3)


def test_image_creation_invalid_type():
    with pytest.raises(TypeError):
        Image(np.ones((64, 64, 3), dtype=np.float64))


def test_image_creation_invalid_shape():
    with pytest.raises(ValueError):
        Image(np.ones((10, 10, 1), dtype=np.uint8))


def test_image_resizing():
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[0:50, 0:50, :], pixels[0:50, 50:100, :] = 20, 40
    pixels[50:100, 0:50, :], pixels[50:100, 50:100, :] = 60, 80
    img = Image(pixels)
    img.resize((6, 6))
    downsized_pixels = img.pixels
    assert downsized_pixels.shape == (6, 6, 3)
    probe_pixels = [downsized_pixels[1, 1, 0], downsized_pixels[1, 4, 0],
                    downsized_pixels[4, 1, 0], downsized_pixels[4, 4, 0]]
    np.testing.assert_allclose(probe_pixels, [20, 40, 60, 80], atol=1)


def test_image_from_file():
    img = ImageFromFile("data/dummy_cover.jpg")
    assert img.pixels.shape == (100, 100, 3)
    assert all(img.pixels[0, 0] == [254, 0, 0])
    assert all(img.pixels[99, 99] == [0, 0, 254])
    img.resize((50, 50))
    assert img.pixels.shape == (50, 50, 3)
    img.reload_original()
    assert img.pixels.shape == (100, 100, 3)


@pytest.fixture
def example_img_grid():
    num_images, img_side_length = 4, 64
    imgs = [Image(np.ones((img_side_length, img_side_length, 3), dtype=np.uint8) * i * 25) for i in range(num_images)]
    return SquareGrid(imgs)


def test_collage_rendering(example_img_grid):
    collage = Collage(example_img_grid, (128, 256))
    assert collage.img_shape == (64, 128)
    result = collage.render().pixels
    assert result[0, 0, 0] == 0
    assert result[28, 208, 2] == 25
    assert result[90, 20, 2] == 50
    assert result[70, 140, 1] == 75
    assert result.shape == (128, 256, 3)


def test_collage_img_shape_matching(example_img_grid):
    collage = Collage(example_img_grid, (129, 257))
    assert collage.img_shape == (64, 128)


def test_get_pixel_coordinates(example_img_grid):
    collage = Collage(example_img_grid, (128, 128))
    assert collage.get_top_left_pixel_coordinates(0, 0) == (0, 0)
    assert collage.get_top_left_pixel_coordinates(1, 0) == (0, 64)
    assert collage.get_top_left_pixel_coordinates(0, 1) == (64, 0)
    assert collage.get_top_left_pixel_coordinates(1, 1) == (64, 64)
    with pytest.raises(IndexError):
        collage.get_top_left_pixel_coordinates(-1, 0)
    with pytest.raises(IndexError):
        collage.get_top_left_pixel_coordinates(2, 0)
    with pytest.raises(IndexError):
        collage.get_top_left_pixel_coordinates(0, -1)
    with pytest.raises(IndexError):
        collage.get_top_left_pixel_coordinates(0, 2)


def test_get_grid_coordinates(example_img_grid):
    collage = Collage(example_img_grid, (128, 128))
    assert collage.get_grid_coordinates(0, 0) == (0, 0)
    assert collage.get_grid_coordinates(28, 104) == (1, 0)
    assert collage.get_grid_coordinates(90, 10) == (0, 1)
    assert collage.get_grid_coordinates(70, 70) == (1, 1)
    with pytest.raises(IndexError):
        collage.get_grid_coordinates(-1, 0)
    with pytest.raises(IndexError):
        collage.get_grid_coordinates(128, 0)
    with pytest.raises(IndexError):
        collage.get_grid_coordinates(0, -1)
    with pytest.raises(IndexError):
        collage.get_grid_coordinates(0, 128)
