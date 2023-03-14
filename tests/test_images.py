import pytest
import numpy as np

from images import Image, Collage
from grid import SquareGrid


def test_image_creation():
    example_value = 0.42
    pixels = np.ones((64, 64, 3)) * example_value
    img = Image(pixels)
    assert img.get_pixels()[0, 0, 0] == example_value
    assert img.get_pixels().shape == (64, 64, 3)


def test_image_creation_invalid_range():
    with pytest.raises(ValueError):
        Image(np.ones((64, 64, 3)) * 3)


def test_image_creation_invalid_shape():
    with pytest.raises(ValueError):
        Image(np.ones((10, 10, 1)))


def test_image_resizing():
    pixels = np.zeros((100, 100, 3))
    pixels[0:50, 0:50, :], pixels[0:50, 50:100, :] = 0.2, 0.4
    pixels[50:100, 0:50, :], pixels[50:100, 50:100, :] = 0.6, 0.8
    img = Image(pixels)
    downsized_pixels = img.get_pixels(shape=(6, 6))
    assert downsized_pixels.shape == (6, 6, 3)
    probe_pixels = [downsized_pixels[1, 1, 0], downsized_pixels[1, 4, 0],
                    downsized_pixels[4, 1, 0], downsized_pixels[4, 4, 0]]
    np.testing.assert_allclose(probe_pixels, [0.2, 0.4, 0.6, 0.8], rtol=0.01)


@pytest.fixture
def example_img_grid():
    num_images, img_side_length = 4, 64
    imgs = [Image(np.ones((img_side_length, img_side_length, 3)) * i / num_images) for i in range(num_images)]
    return SquareGrid(imgs)


def test_collage_rendering(example_img_grid):
    result = Collage(example_img_grid, (64, 128)).render().get_pixels()
    assert result[0, 0, 0] == 0
    assert result[28, 208, 2] == 0.25
    assert result[90, 20, 2] == 0.5
    assert result[70, 140, 1] == 0.75
    assert result.shape == (128, 256, 3)


def test_collage_rendering_resized(example_img_grid):
    collage = Collage(example_img_grid, (32, 64))
    result = collage.render().get_pixels()
    assert result[0, 0, 0] == 0
    assert result[14, 104, 2] == 0.25
    assert result[45, 10, 2] == 0.5
    assert result[35, 70, 1] == 0.75
    assert result.shape == (64, 128, 3)
    collage.img_shape = (64, 128)
    result = collage.render().get_pixels()
    assert result[0, 0, 0] == 0
    assert result[28, 208, 2] == 0.25
    assert result[90, 20, 2] == 0.5
    assert result[70, 140, 1] == 0.75
    assert result.shape == (128, 256, 3)


def test_get_pixel_coordinates(example_img_grid):
    collage = Collage(example_img_grid, (64, 64))
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
    collage = Collage(example_img_grid, (64, 64))
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
