import pytest
import numpy as np

from images import Image


def test_image_creation():
    example_value = 0.42
    pixels = np.ones((64, 64, 3)) * example_value
    img = Image(pixels)
    assert img.get_pixels()[0, 0, 0] == example_value
    with pytest.raises(ValueError):
        Image(pixels * 3)
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
