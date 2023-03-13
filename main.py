import glob
from skimage.io import imread, imsave
from skimage.util import img_as_float

from grid import SquareGrid
from images import Image, Collage


def load_images_in_dir(dir_path: str):
    result = []
    for image_file_path in sorted(glob.glob(dir_path + "/*.*")):
        try:
            img = Image(img_as_float(imread(image_file_path)))
            result.append(img)
        except Exception as e:
            print(f"Problem with img {image_file_path}: {e}")
    return result


def generate_collage_from_image_dir():
    image_dir = input("directory with image files:")
    images = load_images_in_dir(image_dir)
    grid = SquareGrid.create_by_truncating_excess_elements(images)
    renderer = Collage(grid, (128, 128))
    collage = renderer.render()
    imsave("collage.jpg", collage.get_pixels())


if __name__ == '__main__':
    generate_collage_from_image_dir()
