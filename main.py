import glob
from tqdm import tqdm
from skimage.io import imsave
from grid import SquareGrid
from images import ImageFromFile, Collage
from pyplot_ui import CollagePltUi


def load_images_in_dir(dir_path: str):
    result = []
    for image_file_path in sorted(glob.glob(dir_path + "/*.*")):
        try:
            img = ImageFromFile(image_file_path)
            result.append(img)
        except Exception as e:
            print(f"Problem with img {image_file_path}: {e}")
    return result


def reload_grid_contents(img_grid: SquareGrid[ImageFromFile]):
    for img in tqdm(img_grid.to_list(), desc="Reloading full-res images from disk"):
        img.reload_original()


def save_as_high_res_collage(img_grid: SquareGrid[ImageFromFile], desired_shape: tuple[int, int], out_path: str):
    reload_grid_contents(img_grid)
    high_res_collage = Collage(img_grid, desired_shape)
    imsave(out_path, high_res_collage.render().pixels)


def generate_collage_from_image_dir():
    image_dir = input("directory with image files:")
    images = load_images_in_dir(image_dir)
    img_grid = SquareGrid.create_by_truncating_excess_elements(images)
    collage = Collage(img_grid, (1000, 1000))
    CollagePltUi(collage)
    save_as_high_res_collage(img_grid, (4000, 4000), "collage.jpg")


if __name__ == '__main__':
    generate_collage_from_image_dir()
