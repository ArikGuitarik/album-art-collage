from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
from images import Collage


class CollagePltUi:
    def __init__(self, collage: Collage):
        self.collage = collage
        self.selected_grid_coordinates = None
        with mpl.rc_context({'toolbar': 'None'}):
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.imshow = plt.imshow(self.collage.render().get_pixels())
            plt.axis('off')
            plt.show()

    def on_click(self, event: mpl.backend_bases.MouseEvent):
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        row, col = self.collage.get_grid_coordinates(x, y)
        self._select_grid_element(row, col)

    def _select_grid_element(self, row: int, col: int):
        if self.selected_grid_coordinates is None:
            self.selected_grid_coordinates = row, col
        else:
            self.collage.img_grid.swap(row, col, *self.selected_grid_coordinates)
            self.selected_grid_coordinates = None
        self.redraw()

    def redraw(self):
        self.imshow.set_data(self.collage.render().get_pixels())
        self._render_selection()
        plt.draw()

    def _render_selection(self):
        if self.selected_grid_coordinates is None:
            self._remove_any_rectangle()
        else:
            rectangle = self._create_rectangle_around(*self.selected_grid_coordinates)
            self.ax.add_patch(rectangle)

    def _remove_any_rectangle(self):
        [p.remove() for p in self.ax.patches]

    def _create_rectangle_around(self, row, col, line_width=4):
        top_left_pixel = self.collage.get_top_left_pixel_coordinates(row, col)
        height, width = self.collage.img_shape
        return patches.Rectangle(top_left_pixel, width, height,
                                 linewidth=line_width, edgecolor='r', facecolor='none')
