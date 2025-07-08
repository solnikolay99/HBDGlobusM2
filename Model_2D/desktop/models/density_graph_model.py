from desktop.models.mask_model import MaskModel


class DensityGraphModel(MaskModel):
    def __init__(self):
        super().__init__()
        self._main_box: list[list[int]] = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self._points: list[list[int]] = []

    @property
    def main_box(self):
        return self._main_box

    @main_box.setter
    def main_box(self, value):
        self._main_box = value
        self.notify_observers()

    def set_main_box(self, x_lo, y_lo, x_hi, y_hi):
        self.main_box = [[x_lo, y_lo],
                         [x_lo, y_hi],
                         [x_hi, y_hi],
                         [x_hi, y_lo]]

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value
        self.notify_observers()
