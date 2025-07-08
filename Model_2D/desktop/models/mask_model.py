from desktop.models.base_model import BaseModel


class MaskModel(BaseModel):
    def __init__(self):
        super().__init__()
        self._mask_points: list[list[list[int]]] = []

    @property
    def mask_points(self):
        return self._mask_points

    @mask_points.setter
    def mask_points(self, value):
        self._mask_points = value
        self.notify_observers()
