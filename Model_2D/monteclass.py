import numpy as np


class Point:
    def __init__(self,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 v_x: float = None,
                 v_y: float = None,
                 v_z: float = None,
                 is_in: bool = None):
        if x is not None:
            self.x: float = x  # x coordinate
        if y is not None:
            self.y: float = y  # y coordinate
        if z is not None:
            self.z: float = z  # z coordinate
        if v_x is not None:
            self.v_x: float = v_x  # x velocity
        if v_y is not None:
            self.v_y: float = v_y  # y velocity
        if v_z is not None:
            self.v_z: float = v_z  # z velocity
        if is_in is not None:
            self.is_in: bool = is_in
        pass

    def __deepcopy__(self, memodict={}):
        return Point(self.x, self.y)


class AdditParams:
    def __init__(self,
                 v_x: float = 0.0,
                 v_y: float = 0.0,
                 v_z: float = 0.0,
                 is_in: bool = True):
        self.v_x: float = v_x  # x velocity
        self.v_y: float = v_y  # y velocity
        self.v_z: float = v_z  # z velocity
        self.is_in: bool = is_in
        pass


class Randoms:
    randoms = []
    counter = -1
    size = -1
    start = 0
    end = 0

    def __init__(self, start=0, end=0, size: int = 0):
        self.counter = -1
        self.start = start
        self.end = end
        self.size = size
        self.randoms = np.random.uniform(start, end, size)
        pass

    def get_next(self):
        self.counter += 1
        if self.counter == self.size:
            self.__init__(self.start, self.end, self.size)
            self.counter = 0
        return self.randoms[self.counter]
