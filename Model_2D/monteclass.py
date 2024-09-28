import numpy as np


class Point:
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 v_x: float = 0.0,
                 v_y: float = 0.0,
                 v_z: float = 0.0,
                 is_in: bool = True):
        self.x: float = x  # x coordinate
        self.y: float = y  # y coordinate
        self.z: float = z  # z coordinate
        self.v_x: float = v_x  # x velocity
        self.v_y: float = v_y  # y velocity
        self.v_z: float = v_z  # z velocity
        self.is_in: bool = is_in
        pass

    def __deepcopy__(self, memodict={}):
        return Point(self.x, self.y, self.z, self.v_x, self.v_y, self.v_z, self.is_in)


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
