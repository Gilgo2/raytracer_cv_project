import numpy as np
class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal)
        self.offset = offset
        self.material_index = material_index


    def intersect(self, ray_origin, ray_direction):
        t = - (ray_origin @ self.normal + self.offset) / (ray_direction @ self.normal)
        if t < 0:
            return None
        return t

    def get_normal(self, intersection_point):
        return self.normal