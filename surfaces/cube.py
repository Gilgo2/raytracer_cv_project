
import numpy as np
class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):

        # Slab method
        t_min = -float('inf')
        t_max = float('inf')
        for i in range(3):
            if ray_direction[i] == 0:
                if ray_origin[i] < self.position[i] - self.scale/2 or ray_origin[i] > self.position[i] + self.scale/2:
                    return None
            else:
                t1 = (self.position[i] - self.scale/2 - ray_origin[i]) / ray_direction[i]
                t2 = (self.position[i] + self.scale/2 - ray_origin[i]) / ray_direction[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
                if t_min > t_max:
                    return None
        return t_min
    
    def get_normal(self, intersection_point):
        """
        Finds the intersecting face using the axis with the largest difference between the intersection point and the cube's position
        Then calculate the sign of the normal based on the intersection point's position relative to the cube's position
        """
        delta = np.abs(intersection_point - self.position)
        dominating_axis = np.argmax(delta)
        normal = np.zeros(3)
        normal[dominating_axis] = np.sign(intersection_point[dominating_axis] - self.position[dominating_axis])
        return normal