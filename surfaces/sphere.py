import numpy as np
class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = np.array(position)
        self.radius = radius
        self.material_index = material_index

    
    def intersect(self, ray_origin, ray_direction, margin=1e-2):
        L = self.position - ray_origin
        tca = L @ ray_direction        
        if tca < 0:
            return None
        d2 = L @ L - tca **2
        if d2 < 0 or d2 > self.radius ** 2:
            return None
        thc = np.sqrt(self.radius ** 2 - d2) 
        t = min(tca - thc, tca+thc)
        return t

    def get_normal(self, intersection_point):
        return (intersection_point - self.position) / self.radius
    
    

