class Ray:

    def __init__(self, ray_origin, ray_direction):
        self.ray_origin = ray_origin
        self.ray_direction = ray_direction

    
    def find_intersections(self, objects):
        intersections = []
        for obj in objects:
            intersection = obj.intersect(self.ray_origin, self.ray_direction)
            if intersection is not None:
                intersections.append((obj, intersection))
        intersections.sort(key=lambda x: x[1])

        return intersections