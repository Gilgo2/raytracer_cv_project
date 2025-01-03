import numpy as np
def normalize(vector):
        """Normalize a vector to unit length."""
        return vector / np.linalg.norm(vector)
class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width

        self.forward = self.look_at - self.position
        self.forward /= np.linalg.norm(self.forward)
        self.right = np.cross(self.forward, self.up_vector)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)

        

    def get_ray(self, i, j, width, height):
        aspect_ratio = width / height
        screen_height = self.screen_width / aspect_ratio
        pixel_width = self.screen_width / width
        pixel_height = screen_height / height

        screen_center = self.position + self.forward * self.screen_distance
        pixel_x = (i + 0.5) * pixel_width - self.screen_width / 2 
        pixel_y = (j + 0.5) * pixel_height - screen_height / 2 
        pixel_position = screen_center + self.right * pixel_x + self.up * pixel_y

        ray_direction = pixel_position - self.position
        ray_direction /= np.linalg.norm(ray_direction)
        return ray_direction

    

        
        