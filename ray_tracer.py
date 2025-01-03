import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def find_intersections(ray_origin, ray_direction, objects):
    intersections = []
    for obj in objects:
        intersection = obj.intersect(ray_origin, ray_direction)
        if intersection is not None:
            intersections.append((obj, intersection))
    intersections.sort(key=lambda x: x[1])

         
    return intersections

def get_color(intersection_point, surface, surfaces,  lights, background_color, materials):
    surface_material = materials[surface.material_index - 1]
    color = surface_material.transparency * background_color + surface_material.reflection_color
    for light in lights:
        light_direction = intersection_point - light.position   
        direction_distance = np.linalg.norm(light_direction)
        light_direction /= direction_distance
        
        light_intersections = find_intersections(light.position, light_direction, surfaces)

        if len(light_intersections) >= 1 and light_intersections[0][1] >= direction_distance - 1e-5:
            surface_normal = surface.get_normal(intersection_point)

            
            diffuse_color = surface_material.diffuse_color * light.specular_intensity * max(0, -light_direction @ surface_normal)

            reflection = 2 * (light_direction @ surface_normal) * surface_normal - light_direction
            reflection /= np.linalg.norm(reflection)
            view_direction = -light_direction
            specular_color = surface_material.specular_color * light.specular_intensity * max(0, reflection @ view_direction) ** surface_material.shininess
            
            color += (diffuse_color + specular_color) * (1 - surface_material.transparency) 
    return np.clip(color * 255, 0, 255)
    
    

def ray_trace(camera, scene_settings, objects, width, height):
    image_array = np.zeros((height, width, 3))
    lights = [obj for obj in objects if isinstance(obj, Light)]
    surfaces = [obj for obj in objects if isinstance(obj, Cube) or isinstance(obj, Sphere) or isinstance(obj, InfinitePlane)]
    
    materials = [obj for obj in objects if isinstance(obj, Material)]
    for j in tqdm(range(height)):
        for i in range(width):
            ray_direction = camera.get_ray(i, j, width, height)
            intersections = find_intersections(camera.position, ray_direction, surfaces)
            if len(intersections) > 0:
                obj, intersection = intersections[0]
                image_array[height-1-j,width-i-1] = get_color(intersection * ray_direction + camera.position, obj, surfaces, lights, scene_settings.background_color, materials)
    return image_array


def save_image(image_array, output_image):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(output_image)

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    image_array = ray_trace(camera, scene_settings, objects, args.width, args.height)

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
