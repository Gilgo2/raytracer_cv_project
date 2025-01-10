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
'''
def get_color(ray_origin, intersection_point, surface, surfaces,  lights, background_color, materials, margin = 1e-5):
    surface_material = materials[surface.material_index - 1]
    background_color = np.array(background_color, dtype=float)
    reflection_color = np.array(surface_material.reflection_color, dtype=float)
    color = surface_material.transparency * background_color  + reflection_color
    for light in lights:
        light_direction = intersection_point - light.position   
        direction_distance = np.linalg.norm(light_direction)
        light_direction /= direction_distance
        
        light_intersections = find_intersections(light.position, light_direction, surfaces)

        obj, intersection_t = light_intersections[0]
        if len(light_intersections) >= 1 and intersection_t >= direction_distance - margin:
            surface_normal = surface.get_normal(intersection_point)

            #k_a = surface_material.diffuse_color # object ambient color 
            #k_s = surface_material.specular_color # specular color of surface of intersection point - scalar

            diffuse_color = surface_material.diffuse_color * light.specular_intensity * max(0, -light_direction @ surface_normal)

            reflection = 2 * (light_direction @ surface_normal) * surface_normal - light_direction
            reflection /= np.linalg.norm(reflection)
            view_direction = intersection_point - ray_origin
            view_direction /= np.linalg.norm(view_direction)
            specular_color = surface_material.specular_color * light.specular_intensity * max(0, reflection @ view_direction) ** surface_material.shininess 
            
            color += (diffuse_color + specular_color) * (1 - surface_material.transparency)
    return np.clip(color * 255, 0, 255)
'''
    

def ray_trace(camera, scene_settings, objects, width, height):
    image_array = np.zeros((height, width, 3))
    lights = [obj for obj in objects if isinstance(obj, Light)]
    surfaces = [obj for obj in objects if isinstance(obj, Cube) or isinstance(obj, Sphere) or isinstance(obj, InfinitePlane)]
    materials = [obj for obj in objects if isinstance(obj, Material)]
    for j in tqdm(range(height)):
        for i in range(width):
            ray_direction = camera.get_ray(i, j, width, height)
            #intersections = find_intersections(camera.position, ray_direction, surfaces)
            #if len(intersections) > 0:
            #    obj, intersection = intersections[0]
            #    image_array[height-1-j,width-i-1] = get_color(camera.position, intersection * ray_direction + camera.position, obj, surfaces, lights, scene_settings.background_color, materials)
            color = trace_ray(camera.position, ray_direction,
                              surfaces, lights, scene_settings.background_color, materials)  # for example
            image_array[height - 1 - j, width - i - 1] = color
    
    
    return image_array








def trace_ray(ray_origin, ray_direction,surfaces, lights, background_color, materials,depth=0, max_depth=3, margin=1e-5):
    """
    Traces a single ray into the scene:
      1) Finds the closest intersection, if any.
      2) Returns the color at that intersection by calling get_color(...).
      3) If no intersection, returns background_color.
    'depth' is the current recursion depth.
    'max_depth' is a global cap on bounces.
    """
    #if depth > max_depth:
    #    return np.array(background_color)

    intersections = find_intersections(ray_origin, ray_direction, surfaces)
    if len(intersections)==0:
        return np.array(background_color)
    
    # We have at least one intersection
    obj, t_closest = intersections[0]
    intersection_pt = ray_origin + t_closest * ray_direction

    # Compute shading at that point
    color_at_hit = get_color(
        ray_origin,
        intersection_pt,
        obj,
        surfaces,
        lights,
        background_color,
        materials,
        depth,
        max_depth,
        margin
    )

    return color_at_hit


def get_color(ray_origin, intersection_point, surface, surfaces,
              lights, background_color, materials,depth, max_depth, margin=1e-5):
    """
    Returns the color at the intersection_point on 'surface'.
    Potentially spawns reflection rays if the material is reflective.
    """

    surface_material = materials[surface.material_index - 1]
    background_color = np.array(background_color, dtype=float)
    reflection_color = np.array(surface_material.reflection_color, dtype=float)
    color = surface_material.transparency * background_color  + reflection_color
    for light in lights:
        light_direction = intersection_point - light.position   
        direction_distance = np.linalg.norm(light_direction)
        light_direction /= direction_distance
        
        light_intersections = find_intersections(light.position, light_direction, surfaces)

        obj, intersection_t = light_intersections[0]
        if len(light_intersections) >= 1 and intersection_t >= direction_distance - margin:
            surface_normal = surface.get_normal(intersection_point)

            #k_a = surface_material.diffuse_color # object ambient color 
            #k_s = surface_material.specular_color # specular color of surface of intersection point - scalar

            diffuse_color = surface_material.diffuse_color * light.specular_intensity * max(0, -light_direction @ surface_normal)

            reflection = 2 * (light_direction @ surface_normal) * surface_normal - light_direction
            reflection /= np.linalg.norm(reflection)
            view_direction = intersection_point - ray_origin
            view_direction /= np.linalg.norm(view_direction)
            specular_color = surface_material.specular_color * light.specular_intensity * max(0, reflection @ view_direction) ** surface_material.shininess 
            color += (diffuse_color + specular_color) * (1 - surface_material.transparency)
    '''
    # 4) Reflection Ray (the actual recursion):
    #    Suppose we define 'reflection_intensity' in the material
    #    e.g. 'mat.reflection_intensity' in [0..1]
    if depth < max_depth:
        # compute reflection direction
        normal = surface.get_normal(intersection_point)
        normal /= np.linalg.norm(normal)
        incident_dir = intersection_point - ray_origin
        incident_dir /= np.linalg.norm(incident_dir)
        reflect_dir = incident_dir - 2 * np.dot(incident_dir, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)

        # offset to avoid self-intersection
        reflect_origin = intersection_point + margin * reflect_dir

        # recursively get the color
        reflection_color = trace_ray(
            reflect_origin, reflect_dir,
            surfaces, lights, background_color, materials,
            depth+1, max_depth, margin
        )

        # Add reflection contribution
        color += reflection_color * (1 - surface_material.transparency)
    '''
    # 5) Return color in [0..255]
    return np.clip(color, 0, 255)  # or you can keep 0..1 internally

'''
def get_color(ray_origin, intersection_point, surface, surfaces,
              lights, background_color, materials,
              depth=0, max_depth=3, margin=1e-5):
    """
    Returns the color at 'intersection_point' on 'surface', including:
      - Local shading (diffuse + specular).
      - (Optional) An ambient term if you like.
      - A single reflection bounce (if depth < max_depth).
    """

    # Grab the material
    mat = materials[surface.material_index - 1]

    # Convert colors to float [0..1] for internal math
    diffuse_c   = mat.diffuse_color    / 255.0
    specular_c  = mat.specular_color   / 255.0
    reflect_c   = mat.reflection_color / 255.0
    bg_c        = np.array(background_color, dtype=float) / 255.0

    # We'll build up 'color' in [0..1]
    color = np.zeros(3, dtype=float)

    # ----------------------------------------------------
    # 1) LOCAL SHADING: DIFFUSE + SPECULAR (+ optional ambient)
    # ----------------------------------------------------

    # Get the surface normal
    normal = surface.get_normal(intersection_point)
    normal = normal / np.linalg.norm(normal)

    # (Optional) add a small ambient term so unlit areas aren't pure black:
    # e.g. 0.05 or 0.1 times the diffuse color
    ambient_factor = 0.05
    color += ambient_factor * diffuse_c

    for light in lights:
        # Direction from intersection to light
        L = light.position - intersection_point
        dist_to_light = np.linalg.norm(L)
        L /= dist_to_light

        # 1a) Shadow Ray
        shadow_origin = intersection_point + margin * L
        shadow_inters = find_intersections(shadow_origin, L, surfaces)

        # By default, assume we can see the light (not in shadow)
        in_shadow = False
        if shadow_inters:
            # If the closest intersection is closer than the light itself, it's shadowed
            obj_block, t_block = shadow_inters[0]
            if t_block < dist_to_light - margin:
                in_shadow = True

        if not in_shadow:
            # --- Diffuse ---
            lambert = max(0.0, np.dot(normal, L))
            diffuse  = diffuse_c * lambert * light.specular_intensity

            # --- Specular ---
            # reflection of L about N:
            R = 2.0 * np.dot(normal, L) * normal - L
            R /= np.linalg.norm(R)
            # view direction
            V = ray_origin - intersection_point
            V /= np.linalg.norm(V)
            spec_angle  = max(0.0, np.dot(R, V)) ** mat.shininess
            specular    = specular_c * spec_angle * light.specular_intensity

            # Combine
            color += diffuse + specular

    # ----------------------------------------------------
    # 2) REFLECTION (RECURSIVE RAY)
    # ----------------------------------------------------
    if depth < max_depth:
        # direction from the surface point back toward the camera
        incident_dir = (intersection_point - ray_origin)
        incident_dir /= np.linalg.norm(incident_dir)

        # reflection direction
        reflect_dir = incident_dir - 2.0 * np.dot(incident_dir, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)

        reflect_origin = intersection_point + margin * reflect_dir

        # Recursively trace the reflection ray
        reflection_col = trace_ray(
            reflect_origin, 
            reflect_dir,
            surfaces, 
            lights, 
            background_color, 
            materials,
            depth + 1, 
            max_depth, 
            margin
        ) / 255.0  # 'trace_ray' returns in [0..255], convert to [0..1]

        # Multiply by 'reflection_color' to tint or scale reflection
        # If reflection_color == (1,1,1), it's a perfect mirror.
        color += reflect_c * reflection_col

    # Convert [0..1] => [0..255]
    color = np.clip(color, 0.0, 1.0) * 255.0
    return color

'''














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
