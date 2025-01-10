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
import random

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



def is_unblocked(ray_origin, ray_dir, max_dist, surfaces, margin=1e-5):
    """
    Returns True if there's no intersection from 'ray_origin' in 'ray_dir'
    within distance < max_dist. 
    Otherwise, returns False.
    """
    intersections = find_intersections(ray_origin, ray_dir, surfaces)
    if not intersections:
        return True

    # If the closest intersection is t < max_dist, it's blocked
    _, t_closest = intersections[0]
    return (t_closest >= max_dist - margin)



def hard_shadow_check(intersection_point, surface, light, surfaces, margin=1e-5):
    """
    Performs a single-ray "hard shadow" check from intersection_point to 'light.position'.
    Returns 1.0 if unblocked (lit), 0.0 if blocked (in shadow).

    Offsets the ray origin outward by 'margin' in the surface's normal direction
    to avoid self-intersection.
    """
    # Get the normal from the surface at the intersection
    normal = surface.get_normal(intersection_point)
    normal /= (np.linalg.norm(normal) + 1e-15)

    # Offset the origin slightly so we don't re-hit the same surface
    shadow_origin = intersection_point + margin * normal

    # Direction to the light
    L = light.position - shadow_origin
    dist_to_light = np.linalg.norm(L)
    if dist_to_light < margin:
        return 1.0  # intersection is basically at the light

    L_dir = L / dist_to_light

    # If unblocked, return 1.0 (fully lit), else 0.0 (shadowed)
    if is_unblocked(shadow_origin, L_dir, dist_to_light, surfaces, margin):
        return 1.0
    else:
        return 0.0


def get_color_new(camera_point,
    ray_direction,
    scene_settings,
    surfaces,
    lights,
    background_color,
    materials,
    depth,
    max_depth,
    n_shadow_rays=1,   # <--- here's your shadow ray count
    margin=1e-5
):
    """
    - Finds the first intersection from the camera.
    - If none, returns background.
    - Otherwise, does local shading:
      * compute_shadow_factor(...) for each light
      * do lambert + specular, scaled by the shadow factor
    - reflection recursion is commented out for simplicity
    """

    # 1) intersection
    intersections = find_intersections(camera_point, ray_direction, surfaces)
    if not intersections or depth > max_depth:
        return (np.array(background_color) * 255.0).astype(float)

    surface, t_closest = intersections[0]
    intersection_point = camera_point + t_closest * ray_direction

    # 2) Material
    mat = materials[surface.material_index - 1]
    diffuse_c  = mat.diffuse_color
    specular_c = mat.specular_color
    refl_c     = mat.reflection_color
    transp     = mat.transparency  # in [0..1]

    # base color = partial background + reflection color
    color = transp * np.array(background_color, dtype=float) + refl_c

    # 3) Local shading
    normal = surface.get_normal(intersection_point)
    normal /= (np.linalg.norm(normal) + 1e-15)

    for light in lights:
        # Compute how much this point is lit by 'light' 
        shadow_factor = compute_shadow_factor(
            intersection_point,
            surface,
            light,
            surfaces,
            n_shadow_rays,
            margin
        )
        # shadow_factor in [0..1]

        # standard lambert + spec
        L = light.position - intersection_point
        dist_L = np.linalg.norm(L) + 1e-15
        L_dir = L / dist_L

        lambert = max(0.0, np.dot(normal, L_dir))
        diffuse = diffuse_c * light.specular_intensity * lambert

        # specular
        view_dir = camera_point - intersection_point
        view_dir /= (np.linalg.norm(view_dir) + 1e-15)

        R = 2.0 * np.dot(normal, L_dir) * normal - L_dir
        R /= (np.linalg.norm(R) + 1e-15)

        spec_angle = max(0, np.dot(R, view_dir)) ** mat.shininess
        specular   = specular_c * light.specular_intensity * spec_angle

        local_light_color = (diffuse + specular) * shadow_factor

        # combine
        color += local_light_color * (1.0 - transp)

    # 4) Reflection recursion (commented out for now)
    if depth < max_depth:
        # compute reflection direction
        normal = surface.get_normal(intersection_point)
        normal /= np.linalg.norm(normal)
        incident_dir = intersection_point - camera_point
        incident_dir /= np.linalg.norm(incident_dir)
        reflect_dir = incident_dir - 2 * np.dot(incident_dir, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)

        # offset to avoid self-intersection
        reflect_origin = intersection_point + margin * reflect_dir

        # recursively get the color
        reflection_color = get_color(
            reflect_origin, reflect_dir,scene_settings,
            surfaces, lights, scene_settings.background_color, materials,
            depth+1, max_depth, margin
        )
        # Add reflection contribution
        color += reflection_color*0.0005
    final_color = np.clip(color, 0.0, 1.0) * 255.0
    return final_color






def get_color(camera_point,ray_direction ,scene_settings, surfaces, lights, background_color, materials,depth, max_depth, n_shadow_rays,margin = 1e-5):

    intersections = find_intersections(camera_point, ray_direction, surfaces)
    if len(intersections) ==0:
        return np.array(background_color)

    surface, intersection_t = intersections[0]
    intersection_point = camera_point + intersection_t * ray_direction

    surface_material = materials[surface.material_index - 1]
    #background_color = np.array(background_color, dtype=float)
    #reflection_color = np.array(surface_material.reflection_color, dtype=float)
    color = surface_material.transparency * np.array(background_color)  + surface_material.reflection_color
    for light in lights:
        light_direction = intersection_point - light.position   
        direction_distance = np.linalg.norm(light_direction)
        light_direction /= direction_distance
        
        light_intersections = find_intersections(light.position, light_direction, surfaces)

        obj, intersection_t = light_intersections[0]
        if len(light_intersections) >= 1 and intersection_t >= direction_distance - margin:
            surface_material = materials[surface.material_index - 1]
            surface_normal = surface.get_normal(intersection_point)


            diffuse_color = surface_material.diffuse_color * light.specular_intensity * max(0, -light_direction @ surface_normal)

            reflection = 2 * (light_direction @ surface_normal) * surface_normal - light_direction
            reflection /= np.linalg.norm(reflection)
            view_direction = intersection_point - camera_point
            view_direction /= np.linalg.norm(view_direction)

            specular_color = surface_material.specular_color * light.specular_intensity * max(0, reflection @ view_direction) ** surface_material.shininess 
            
            color += (diffuse_color + specular_color) * (1 - surface_material.transparency)
    #color/=len(lights)

    if depth < max_depth:
        # compute reflection direction
        normal = surface.get_normal(intersection_point)
        normal /= np.linalg.norm(normal)
        incident_dir = intersection_point - camera_point
        incident_dir /= np.linalg.norm(incident_dir)
        reflect_dir = incident_dir - 2 * np.dot(incident_dir, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)

        # offset to avoid self-intersection
        reflect_origin = intersection_point + margin * reflect_dir

        # recursively get the color
        reflection_color = get_color(
            reflect_origin, reflect_dir,scene_settings,
            surfaces, lights, scene_settings.background_color, materials,
            depth+1, max_depth, margin
        )
        # Add reflection contribution
        color += reflection_color*0.0005
    
    return np.clip(color * 255, 0, 255)
    
    

def ray_trace(camera, scene_settings, objects, width, height):
    image_array = np.zeros((height, width, 3))
    lights = [obj for obj in objects if isinstance(obj, Light)]
    surfaces = [obj for obj in objects if isinstance(obj, Cube) or isinstance(obj, Sphere) or isinstance(obj, InfinitePlane)]
    materials = [obj for obj in objects if isinstance(obj, Material)]
    for j in tqdm(range(height)):
        for i in range(width):
            ray_direction = camera.get_ray(i, j, width, height)
            image_array[height-1-j,width-i-1] = get_color_new(camera.position,ray_direction,scene_settings, surfaces, lights, np.array(scene_settings.background_color), materials,depth=0, max_depth=scene_settings.max_recursions,n_shadow_rays=int(scene_settings.root_number_shadow_rays))
    return image_array






import numpy as np
import random

def compute_shadow_factor(
    intersection_point,
    surface,
    light,
    surfaces,
    n_shadow_rays=1,
    margin=1e-5
):
    """
    Returns a shadow factor in [0..1].
      - 0.0 = fully blocked (in shadow)
      - 1.0 = fully lit

    If n_shadow_rays <= 1 or light.radius ~ 0, we do a single hard-shadow ray.
    Otherwise, we do area sampling with n_shadow_rays^2 sub-rays
    for soft shadows.
    
    We assume:
      - surface.get_normal(point) => normal
      - light.position => center of the light
      - light.radius => how large the area light is (float)
    """

    # 1) Get normal from the surface to offset the origin
    normal = surface.get_normal(intersection_point)
    normal_len = np.linalg.norm(normal) + 1e-15
    normal /= normal_len

    # 2) Offset origin to avoid self-intersection
    shadow_origin = intersection_point + margin * normal

    # 3) Vector to light center
    to_light = light.position - shadow_origin
    dist_to_light = np.linalg.norm(to_light)
    if dist_to_light < margin:
        # basically at the light
        return 1.0

    # 4) If single ray or radius is negligible, do a "hard shadow" check
    if n_shadow_rays <= 1 or getattr(light, 'radius', 0.0) < 1e-9:
        L_dir = to_light / (dist_to_light + 1e-15)
        if is_unblocked(shadow_origin, L_dir, dist_to_light, surfaces, margin):
            return 1.0
        else:
            return 0.0

    # 5) Multi-ray approach for soft shadows
    # We'll do n_shadow_rays^2 sub-rays around the area of the light
    unblocked_count = 0
    total_rays = n_shadow_rays * n_shadow_rays

    # Build an orthonormal basis around to_light
    L_dir = to_light / (dist_to_light + 1e-15)
    up = np.array([0,1,0], dtype=float)
    if abs(np.dot(L_dir, up)) > 0.99:
        up = np.array([1,0,0], dtype=float)

    x_axis = np.cross(L_dir, up)
    x_axis /= (np.linalg.norm(x_axis) + 1e-15)
    y_axis = np.cross(L_dir, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-15)

    # How big is the "light radius" area
    cell_size = (2.0 * light.radius) / n_shadow_rays

    for i in range(n_shadow_rays):
        for j in range(n_shadow_rays):
            rx = random.random()  
            ry = random.random()

            px = -light.radius + (i + rx) * cell_size
            py = -light.radius + (j + ry) * cell_size

            offset_vec = px * x_axis + py * y_axis
            sample_light_pos = light.position + offset_vec

            sub_vec = sample_light_pos - shadow_origin
            dist_sub = np.linalg.norm(sub_vec)
            if dist_sub < margin:
                # effectively at the same point
                unblocked_count += 1
                continue

            sub_dir = sub_vec / dist_sub

            if is_unblocked(shadow_origin, sub_dir, dist_sub, surfaces, margin):
                unblocked_count += 1

    shadow_factor = unblocked_count / float(total_rays)
    return shadow_factor








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
