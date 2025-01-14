import numpy as np 
from ray import Ray
import random

def is_unblocked(ray, max_dist, surfaces, margin=1e-5):
    """
    Returns True if there's no intersection from 'ray_origin' in 'ray_dir'
    within distance < max_dist. 
    Otherwise, returns False.
    """
    intersections = ray.find_intersections(surfaces)
    if not intersections:
        return True

    # If the closest intersection is t < max_dist, it's blocked
    _, t_closest = intersections[0]
    return (t_closest >= max_dist - margin)




def get_color(ray,
                  scene_settings,
                  surfaces,
                  lights,
                  background_color,
                  materials,
                  depth,
                  max_depth,
                  n_shadow_rays,
                  margin=1e-5):
    """
    Returns the color for a ray in a scene, handling multiple intersections
    from the farthest transparent object to the nearest one (back-to-front).
    """

    # 1) If we've hit recursion limit, return background
    if depth >= max_depth:
        return np.array(background_color, dtype=float)

    # 2) Gather *all* intersections (not just nearest)
    all_inters = ray.find_intersections(surfaces)
    if len(all_inters)==0:
        return np.array(background_color, dtype=float)


    usable_inters = []
    for i, (obj, t_value) in enumerate(all_inters):
        mat = materials[obj.material_index - 1]
        if mat.transparency == 0:
            # Keep only up to (and including) this index - rays don't pass this material...
            usable_inters = all_inters[:i+1]
            break
    else:
        # If we never 'break', we keep them all
        usable_inters = all_inters

    if len(usable_inters)==0:
        return np.array(background_color, dtype=float)

    
    # 4) We'll accumulate the color from the *farthest* intersection
    #    to the *nearest*. 
    color = None
    accumulated_bg = np.array(background_color, dtype=float)

    # Intersection list is sorted from *nearest* to *farthest*,
    # so we go backward to handle back-to-front transparency.
    count_inters = len(usable_inters) - 1
    while count_inters >= 0:
        surface_obj, t_val = usable_inters[count_inters]
        count_inters -= 1

        # 4a) Intersection point
        intersection_point = ray.ray_origin + t_val * ray.ray_direction
        
        # 4b) Grab material
        mat_index = surface_obj.material_index - 1
        mat = materials[mat_index]
        transp = mat.transparency
        refl_c = mat.reflection_color
        diffuse_c = mat.diffuse_color
        specular_c = mat.specular_color

        # 4c) Reflection recursion
        reflection_color = np.zeros(3, dtype=float)
        if depth < max_depth:
            normal = surface_obj.get_normal(intersection_point)
            normal /= np.linalg.norm(normal)

            # direction from camera_point to intersection
            incident_dir = intersection_point - ray.ray_origin
            incident_dir /= np.linalg.norm(incident_dir)

            reflect_dir = incident_dir - 2.0 * np.dot(incident_dir, normal) * normal
            reflect_dir /= np.linalg.norm(reflect_dir)
            reflect_origin = intersection_point + margin * reflect_dir
            reflection_ray = Ray(reflect_origin, reflect_dir)
            reflection_color = get_color(
                reflection_ray,
                scene_settings, surfaces, lights,
                background_color, materials,
                depth+1, max_depth, n_shadow_rays, margin
            )
            # Scale reflection by the material's reflection color
            reflection_color *= refl_c

        # surface normal
        normal = surface_obj.get_normal(intersection_point)
        normal /= np.linalg.norm(normal)

        specular_sum = np.array([0.0, 0.0, 0.0])
        diffuse_sum = np.array([0.0, 0.0, 0.0])
        for light in lights:
            L = light.position - intersection_point
            dist_L = np.linalg.norm(L)
            L_dir = L / dist_L

            # Lambert
            lambert = max(0.0, np.dot(normal, L_dir))

            #if the light is behind the surface
            if lambert <= 0:
                continue

            shadow_factor = compute_shadow_factor(
                intersection_point,
                surface_obj,
                light,
                surfaces,
                n_shadow_rays,
                margin
            )

            diffuse_sum += np.float64(light.color) * shadow_factor * light.specular_intensity * lambert

            # Specular
            view_dir = ray.ray_origin - intersection_point
            view_dir /= np.linalg.norm(view_dir)
            R = 2.0 * np.dot(normal, L_dir) * normal - L_dir
            R /= np.linalg.norm(R) 

            spec_angle = max(0.0, np.dot(R, view_dir)) ** mat.shininess
            specular_sum += shadow_factor * np.float64(light.color) * light.specular_intensity * spec_angle # specular_sum instead of _c

            # Combine diffuse + specular for this light
        
        local_light_color = (diffuse_sum*diffuse_c + specular_sum*specular_c) 
        local_shading = local_light_color

        new_color = (1.0 - transp) * local_shading + transp * accumulated_bg + reflection_color

        # For layering multiple intersections, we treat new_color as the
        # "background" for the next intersection in front
        accumulated_bg = new_color
        color = new_color
    return color





def compute_shadow_factor(
    intersection_point,
    surface,
    light,
    surfaces,
    n_shadow_rays,
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
    normal_len = np.linalg.norm(normal)
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
    if n_shadow_rays == 0 or getattr(light, 'radius', 0.0) < 1e-9:
        L_dir = to_light / dist_to_light 
        shadow_ray = Ray(shadow_origin, L_dir)
        if is_unblocked(shadow_ray, dist_to_light, surfaces, margin):
            return 1.0
        else:
            return 0.0
    
    # 5) Multi-ray approach for soft shadows
    # We'll do n_shadow_rays^2 sub-rays around the area of the light
    unblocked_count = 0
    total_rays = n_shadow_rays * n_shadow_rays

    # Build an orthonormal basis around to_light
    L_dir = to_light / dist_to_light 
    up = np.array([0,1,0], dtype=float)
    if abs(np.dot(L_dir, up)) > 0.99:
        up = np.array([1,0,0], dtype=float)

    x_axis = np.cross(L_dir, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(L_dir, x_axis)
    y_axis /= np.linalg.norm(y_axis)

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
            shadow_ray = Ray(shadow_origin, sub_dir)
            if is_unblocked(shadow_ray, dist_sub, surfaces, margin):
                unblocked_count += 1

    base_intensity = 1.0 - light.shadow_intensity
    shadow_contribution = light.shadow_intensity * (unblocked_count / float(total_rays))
    shadow_factor = base_intensity + shadow_contribution
    return shadow_factor
