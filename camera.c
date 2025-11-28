#include "camera.h"

Camera camera_create(float aspect_ratio, float viewport_height, float focal_length) {
    Camera cam;

    float h = viewport_height;
    float w = aspect_ratio * h;

    cam.origin = vec3(0, 0, 0);

    cam.horizontal = vec3(w, 0, 0);
    cam.vertical   = vec3(0, h, 0);

    cam.lower_left_corner = vec3_sub(
        vec3_sub(
            vec3_sub(cam.origin,
                     vec3_scale(cam.horizontal, 0.5f)),
            vec3_scale(cam.vertical,   0.5f)),
        vec3(0, 0, focal_length)
    );

    return cam;
}

Ray camera_get_ray(Camera cam, float u, float v) {
    // Ray = origin → lower_left + u*horizontal + v*vertical
    Vec3 dir = vec3_sub(
                    vec3_add(
                        vec3_add(cam.lower_left_corner,
                                 vec3_scale(cam.horizontal, u)),
                        vec3_scale(cam.vertical, v)),
                    cam.origin);

    return ray_create(cam.origin, vec3_normalize(dir));
}
