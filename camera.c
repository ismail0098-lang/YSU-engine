// camera.c
#include "camera.h"
#include "vec3.h"
#include "ray.h"

Camera camera_create(float aspect_ratio, float viewport_height, float focal_length)
{
    Camera cam;

    float h = viewport_height;
    float w = aspect_ratio * h;

    cam.origin = vec3(0.0f, 0.0f, 0.0f);

    cam.horizontal = vec3(w, 0.0f, 0.0f);
    cam.vertical   = vec3(0.0f, h, 0.0f);

    cam.lower_left_corner = vec3_sub(
        vec3_sub(
            vec3_sub(cam.origin, vec3_scale(cam.horizontal, 0.5f)),
            vec3_scale(cam.vertical, 0.5f)
        ),
        vec3(0.0f, 0.0f, focal_length)
    );

    return cam;
}

Ray camera_get_ray(Camera cam, float u, float v)
{
    Vec3 p = vec3_add(
                vec3_add(cam.lower_left_corner, vec3_scale(cam.horizontal, u)),
                vec3_scale(cam.vertical, v)
             );

    Vec3 dir = vec3_sub(p, cam.origin);
    return ray_create(cam.origin, vec3_normalize(dir));
}
