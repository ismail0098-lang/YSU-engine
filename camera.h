#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"

typedef struct {
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
} Camera;

// Create camera with viewport params
Camera camera_create(float aspect_ratio, float viewport_height, float focal_length);

// Generate ray from (u, v) screen coords
Ray camera_get_ray(Camera cam, float u, float v);

#endif
