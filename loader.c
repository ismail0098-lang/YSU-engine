// scene_loader.c
#include <stdio.h>
#include "sphere.h"
#include "vec3.h"
#include "color.h"

int load_scene(const char *path, struct sphere *out, int max_spheres) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int count = 0;
    while (!feof(f) && count < max_spheres) {
        char tag[16];
        double cx, cy, cz, r, cr, cg, cb;
        if (fscanf(f, "%15s %lf %lf %lf %lf %lf %lf %lf",
                   tag, &cx, &cy, &cz, &r, &cr, &cg, &cb) == 8) {
            if (strcmp(tag, "sphere") == 0) {
                // sphere_init yerine kendi fonksiyonunu koy
                out[count].center = vec3(cx, cy, cz);
                out[count].radius = r;
                out[count].albedo = color(cr, cg, cb);
                count++;
            }
        }
    }
    fclose(f);
    return count;
}
