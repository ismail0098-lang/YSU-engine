#include <stdio.h>
#include <string.h>
#include "sceneloader.h"

// scene.txt formatı:
// sphere cx cy cz radius r g b
// r,g,b = 0–1 arası renk

int load_scene(const char *path, SceneSphere *out, int max_spheres) {
    FILE *f = fopen(path, "r");
    if (!f) {
        printf("Could not open scene file '%s'\n", path);
        return 0;
    }

    int count = 0;
    while (!feof(f) && count < max_spheres) {
        char tag[16];
        double cx, cy, cz, r, cr, cg, cb;

        int n = fscanf(f, "%15s %lf %lf %lf %lf %lf %lf %lf",
                       tag, &cx, &cy, &cz, &r, &cr, &cg, &cb);
        if (n != 8) {
            // satır bozuksa atla
            char buf[256];
            fgets(buf, sizeof(buf), f);
            continue;
        }

        if (strcmp(tag, "sphere") == 0) {
            out[count].center = vec3((float)cx, (float)cy, (float)cz);
            out[count].radius = (float)r;
            out[count].albedo = vec3((float)cr, (float)cg, (float)cb);
            count++;
        }
    }

    fclose(f);
    return count;
}
