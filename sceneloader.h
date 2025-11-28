#ifndef SCENELOADER_H
#define SCENELOADER_H

#include "vec3.h"

// Editörden gelen sahne için basit veri yapısı
typedef struct {
    Vec3 center;
    float radius;
    Vec3 albedo; // 0–1 arası renk
} SceneSphere;

// scene.txt dosyasını okur, en fazla max_spheres kadar doldurur.
// Başarılı okunan küre sayısını döner.
int load_scene(const char *path, SceneSphere *out, int max_spheres);

#endif
