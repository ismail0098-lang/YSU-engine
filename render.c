#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "sceneloader.h"

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"
#include "image.h"
#include "color.h"
#include "render.h"

// 360 panorama fonksiyonunu başka dosyadan kullanacağız
// (ysu_360_engine_integration.c içinde tanımlı olmalı)
extern void ysu_render_360(const Camera *cam, const char *out_ppm);

// Varsayılan ayarlar (header'da da tanımlı)
#define IMAGE_WIDTH       IMAGE_WIDTH_DEFAULT
#define ASPECT_RATIO      ASPECT_RATIO_DEFAULT
#define SAMPLES_PER_PIXEL SAMPLES_PER_PIXEL_DEFAULT
#define MAX_DEPTH         MAX_DEPTH_DEFAULT

#define MAX_SPHERES   16
#define MAX_MATERIALS 16

static Sphere   g_spheres[MAX_SPHERES];
static int      g_num_spheres   = 0;
static Material g_materials[MAX_MATERIALS];
static int      g_num_materials = 0;

// Basit rastgele sayı [0,1)
static float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

// Dünya ile çarpışma kontrolü
static int hit_world(Ray r, float t_min, float t_max, HitRecord *out_rec) {
    HitRecord temp_rec;
    int   hit_anything   = 0;
    float closest_so_far = t_max;

    for (int i = 0; i < g_num_spheres; i++) {
        temp_rec = sphere_intersect(g_spheres[i], r, t_min, closest_so_far);
        if (temp_rec.hit) {
            hit_anything   = 1;
            closest_so_far = temp_rec.t;
            *out_rec       = temp_rec;
        }
    }

    return hit_anything;
}

// Işının rengi (360 rendering için de kullanacağız)
Vec3 ray_color_internal(Ray r, int depth) {
    if (depth <= 0) {
        return vec3(0, 0, 0);
    }

    HitRecord rec;
    if (hit_world(r, 0.001f, 1000.0f, &rec)) {
        if (rec.material_index < 0 || rec.material_index >= g_num_materials) {
            return vec3(1, 0, 1); // hata ayıklama için mor
        }

        const Material *mat = &g_materials[rec.material_index];
        Ray  scattered;
        Vec3 attenuation;

        if (material_scatter(mat, r, rec.point, rec.normal, &scattered, &attenuation)) {
            Vec3 col = ray_color_internal(scattered, depth - 1);  // recursive call
            return vec3_mul(attenuation, col);
        } else {
            return vec3(0, 0, 0);
        }
    }

    // Arka plan (sky gradient)
    Vec3 unit_dir = vec3_normalize(r.direction);
    float t = 0.5f * (unit_dir.y + 1.0f);
    Vec3 c1 = vec3(1.0f, 1.0f, 1.0f);
    Vec3 c2 = vec3(0.5f, 0.7f, 1.0f);
    return vec3_add(vec3_scale(c1, (1.0f - t)), vec3_scale(c2, t));
}

// HEADER ile uyumlu dış fonksiyon (eski kodun çağırdığı)
Vec3 ray_color(Ray r, int depth) {
    return ray_color_internal(r, depth);
}

// Materyal ekle, index döndür
static int add_material(MaterialType type, Vec3 albedo, float fuzz) {
    if (g_num_materials >= MAX_MATERIALS) return -1;
    Material m;
    m.type   = type;
    m.albedo = albedo;
    m.fuzz   = fuzz;
    g_materials[g_num_materials] = m;
    return g_num_materials++;
}

// Küre ekle
static void add_sphere(Vec3 center, float radius, int material_index) {
    if (g_num_spheres >= MAX_SPHERES) return;
    g_spheres[g_num_spheres++] = sphere_create(center, radius, material_index);
}

// -----------------------------------------------------------------------------
//  SCENE LOADER ENTEGRASYONU
// -----------------------------------------------------------------------------

// scene.txt'den sahne kurmayı dener.
// Başarılıysa 1, başarısızsa 0 döner.
static int setup_scene_from_file(const char *path) {
    SceneSphere temp[MAX_SPHERES];
    int count = load_scene(path, temp, MAX_SPHERES);
    if (count <= 0) {
        return 0; // hiçbir şey yüklenemediyse fallback kullan
    }

    g_num_spheres   = 0;
    g_num_materials = 0;

    for (int i = 0; i < count; ++i) {
        // Her küreye kendi albedosuyla lambertian materyal veriyoruz
        int mat = add_material(MAT_LAMBERTIAN, temp[i].albedo, 0.0f);
        if (mat < 0) break;
        add_sphere(temp[i].center, temp[i].radius, mat);
    }

    printf("Scene loaded from '%s' with %d spheres.\n", path, g_num_spheres);
    return 1;
}

// Eski hard-coded sahne (fallback)
static void setup_default_scene(void) {
    g_num_spheres   = 0;
    g_num_materials = 0;

    // Zemin
    int ground_mat = add_material(MAT_LAMBERTIAN, vec3(0.8f, 0.8f, 0.0f), 0.0f);
    add_sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, ground_mat);

    // Ortadaki diffuse küre
    int center_mat = add_material(MAT_LAMBERTIAN, vec3(0.1f, 0.2f, 0.5f), 0.0f);
    add_sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, center_mat);

    // Sağda metal küre
    int metal_mat = add_material(MAT_METAL, vec3(0.8f, 0.6f, 0.2f), 0.1f);
    add_sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, metal_mat);

    // Solda biraz daha karanlık diffuse küre
    int left_mat = add_material(MAT_LAMBERTIAN, vec3(0.8f, 0.1f, 0.1f), 0.0f);
    add_sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, left_mat);

    printf("Default hard-coded scene loaded.\n");
}

// Dışarıdan çağrılan sahne kurma fonksiyonu (render.h'la uyumlu)
void setup_scene(void) {
   
    if (!setup_scene_from_file("scene.txt")) {
        
        setup_default_scene();
    }
}

// 🔹 ANA RENDER FONKSİYONU 🔹
// pixels: image_width * image_height uzunlukta Vec3 dizisi
void render_scene(Vec3 *pixels,
                  int image_width,
                  int image_height,
                  Camera cam,
                  int samples_per_pixel,
                  int max_depth)
{
    for (int j = 0; j < image_height; j++) {
        printf("Scanline %d / %d\r", j + 1, image_height);
        fflush(stdout);

        for (int i = 0; i < image_width; i++) {
            Vec3 col = vec3(0, 0, 0);

            for (int s = 0; s < samples_per_pixel; s++) {
                float u = ((float)i + rand_float()) / (float)(image_width  - 1);
                float v = ((float)j + rand_float()) / (float)(image_height - 1);

                Ray r = camera_get_ray(cam, u, v);
                Vec3 sample = ray_color(r, max_depth);
                col = vec3_add(col, sample);
            }

            // Ortalama al
            float scale = 1.0f / (float)samples_per_pixel;
            col = vec3_scale(col, scale);

            // Basit gamma düzeltme (gamma=2)
            col.x = sqrtf(col.x);
            col.y = sqrtf(col.y);
            col.z = sqrtf(col.z);

            pixels[j * image_width + i] = col;
        }
    }
}


int main(void) {
    srand((unsigned int)time(NULL));

    const int image_width  = IMAGE_WIDTH;
    const int image_height = (int)(image_width / ASPECT_RATIO);

    printf("Image size: %d x %d\n", image_width, image_height);

    // Sahne kur (scene.txt varsa oradan, yoksa default)
    setup_scene();

    // Kamera
    float viewport_height = 2.0f;
    float focal_length    = 1.0f;
    Camera cam = camera_create(ASPECT_RATIO, viewport_height, focal_length);

    // Pixel buffer
    Vec3 *pixels = (Vec3 *)malloc(sizeof(Vec3) * image_width * image_height);
    if (!pixels) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Normal render (output.ppm)
    render_scene(pixels, image_width, image_height,
                 cam, SAMPLES_PER_PIXEL, MAX_DEPTH);

    printf("\nWriting image (output.ppm)...\n");
    image_write_ppm("output.ppm", image_width, image_height, pixels);

    // Aynı sahne ve kamerayla 360 panorama
    printf("Rendering 360 panorama (ysu_360.ppm)...\n");
    ysu_render_360(&cam, "ysu_360.ppm");

    free(pixels);

    printf("Done. Images: output.ppm + ysu_360.ppm\n");
    return 0;
}
