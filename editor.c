// editor.c
#include "raylib.h"
#include "render.h"
#include "vec3.h"
#include "camera.h"
#include <stdlib.h>

int main(void)
{
    const int viewWidth  = 640;
    const int viewHeight = 360;

    InitWindow(viewWidth, viewHeight, "ysuEngine Mini Editor");
    SetTargetFPS(30);

    // Sahneyi kur
    setup_scene();

    // Kamera (sabit)
    float viewport_height = 2.0f;
    float focal_length    = 1.0f;
    float aspect          = (float)viewWidth / (float)viewHeight;
    Camera cam = camera_create(aspect, viewport_height, focal_length);

    // Render için buffer'lar
    Vec3 *pixels = (Vec3 *)malloc(sizeof(Vec3) * viewWidth * viewHeight);
    unsigned char *rgb = (unsigned char *)malloc((size_t)viewWidth * viewHeight * 3);

    if (!pixels || !rgb) {
        CloseWindow();
        return 1;
    }

    // İlk render
    render_scene(pixels, viewWidth, viewHeight, cam,
                 SAMPLES_PER_PIXEL_DEFAULT, MAX_DEPTH_DEFAULT);

    // Vec3 -> RGB
    int idx = 0;
    for (int j = 0; j < viewHeight; ++j) {
        for (int i = 0; i < viewWidth; ++i) {
            Vec3 c = pixels[j * viewWidth + i];
            int ir = (int)(255.999f * c.x);
            int ig = (int)(255.999f * c.y);
            int ib = (int)(255.999f * c.z);
            rgb[idx++] = (unsigned char)ir;
            rgb[idx++] = (unsigned char)ig;
            rgb[idx++] = (unsigned char)ib;
        }
    }

    Image img = {
        .data    = rgb,
        .width   = viewWidth,
        .height  = viewHeight,
        .mipmaps = 1,
        .format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8
    };
    Texture2D tex = LoadTextureFromImage(img);

    while (!WindowShouldClose())
    {
        // İleride buraya WASDQE kamera hareketleri eklenicek

        // R'ye basınca yeniden render
        if (IsKeyPressed(KEY_R)) {
            render_scene(pixels, viewWidth, viewHeight, cam,
                         SAMPLES_PER_PIXEL_DEFAULT, MAX_DEPTH_DEFAULT);

            int idx2 = 0;
            for (int j = 0; j < viewHeight; ++j) {
                for (int i = 0; i < viewWidth; ++i) {
                    Vec3 c = pixels[j * viewWidth + i];
                    int ir = (int)(255.999f * c.x);
                    int ig = (int)(255.999f * c.y);
                    int ib = (int)(255.999f * c.z);
                    rgb[idx2++] = (unsigned char)ir;
                    rgb[idx2++] = (unsigned char)ig;
                    rgb[idx2++] = (unsigned char)ib;
                }
            }

            UpdateTexture(tex, rgb);
        }

        BeginDrawing();
        ClearBackground(BLACK);

        DrawTexturePro(
            tex,
            (Rectangle){0, 0, (float)tex.width, (float)tex.height},
            (Rectangle){0, 0, (float)viewWidth, (float)viewHeight},
            (Vector2){0, 0},
            0.0f,
            WHITE
        );

        DrawText("R: yeniden render | ESC: cikis", 10, 10, 10, RAYWHITE);
        EndDrawing();
    }

    UnloadTexture(tex);
    free(pixels);
    free(rgb);
    CloseWindow();
    return 0;
}
