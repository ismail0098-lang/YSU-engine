#include "raylib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Basit P3 PPM -> Texture loader
typedef struct {
    int width;
    int height;
} PpmInfo;

static int LoadPPMToTexture(const char *filename, Texture2D *outTex, PpmInfo *info) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("PPM dosyasi acilamadi: %s\n", filename);
        return 0;
    }

    char header[3] = {0};
    if (fscanf(f, "%2s", header) != 1) {
        printf("PPM header okunamadi.\n");
        fclose(f);
        return 0;
    }

    if (strcmp(header, "P3") != 0) {
        printf("Sadece ASCII P3 PPM destekleniyor (P3 bekleniyordu, buldugum: %s)\n", header);
        fclose(f);
        return 0;
    }

    int width, height, maxval;

    // Yorum satirlarini (# ...) atla
    int c;
    do {
        c = fgetc(f);
        if (c == '#') {
            while (c != '\n' && c != EOF) c = fgetc(f);
        }
    } while (c == '#');
    ungetc(c, f);

    if (fscanf(f, "%d %d", &width, &height) != 2) {
        printf("PPM genislik/yukseklik okunamadi.\n");
        fclose(f);
        return 0;
    }

    if (fscanf(f, "%d", &maxval) != 1) {
        printf("PPM maxval okunamadi.\n");
        fclose(f);
        return 0;
    }

    if (maxval <= 0 || maxval > 255) {
        printf("Desteklenmeyen maxval: %d\n", maxval);
        fclose(f);
        return 0;
    }

    unsigned char *data = (unsigned char *)malloc((size_t)width * height * 3);
    if (!data) {
        printf("PPM icin bellek ayrılamadi.\n");
        fclose(f);
        return 0;
    }

    for (int i = 0; i < width * height; i++) {
        int r, g, b;
        if (fscanf(f, "%d %d %d", &r, &g, &b) != 3) {
            printf("PPM piksel verisi eksik.\n");
            free(data);
            fclose(f);
            return 0;
        }
        if (r < 0) r = 0; if (r > 255) r = 255;
        if (g < 0) g = 0; if (g > 255) g = 255;
        if (b < 0) b = 0; if (b > 255) b = 255;
        data[i*3 + 0] = (unsigned char)r;
        data[i*3 + 1] = (unsigned char)g;
        data[i*3 + 2] = (unsigned char)b;
    }

    fclose(f);

    Image img = {
        .data    = data,
        .width   = width,
        .height  = height,
        .mipmaps = 1,
        .format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8
    };

    if (outTex->id != 0) {
        UnloadTexture(*outTex);
    }

    *outTex = LoadTextureFromImage(img);
    UnloadImage(img); // data'yı free eder

    info->width  = width;
    info->height = height;

    printf("PPM yüklendi: %s (%dx%d)\n", filename, width, height);
    return 1;
}

// Equirectangular 360 pano shader (GLSL 330)
static const char *fs360 =
"#version 330\n"
"in vec2 fragTexCoord;\n"
"in vec4 fragColor;\n"
"out vec4 finalColor;\n"
"uniform sampler2D texture0;\n"
"uniform vec4 colDiffuse;\n"
"uniform float yaw;\n"
"uniform float pitch;\n"
"uniform float fovY;\n"
"const float PI = 3.14159265359;\n"
"void main()\n"
"{\n"
"    vec2 ndc = fragTexCoord * 2.0 - 1.0;\n"
"    float x = ndc.x * tan(fovY * 0.5);\n"
"    float y = -ndc.y * tan(fovY * 0.5);\n"
"    vec3 dir = normalize(vec3(x, y, 1.0));\n"
"\n"
"    float cy = cos(yaw);\n"
"    float sy = sin(yaw);\n"
"    float cp = cos(pitch);\n"
"    float sp = sin(pitch);\n"
"\n"
"    vec3 d1 = vec3(cy*dir.x + sy*dir.z, dir.y, -sy*dir.x + cy*dir.z);\n"
"    vec3 d2 = vec3(d1.x, cp*d1.y - sp*d1.z, sp*d1.y + cp*d1.z);\n"
"\n"
"    float lon = atan(d2.z, d2.x);\n"
"    float lat = asin(clamp(d2.y, -1.0, 1.0));\n"
"\n"
"    float u = lon / (2.0*PI) + 0.5;\n"
"    float v = 0.5 - lat / PI;\n"
"\n"
"    vec4 texColor = texture(texture0, vec2(u, v));\n"
"    finalColor = texColor * colDiffuse * fragColor;\n"
"}\n";

int main(void) {
    const int screenWidth  = 1280;
    const int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "YSU 360 Viewer - ysu_360.ppm");
    SetTargetFPS(60);

    Texture2D panoTex = {0};
    PpmInfo panoInfo = {0};

    // 360 render dosyasi: ysu_360.ppm
    if (!LoadPPMToTexture("ysu_360.ppm", &panoTex, &panoInfo)) {
        printf("ysu_360.ppm bulunamadi veya okunamadi. Once ysuengine.exe calistir.\n");
    }

    // Shader'i bellekteki string'ten yükle
    Shader sh360 = LoadShaderFromMemory(NULL, fs360);
    int locYaw   = GetShaderLocation(sh360, "yaw");
    int locPitch = GetShaderLocation(sh360, "pitch");
    int locFovY  = GetShaderLocation(sh360, "fovY");

    // Başlangıç FOV
    float fovDeg = 60.0f;
    float fovRad = fovDeg * (3.14159265359f / 180.0f);
    SetShaderValue(sh360, locFovY, &fovRad, SHADER_UNIFORM_FLOAT);

    // Başlangıç kamera açıları
    float yaw   = 0.0f;  // sağ-sol
    float pitch = 0.0f;  // yukarı-aşağı

    Vector2 lastMouse = {0};
    int rotating = 0;
    const float mouseSens = 0.005f; // hassasiyet

    while (!WindowShouldClose()) {
        // R: 360 dosyayi yeniden yukle (ysuengine yeni render ürettiğinde)
        if (IsKeyPressed(KEY_R)) {
            if (LoadPPMToTexture("ysu_360.ppm", &panoTex, &panoInfo)) {
                printf("ysu_360.ppm yeniden yüklendi.\n");
            }
        }

        // Mouse wheel ile zoom (FOV)
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            // ileri wheel pozitif -> FOV küçülsün (zoom in)
            fovDeg -= wheel * 5.0f;
            if (fovDeg < 20.0f)  fovDeg = 20.0f;
            if (fovDeg > 100.0f) fovDeg = 100.0f;
            fovRad = fovDeg * (3.14159265359f / 180.0f);
            SetShaderValue(sh360, locFovY, &fovRad, SHADER_UNIFORM_FLOAT);
        }

        // LMB veya Alt+LMB ile çevrene bak
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) ||
            (IsKeyDown(KEY_LEFT_ALT) && IsMouseButtonDown(MOUSE_BUTTON_LEFT))) {

            Vector2 mouse = GetMousePosition();
            if (!rotating) {
                rotating  = 1;
                lastMouse = mouse;
            } else {
                Vector2 delta = {
                    mouse.x - lastMouse.x,
                    mouse.y - lastMouse.y
                };
                lastMouse = mouse;

                yaw   -= delta.x * mouseSens;
                pitch += delta.y * mouseSens;

                // pitch clamp (-89°, +89°)
                const float limit = 1.55f;
                if (pitch >  limit) pitch =  limit;
                if (pitch < -limit) pitch = -limit;
            }
        } else {
            rotating = 0;
        }

        // Shader'a yeni açıları gönder
        SetShaderValue(sh360, locYaw,   &yaw,   SHADER_UNIFORM_FLOAT);
        SetShaderValue(sh360, locPitch, &pitch, SHADER_UNIFORM_FLOAT);

        BeginDrawing();
        ClearBackground(BLACK);

        if (panoTex.id != 0) {
            BeginShaderMode(sh360);

            // Tam ekran quad çiz – UV [0,1] araliginda, shader kendi 360 projeksiyonu yapiyor
            Rectangle src = { 0.0f, 0.0f, (float)panoTex.width, (float)panoTex.height };
            Rectangle dst = { 0.0f, 0.0f, (float)screenWidth,   (float)screenHeight };
            Vector2 origin = { 0.0f, 0.0f };
            DrawTexturePro(panoTex, src, dst, origin, 0.0f, WHITE);

            EndShaderMode();

            DrawText("LMB / ALT+LMB: etrafa bak", 10, 10, 18, RAYWHITE);
            DrawText("Mouse wheel: zoom in/out", 10, 32, 16, RAYWHITE);
            DrawText("R: ysu_360.ppm yeniden yukle", 10, 52, 16, RAYWHITE);
        } else {
            DrawText("ysu_360.ppm bulunamadi. Once ysuengine.exe calistir.", 40, screenHeight/2 - 10, 20, RAYWHITE);
        }

        EndDrawing();
    }

    if (panoTex.id != 0) {
        UnloadTexture(panoTex);
    }
    UnloadShader(sh360);

    CloseWindow();
    return 0;
}
