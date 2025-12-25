// ysu_edit_mode.c
#include "raylib.h"
#include "raymath.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "vec3.h"
#include "sceneloader.h"   // SceneSphere + load_scene

#define MAX_SCENE_SPHERES 16

// ---------- Transform / selection tipleri ----------

typedef enum {
    TRANSFORM_NONE = 0,
    TRANSFORM_GRAB_SPHERE = 1,
    TRANSFORM_ROTATE_SPHERE = 2,
    TRANSFORM_GRAB_EDGE = 3
} TransformMode;

typedef enum {
    AXIS_FREE = 0,
    AXIS_X    = 1,
    AXIS_Y    = 2,
    AXIS_Z    = 3
} TransformAxis;

// ---------- Edit mesh (küp) için yapılar ----------

#define MAX_EDIT_VERTS 16
#define MAX_EDIT_EDGES 32

typedef struct {
    Vector3 pos;
} EditVertex;

typedef struct {
    int v0, v1;
} EditEdge;

static EditVertex g_editVerts[MAX_EDIT_VERTS];
static int        g_numEditVerts = 0;

static EditEdge   g_editEdges[MAX_EDIT_EDGES];
static int        g_numEditEdges = 0;

static int        g_selectedEdge = -1;

// Küçük bir küp oluştur (merkez 0,0.5,0 civarı)
static void InitEditCube(void)
{
    g_numEditVerts = 8;
    g_numEditEdges = 12;

    float s = 1.0f;
    float y0 = 0.0f;
    float y1 = 2.0f;

    g_editVerts[0].pos = (Vector3){ -s, y0, -s };
    g_editVerts[1].pos = (Vector3){  s, y0, -s };
    g_editVerts[2].pos = (Vector3){  s, y0,  s };
    g_editVerts[3].pos = (Vector3){ -s, y0,  s };

    g_editVerts[4].pos = (Vector3){ -s, y1, -s };
    g_editVerts[5].pos = (Vector3){  s, y1, -s };
    g_editVerts[6].pos = (Vector3){  s, y1,  s };
    g_editVerts[7].pos = (Vector3){ -s, y1,  s };

    int e = 0;
    // Alt yüz
    g_editEdges[e++] = (EditEdge){0,1};
    g_editEdges[e++] = (EditEdge){1,2};
    g_editEdges[e++] = (EditEdge){2,3};
    g_editEdges[e++] = (EditEdge){3,0};
    // Üst yüz
    g_editEdges[e++] = (EditEdge){4,5};
    g_editEdges[e++] = (EditEdge){5,6};
    g_editEdges[e++] = (EditEdge){6,7};
    g_editEdges[e++] = (EditEdge){7,4};
    // Dikey kenarlar
    g_editEdges[e++] = (EditEdge){0,4};
    g_editEdges[e++] = (EditEdge){1,5};
    g_editEdges[e++] = (EditEdge){2,6};
    g_editEdges[e++] = (EditEdge){3,7};
}

// ---------- Ortak yardımcılar ----------

static int save_scene(const char *path, SceneSphere *spheres, int count)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        printf("scene.txt yazmak icin acilamadi.\n");
        return 0;
    }

    for (int i = 0; i < count; ++i) {
        SceneSphere *s = &spheres[i];
        // Şu an sadece sphere verisini yazıyoruz
        fprintf(f, "sphere %f %f %f %f %f %f %f\n",
                s->center.x,  s->center.y,  s->center.z,
                s->radius,
                s->albedo.x,  s->albedo.y,  s->albedo.z);
    }

    fclose(f);
    printf("scene.txt kaydedildi (%d sphere).\n", count);
    return 1;
}

static Color ColorFromVec3(Vec3 v)
{
    int r = (int)(v.x * 255.0f);
    int g = (int)(v.y * 255.0f);
    int b = (int)(v.z * 255.0f);

    if (r < 0)   r = 0;
    if (r > 255) r = 255;

    if (g < 0)   g = 0;
    if (g > 255) g = 255;

    if (b < 0)   b = 0;
    if (b > 255) b = 255;

    return (Color){ (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 };
}

// Yeni sphere ekleme helper'i
static int add_sphere(SceneSphere *spheres, int *pCount, Vec3 center, float radius, Vec3 albedo)
{
    if (*pCount >= MAX_SCENE_SPHERES) {
        printf("MAX_SCENE_SPHERES limitine ulastin, yeni sphere eklenemiyor.\n");
        return -1;
    }

    int idx = (*pCount)++;
    spheres[idx].center = center;
    spheres[idx].radius = radius;
    spheres[idx].albedo = albedo;

    printf("Edit mode: yeni sphere eklendi (index=%d)\n", idx);
    return idx;
}

// Ray - segment arası en küçük mesafenin karesi
static float DistanceRayToSegmentSq(Ray ray, Vector3 a, Vector3 b)
{
    Vector3 v = Vector3Subtract(b, a);
    Vector3 w0 = Vector3Subtract(ray.position, a);

    float A = Vector3DotProduct(ray.direction, ray.direction);
    float B = Vector3DotProduct(ray.direction, v);
    float C = Vector3DotProduct(v, v);
    float D = Vector3DotProduct(ray.direction, w0);
    float E = Vector3DotProduct(v, w0);

    float denom = A*C - B*B;
    float sc, tc;

    if (fabsf(denom) < 1e-6f) {
        sc = 0.0f;
        tc = E / C;
    } else {
        sc = (B*E - C*D) / denom;
        tc = (A*E - B*D) / denom;
    }

    if (sc < 0.0f) sc = 0.0f;
    if (tc < 0.0f) tc = 0.0f;
    if (tc > 1.0f) tc = 1.0f;

    Vector3 pRay = Vector3Add(ray.position, Vector3Scale(ray.direction, sc));
    Vector3 pSeg = Vector3Add(a, Vector3Scale(v, tc));
    Vector3 diff = Vector3Subtract(pRay, pSeg);
    return Vector3DotProduct(diff, diff);
}

// ---------- main ----------

int main(void)
{
    const int screenWidth  = 1280;
    const int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "YSU Edit Mode - scene.txt");
    SetTargetFPS(60);

    // Orbit camera
    Vector3 target = (Vector3){ 0.0f, 0.5f, 0.0f };
    float distance = 6.0f;
    float yaw      = 0.0f;
    float pitch    = 0.35f;

    Camera3D cam = {0};
    cam.position   = (Vector3){ 0.0f, 2.0f, -distance };
    cam.target     = target;
    cam.up         = (Vector3){ 0.0f, 1.0f, 0.0f };
    cam.fovy       = 60.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    Vector2 lastMouse = {0};
    int rotatingCam = 0;
    const float sensOrbit = 0.005f;

    // Scene spheres
    SceneSphere spheres[MAX_SCENE_SPHERES];
    int sphereCount = 0;
    int selectedSphere = -1;

    // Transform state
    TransformMode mode = TRANSFORM_NONE;
    TransformAxis axis = AXIS_FREE;

    // Sphere grab
    Vector2 grabStartMouse  = {0};
    Vec3    grabStartCenter = {0};

    // Sphere rotate (şimdilik basit – sadece y ekseni göstergesi için yer bırakıyoruz)
    Vector2 rotStartMouse   = {0};
    float   rotStartYawDeg  = 0.0f;  // sadece y ekseni etrafında açı
    float   sphereYawDeg[MAX_SCENE_SPHERES] = {0}; // her sphere için basit yaw

    // Edge grab
    Vector2 edgeGrabStartMouse = {0};
    Vector3 edgeStartV0 = {0};
    Vector3 edgeStartV1 = {0};

    // Edit mesh (küp)
    InitEditCube();

    // İlk scene.txt yükle
    sphereCount = load_scene("scene.txt", spheres, MAX_SCENE_SPHERES);
    if (sphereCount > 0) {
        selectedSphere = 0;
    }
    printf("Edit mode: scene.txt içinden %d sphere yüklendi.\n", sphereCount);

    while (!WindowShouldClose())
    {
        // ---------- Kamera kontrolü ----------
        float wheel = GetMouseWheelMove();
        distance -= wheel * 0.5f;
        if (distance < 1.5f) distance = 1.5f;
        if (distance > 30.0f) distance = 30.0f;

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && IsKeyDown(KEY_LEFT_ALT))
        {
            Vector2 m = GetMousePosition();
            if (!rotatingCam) { rotatingCam = 1; lastMouse = m; }
            else {
                Vector2 d = { m.x - lastMouse.x, m.y - lastMouse.y };
                lastMouse = m;

                yaw   -= d.x * sensOrbit;
                pitch -= d.y * sensOrbit;

                float limit = 1.55f;
                if (pitch >  limit) pitch =  limit;
                if (pitch < -limit) pitch = -limit;
            }
        }
        else rotatingCam = 0;

        float cp = cosf(pitch);
        cam.position.x = target.x + distance * cp * cosf(yaw);
        cam.position.y = target.y + distance * sinf(pitch);
        cam.position.z = target.z + distance * cp * sinf(yaw);
        cam.target = target;

        // Kamera yön vektörleri
        Vector3 forward = Vector3Subtract(cam.target, cam.position);
        if (Vector3Length(forward) < 0.001f) {
            forward = (Vector3){0.0f, 0.0f, 1.0f};
        }
        forward = Vector3Normalize(forward);
        Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, (Vector3){0.0f,1.0f,0.0f}));
        Vector3 up    = (Vector3){0.0f,1.0f,0.0f};

        // ---------- Edge selection (LMB, ALT yok, transform yokken) ----------
        if (mode == TRANSFORM_NONE &&
            !IsKeyDown(KEY_LEFT_ALT) &&
            IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            Ray ray = GetMouseRay(GetMousePosition(), cam);
            float bestDistSq = 0.05f * 0.05f; // threshold
            int bestEdge = -1;

            for (int i = 0; i < g_numEditEdges; ++i) {
                EditEdge e = g_editEdges[i];
                Vector3 a = g_editVerts[e.v0].pos;
                Vector3 b = g_editVerts[e.v1].pos;

                float d2 = DistanceRayToSegmentSq(ray, a, b);
                if (d2 < bestDistSq) {
                    bestDistSq = d2;
                    bestEdge = i;
                }
            }

            g_selectedEdge = bestEdge;

            if (g_selectedEdge >= 0) {
                printf("Edge secildi: %d\n", g_selectedEdge);
            }
        }

        // ---------- Sphere ekleme ----------
        if (IsKeyPressed(KEY_N) && mode == TRANSFORM_NONE) {
            Vec3 c   = vec3(0.0f, 0.5f, -2.0f);
            Vec3 col = vec3(0.3f, 0.9f, 0.4f);
            int idx  = add_sphere(spheres, &sphereCount, c, 0.5f, col);
            if (idx >= 0) selectedSphere = idx;
        }

        // ---------- Sphere seçimi (TAB) ----------
        if (sphereCount > 0 && mode == TRANSFORM_NONE) {
            if (IsKeyPressed(KEY_TAB)) {
                selectedSphere = (selectedSphere + 1) % sphereCount;
                g_selectedEdge = -1; // edge selection'ı temizle
            }
        }

        // ---------- Grab başlatma (G) ----------
        if (mode == TRANSFORM_NONE && IsKeyPressed(KEY_G)) {
            // Önce edge öncelikli
            if (g_selectedEdge >= 0) {
                mode = TRANSFORM_GRAB_EDGE;
                axis = AXIS_FREE;
                edgeGrabStartMouse = GetMousePosition();
                EditEdge e = g_editEdges[g_selectedEdge];
                edgeStartV0 = g_editVerts[e.v0].pos;
                edgeStartV1 = g_editVerts[e.v1].pos;
                printf("EDGE GRAB mode basladi. edge=%d\n", g_selectedEdge);
            }
            // Edge yoksa sphere
            else if (sphereCount > 0 && selectedSphere >= 0) {
                mode = TRANSFORM_GRAB_SPHERE;
                axis = AXIS_FREE;
                grabStartMouse  = GetMousePosition();
                grabStartCenter = spheres[selectedSphere].center;
                printf("SPHERE GRAB mode basladi. sphere=%d\n", selectedSphere);
            }
        }

        // ---------- Rotate başlatma (R) – sadece sphere ----------
        if (sphereCount > 0 && selectedSphere >= 0 && mode == TRANSFORM_NONE) {
            if (IsKeyPressed(KEY_R)) {
                mode = TRANSFORM_ROTATE_SPHERE;
                axis = AXIS_FREE;
                rotStartMouse  = GetMousePosition();
                rotStartYawDeg = sphereYawDeg[selectedSphere];
                printf("SPHERE ROTATE mode basladi. sphere=%d\n", selectedSphere);
            }
        }

        // ---------- Transform modunda eksen kilidi (X/Y/Z) ----------
        if (mode == TRANSFORM_GRAB_SPHERE ||
            mode == TRANSFORM_GRAB_EDGE   ||
            mode == TRANSFORM_ROTATE_SPHERE)
        {
            if (IsKeyPressed(KEY_X)) { axis = AXIS_X; printf("Axis: X\n"); }
            if (IsKeyPressed(KEY_Y)) { axis = AXIS_Y; printf("Axis: Y\n"); }
            if (IsKeyPressed(KEY_Z)) { axis = AXIS_Z; printf("Axis: Z\n"); }
        }

        // ---------- SPHERE GRAB güncelle ----------
        if (mode == TRANSFORM_GRAB_SPHERE && sphereCount > 0 && selectedSphere >= 0) {
            SceneSphere *s = &spheres[selectedSphere];

            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER)) {
                mode = TRANSFORM_NONE;
                printf("SPHERE GRAB onaylandi.\n");
            }
            else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                s->center = grabStartCenter;
                mode = TRANSFORM_NONE;
                printf("SPHERE GRAB iptal.\n");
            }
            else {
                Vector2 m = GetMousePosition();
                Vector2 d = { m.x - grabStartMouse.x, m.y - grabStartMouse.y };

                float moveScale = 0.01f;
                float dx = d.x * moveScale;
                float dy = -d.y * moveScale;

                Vec3 newPos = grabStartCenter;

                if (axis == AXIS_FREE) {
                    newPos.x += right.x * dx;
                    newPos.z += right.z * dx;
                    newPos.y += dy;
                }
                else if (axis == AXIS_X) {
                    newPos.x += dx;
                }
                else if (axis == AXIS_Y) {
                    newPos.y += dy;
                }
                else if (axis == AXIS_Z) {
                    float dz = dy;
                    newPos.x += forward.x * dz;
                    newPos.z += forward.z * dz;
                }

                s->center = newPos;
            }
        }

        // ---------- EDGE GRAB güncelle ----------
        if (mode == TRANSFORM_GRAB_EDGE && g_selectedEdge >= 0) {
            EditEdge e = g_editEdges[g_selectedEdge];

            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER)) {
                mode = TRANSFORM_NONE;
                printf("EDGE GRAB onaylandi.\n");
            }
            else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                g_editVerts[e.v0].pos = edgeStartV0;
                g_editVerts[e.v1].pos = edgeStartV1;
                mode = TRANSFORM_NONE;
                printf("EDGE GRAB iptal.\n");
            }
            else {
                Vector2 m = GetMousePosition();
                Vector2 d = { m.x - edgeGrabStartMouse.x, m.y - edgeGrabStartMouse.y };

                float moveScale = 0.01f;
                float dx = d.x * moveScale;
                float dy = -d.y * moveScale;

                Vector3 delta = {0};

                if (axis == AXIS_FREE) {
                    delta.x += right.x * dx;
                    delta.z += right.z * dx;
                    delta.y += dy;
                }
                else if (axis == AXIS_X) {
                    delta.x += dx;
                }
                else if (axis == AXIS_Y) {
                    delta.y += dy;
                }
                else if (axis == AXIS_Z) {
                    float dz = dy;
                    delta.x += forward.x * dz;
                    delta.z += forward.z * dz;
                }

                g_editVerts[e.v0].pos = Vector3Add(edgeStartV0, delta);
                g_editVerts[e.v1].pos = Vector3Add(edgeStartV1, delta);
            }
        }

        // ---------- SPHERE ROTATE güncelle (sadece yaw) ----------
        if (mode == TRANSFORM_ROTATE_SPHERE && sphereCount > 0 && selectedSphere >= 0) {
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER)) {
                mode = TRANSFORM_NONE;
                printf("SPHERE ROTATE onaylandi.\n");
            }
            else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                sphereYawDeg[selectedSphere] = rotStartYawDeg;
                mode = TRANSFORM_NONE;
                printf("SPHERE ROTATE iptal.\n");
            }
            else {
                Vector2 m = GetMousePosition();
                Vector2 d = { m.x - rotStartMouse.x, m.y - rotStartMouse.y };

                float rotScale = 0.3f; // 1 pixel -> 0.3 derece
                float newYaw = rotStartYawDeg + d.x * rotScale; // yaw = ekran X
                sphereYawDeg[selectedSphere] = newYaw;
            }
        }

        // ---------- scene.txt reload / save ----------
        if (mode == TRANSFORM_NONE) {
            if (IsKeyPressed(KEY_R)) {
                sphereCount = load_scene("scene.txt", spheres, MAX_SCENE_SPHERES);
                if (sphereCount > 0) selectedSphere = 0;
                else selectedSphere = -1;
                printf("scene.txt yeniden yüklendi (%d sphere).\n", sphereCount);
            }

            if (IsKeyPressed(KEY_F5)) {
                save_scene("scene.txt", spheres, sphereCount);
            }
        }

        // ---------- Çizim ----------
        BeginDrawing();
        ClearBackground((Color){ 18, 18, 24, 255 });

        BeginMode3D(cam);

        DrawGrid(20, 1.0f);

        // Küreler
        for (int i = 0; i < sphereCount; ++i) {
            SceneSphere *s = &spheres[i];
            Vector3 pos = { s->center.x, s->center.y, s->center.z };
            Color col = ColorFromVec3(s->albedo);

            if (i == selectedSphere) {
                col = (Color){ 255, col.g, col.b, 255 };
                DrawSphereWires(pos, s->radius * 1.02f, 16, 16, (Color){255,255,255,128});
            }

            DrawSphere(pos, s->radius, col);

            // Basit yaw oku
            float yawDeg = sphereYawDeg[i];
            float yawRad = yawDeg * DEG2RAD;
            Vector3 dir = { cosf(yawRad), 0.0f, sinf(yawRad) };
            Vector3 arrowEnd = {
                pos.x + dir.x * s->radius * 1.5f,
                pos.y,
                pos.z + dir.z * s->radius * 1.5f
            };
            DrawLine3D(pos, arrowEnd, (Color){255,80,80,255});
        }

        // Edit küp vertex & edge’leri
        for (int i = 0; i < g_numEditEdges; ++i) {
            EditEdge e = g_editEdges[i];
            Vector3 a = g_editVerts[e.v0].pos;
            Vector3 b = g_editVerts[e.v1].pos;

            Color c = (i == g_selectedEdge)
                ? (Color){ 80, 220, 255, 255 }
                : (Color){ 150, 150, 200, 255 };

            DrawLine3D(a, b, c);
        }

        for (int i = 0; i < g_numEditVerts; ++i) {
            Vector3 p = g_editVerts[i].pos;
            DrawSphere(p, 0.05f, (Color){200,200,255,255});
        }

        EndMode3D();

        DrawText("YSU Edit Mode", 10, 10, 20, RAYWHITE);
        DrawText("ALT+LMB: Orbit, Wheel: Zoom", 10, 34, 16, RAYWHITE);
        DrawText("TAB: Sphere sec, N: Sphere ekle", 10, 54, 16, RAYWHITE);
        DrawText("G: Grab (once edge seciliyse edge, degilse sphere)", 10, 74, 16, RAYWHITE);
        DrawText("R: Sphere rotate (yaw)", 10, 94, 16, RAYWHITE);
        DrawText("Transform modunda X/Y/Z: eksen kilidi (G+X, G+Y, G+Z)", 10, 114, 16, RAYWHITE);
        DrawText("F5: scene.txt kaydet, R (transform yokken): scene.txt reload", 10, 134, 16, RAYWHITE);
        DrawText("Edge secmek icin: ALT yokken LMB ile kenara tikla", 10, 154, 16, RAYWHITE);

        if (g_selectedEdge >= 0) {
            DrawText("EDGE EDIT aktif: G ile tasiyabilirsin.", 10, 176, 16, (Color){80,220,255,255});
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
