#include "raylib.h"
#include "raymath.h"
#include <math.h>

int main(void)
{
    const int screenWidth  = 1280;
    const int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "YSU Realtime Viewport");
    SetTargetFPS(60);

    Vector3 target = { 0.0f, 1.0f, 0.0f };
    float distance = 6.0f;
    float yaw = 0.0f;
    float pitch = 0.35f;

    Camera3D cam = {0};
    cam.position   = (Vector3){ 0.0f, 2.0f, -distance };
    cam.target     = target;
    cam.up         = (Vector3){ 0.0f, 1.0f, 0.0f };
    cam.fovy       = 60.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    Vector2 lastMouse = {0};
    int rotating = 0;
    const float sens = 0.005f;

    while (!WindowShouldClose())
    {
        // Zoom
        float wheel = GetMouseWheelMove();
        distance -= wheel * 0.5f;
        if (distance < 1.5f) distance = 1.5f;
        if (distance > 30.0f) distance = 30.0f;

        // Orbit
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && IsKeyDown(KEY_LEFT_ALT))
        {
            Vector2 m = GetMousePosition();
            if (!rotating) { rotating = 1; lastMouse = m; }
            else
            {
                Vector2 d = { m.x - lastMouse.x, m.y - lastMouse.y };
                lastMouse = m;

                yaw   -= d.x * sens;
                pitch -= d.y * sens;

                float limit = 1.55f;
                if (pitch >  limit) pitch =  limit;
                if (pitch < -limit) pitch = -limit;
            }
        }
        else rotating = 0;

        // Update camera
        float cp = cosf(pitch);
        cam.position.x = target.x + distance * cp * cosf(yaw);
        cam.position.y = target.y + distance * sinf(pitch);
        cam.position.z = target.z + distance * cp * sinf(yaw);
        cam.target = target;

        BeginDrawing();
        ClearBackground((Color){ 18, 18, 24, 255 });

        BeginMode3D(cam);

        DrawGrid(20, 1.0f);
        DrawCube((Vector3){0,0.5f,0}, 1,1,1, BLUE);
        DrawSphere((Vector3){2,1,0}, 1, LIGHTGRAY);

        EndMode3D();

        DrawText("YSU Viewport (ALT+LMB orbit, Wheel zoom)", 10, 10, 20, RAYWHITE);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
