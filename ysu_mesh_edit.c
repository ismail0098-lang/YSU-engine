// ysu_mesh_edit.c
// YSU Mesh Edit 2.0 - Single File Mini Blender-Style Editor
// Requirements: raylib, raymath, glfw3 (via raylib), OpenGL

#include "raylib.h"
#include "raymath.h"

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

// ============================================================
// Limits & basic types
// ============================================================
#define MAX_VERTS      12000
#define MAX_TRIS       6000
#define YSU_MAX_EDGES  (MAX_TRIS * 3)

// ---------- Basic mesh types ----------
typedef struct {
    Vector3 pos;
} EditVertex;

typedef struct {
    int v[3];
} EditTri;

// ---------- Topology ----------
typedef struct {
    int v0, v1;
    int tri0;
    int tri1;
} MeshEdge;

typedef struct {
    MeshEdge edges[YSU_MAX_EDGES];
    int      edgeCount;
} MeshTopology;

// ---------- Modes ----------
typedef enum {
    SEL_VERTEX = 0,
    SEL_EDGE   = 1,
    SEL_FACE   = 2
} SelectionMode;

typedef enum {
    T_NONE    = 0,
    T_GRAB    = 1,
    T_ROTATE  = 2,
    T_SCALE   = 3,
    T_EXTRUDE = 4,
    T_INSET   = 5,
    T_BEVEL   = 6
} TransformMode;

typedef enum {
    AXIS_FREE = 0,
    AXIS_X    = 1,
    AXIS_Y    = 2,
    AXIS_Z    = 3
} TransformAxis;

typedef enum {
    MODE_EDIT     = 0,
    MODE_VIEWPORT = 1
} UIMode;

// ============================================================
// Globals
// ============================================================
static EditVertex   g_verts[MAX_VERTS];
static int          g_vertCount = 0;

static EditTri      g_tris[MAX_TRIS];
static int          g_triCount = 0;
static int          g_triHidden[MAX_TRIS];

static MeshTopology g_topology;

static Vector3      g_origin = {0};

// viewport object transform
static Vector3      vpObjPos   = {0};
static float        vpObjRotY  = 0.0f;
static float        vpObjScale = 1.0f;

// ============================================================
// Small helpers
// ============================================================
static float ClampFloat(float v, float mn, float mx)
{
    if (v < mn) return mn;
    if (v > mx) return mx;
    return v;
}

static void SwapInt(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

static void Topology_SortPair(int *a, int *b)
{
    if (*a > *b) SwapInt(a, b);
}

// Rotate vector v around 'axis' by 'angle' radians
static Vector3 RotateAroundAxis(Vector3 v, Vector3 axis, float angle)
{
    axis = Vector3Normalize(axis);
    if (Vector3LengthSqr(axis) < 1e-6f) return v;
    Quaternion q = QuaternionFromAxisAngle(axis, angle);
    return Vector3RotateByQuaternion(v, q);
}

// ============================================================
// Topology
// ============================================================
static void Topology_Build(MeshTopology *topo,
                           EditTri *tris,
                           int triCount,
                           int vertCount)
{
    (void)vertCount;
    topo->edgeCount = 0;

    for (int ti = 0; ti < triCount; ++ti) {
        EditTri *t = &tris[ti];
        int v[3] = { t->v[0], t->v[1], t->v[2] };

        for (int e = 0; e < 3; ++e) {
            int a = v[e];
            int b = v[(e + 1) % 3];
            if (a < 0 || b < 0) continue;

            Topology_SortPair(&a, &b);

            int found = -1;
            for (int ei = 0; ei < topo->edgeCount; ++ei) {
                MeshEdge *edge = &topo->edges[ei];
                if (edge->v0 == a && edge->v1 == b) {
                    found = ei;
                    break;
                }
            }

            if (found >= 0) {
                MeshEdge *edge = &topo->edges[found];
                if (edge->tri0 == -1)      edge->tri0 = ti;
                else if (edge->tri1 == -1) edge->tri1 = ti;
            } else {
                if (topo->edgeCount >= YSU_MAX_EDGES) continue;
                MeshEdge *edge = &topo->edges[topo->edgeCount++];
                edge->v0   = a;
                edge->v1   = b;
                edge->tri0 = ti;
                edge->tri1 = -1;
            }
        }
    }
}

static int Topology_FindEdge(const MeshTopology *topo, int v0, int v1)
{
    Topology_SortPair(&v0, &v1);
    for (int i = 0; i < topo->edgeCount; ++i) {
        const MeshEdge *e = &topo->edges[i];
        if (e->v0 == v0 && e->v1 == v1) return i;
    }
    return -1;
}

// ============================================================
// Mesh helpers
// ============================================================
static void ClearMesh(void)
{
    g_vertCount = 0;
    g_triCount  = 0;
    for (int i = 0; i < MAX_TRIS; ++i) g_triHidden[i] = 0;
    g_origin    = (Vector3){0,0,0};
    vpObjPos    = (Vector3){0,0,0};
    vpObjRotY   = 0.0f;
    vpObjScale  = 1.0f;
}

static int AddVertex(Vector3 p)
{
    if (g_vertCount >= MAX_VERTS) return -1;
    g_verts[g_vertCount].pos = p;
    return g_vertCount++;
}

static int AddTri(int i0, int i1, int i2)
{
    if (g_triCount >= MAX_TRIS) return -1;
    g_tris[g_triCount].v[0] = i0;
    g_tris[g_triCount].v[1] = i1;
    g_tris[g_triCount].v[2] = i2;
    g_triHidden[g_triCount] = 0;
    return g_triCount++;
}

// ============================================================
// Primitive builders
// ============================================================
static void CreateCubeMesh(void)
{
    ClearMesh();

    int v000 = AddVertex((Vector3){-1,-1,-1});
    int v001 = AddVertex((Vector3){-1,-1, 1});
    int v010 = AddVertex((Vector3){-1, 1,-1});
    int v011 = AddVertex((Vector3){-1, 1, 1});
    int v100 = AddVertex((Vector3){ 1,-1,-1});
    int v101 = AddVertex((Vector3){ 1,-1, 1});
    int v110 = AddVertex((Vector3){ 1, 1,-1});
    int v111 = AddVertex((Vector3){ 1, 1, 1});

    // front
    AddTri(v001, v101, v111);
    AddTri(v001, v111, v011);
    // back
    AddTri(v100, v000, v010);
    AddTri(v100, v010, v110);
    // left
    AddTri(v000, v001, v011);
    AddTri(v000, v011, v010);
    // right
    AddTri(v101, v100, v110);
    AddTri(v101, v110, v111);
    // top
    AddTri(v010, v011, v111);
    AddTri(v010, v111, v110);
    // bottom
    AddTri(v000, v100, v101);
    AddTri(v000, v101, v001);
}

static void CreateSphereMesh(void)
{
    ClearMesh();

    const int seg   = 20;
    const int rings = 20;
    const float r   = 1.0f;

    int idx[rings+1][seg];

    for (int y = 0; y <= rings; ++y) {
        float v = (float)y / (float)rings;
        float phi = v * PI;

        for (int x = 0; x < seg; ++x) {
            float u = (float)x / (float)seg;
            float theta = u * 2.0f * PI;

            Vector3 p = {
                r * sinf(phi) * cosf(theta),
                r * cosf(phi),
                r * sinf(phi) * sinf(theta)
            };
            idx[y][x] = AddVertex(p);
        }
    }

    for (int y = 0; y < rings; ++y) {
        for (int x = 0; x < seg; ++x) {
            int x1 = (x + 1) % seg;

            int v00 = idx[y][x];
            int v01 = idx[y][x1];
            int v10 = idx[y+1][x];
            int v11 = idx[y+1][x1];

            AddTri(v00, v10, v11);
            AddTri(v00, v11, v01);
        }
    }
}

static void CreateCylinderMesh(void)
{
    ClearMesh();

    const int seg = 20;
    const float h = 2.0f;
    const float r = 1.0f;

    int bottom[seg];
    int top[seg];

    for (int i = 0; i < seg; ++i) {
        float a = (float)i / (float)seg * 2.0f * PI;
        float cx = r * cosf(a);
        float cz = r * sinf(a);

        bottom[i] = AddVertex((Vector3){cx, -h*0.5f, cz});
        top[i]    = AddVertex((Vector3){cx,  h*0.5f, cz});
    }

    int centerBottom = AddVertex((Vector3){0.0f, -h*0.5f, 0.0f});
    int centerTop    = AddVertex((Vector3){0.0f,  h*0.5f, 0.0f});

    for (int i = 0; i < seg; ++i) {
        int i1 = (i+1) % seg;

        int b0 = bottom[i];
        int b1 = bottom[i1];
        int t0 = top[i];
        int t1 = top[i1];

        // side
        AddTri(b0, t0, t1);
        AddTri(b0, t1, b1);

        // bottom cap
        AddTri(centerBottom, b1, b0);
        // top cap
        AddTri(centerTop, t0, t1);
    }
}

// ============================================================
// Selection helpers
// ============================================================
static Vector3 ComputePivot(int triIndex, SelectionMode mode, int selIndex)
{
    if (g_triCount <= 0) return (Vector3){0,0,0};
    if (triIndex < 0 || triIndex >= g_triCount) triIndex = 0;
    EditTri *t = &g_tris[triIndex];

    Vector3 a = g_verts[t->v[0]].pos;
    Vector3 b = g_verts[t->v[1]].pos;
    Vector3 c = g_verts[t->v[2]].pos;

    if (mode == SEL_VERTEX) {
        int vi = t->v[selIndex % 3];
        return g_verts[vi].pos;
    }
    if (mode == SEL_EDGE) {
        Vector3 e0, e1;
        if      (selIndex == 0) { e0 = a; e1 = b; }
        else if (selIndex == 1) { e0 = b; e1 = c; }
        else                    { e0 = c; e1 = a; }
        return (Vector3){
            (e0.x + e1.x)*0.5f,
            (e0.y + e1.y)*0.5f,
            (e0.z + e1.z)*0.5f
        };
    }
    // face center
    return (Vector3){
        (a.x + b.x + c.x)/3.0f,
        (a.y + b.y + c.y)/3.0f,
        (a.z + b.z + c.z)/3.0f
    };
}

static void BuildSelectionMask(int *mask,
                               SelectionMode mode,
                               int triIndex,
                               int selIndex)
{
    for (int i = 0; i < g_vertCount; ++i) mask[i] = 0;
    if (g_triCount <= 0) return;
    if (triIndex < 0 || triIndex >= g_triCount) return;

    EditTri *t = &g_tris[triIndex];

    if (mode == SEL_VERTEX) {
        int v = t->v[selIndex % 3];
        if (v >= 0 && v < g_vertCount) mask[v] = 1;
        return;
    }
    if (mode == SEL_EDGE) {
        int a = -1, b = -1;
        if      (selIndex == 0) { a = t->v[0]; b = t->v[1]; }
        else if (selIndex == 1) { a = t->v[1]; b = t->v[2]; }
        else                    { a = t->v[2]; b = t->v[0]; }

        if (a >= 0 && a < g_vertCount) mask[a] = 1;
        if (b >= 0 && b < g_vertCount) mask[b] = 1;
        return;
    }
    // face
    for (int i = 0; i < 3; ++i) {
        int v = t->v[i];
        if (v >= 0 && v < g_vertCount) mask[v] = 1;
    }
}

// ============================================================
// Ray-triangle
// ============================================================
static bool RayIntersectsTriangle(Ray ray, Vector3 v0, Vector3 v1, Vector3 v2)
{
    const float EPS = 1e-6f;

    Vector3 e1 = Vector3Subtract(v1, v0);
    Vector3 e2 = Vector3Subtract(v2, v0);

    Vector3 p = Vector3CrossProduct(ray.direction, e2);
    float det = Vector3DotProduct(e1, p);
    if (det > -EPS && det < EPS) return false;

    float invDet = 1.0f / det;
    Vector3 tvec = Vector3Subtract(ray.position, v0);
    float u = Vector3DotProduct(tvec, p) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    Vector3 q = Vector3CrossProduct(tvec, e1);
    float v = Vector3DotProduct(tvec, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = Vector3DotProduct(e2, q) * invDet;
    if (t < 0.0f) return false;
    return true;
}

// ============================================================
// Viewport transform
// ============================================================
static Vector3 VpTransformPoint(Vector3 p,
                                Vector3 pos,
                                float rotY,
                                float scale)
{
    Vector3 ps = Vector3Scale(p, scale);
    float c = cosf(rotY);
    float s = sinf(rotY);
    Vector3 r = {
        ps.x * c + ps.z * s,
        ps.y,
        -ps.x * s + ps.z * c
    };
    return Vector3Add(r, pos);
}

// ============================================================
// Extrude / Inset / Face Bevel state
// ============================================================
static int     extrudeOldVertCount = 0;
static int     extrudeOldTriCount  = 0;
static int     extrudeNewVerts[3]  = { -1,-1,-1 };
static int     extrudeBaseTri      = -1;
static int     extrudeTopTri       = -1;
static Vector3 extrudeBasePos[3];
static Vector3 extrudeNormal;
static Vector2 extrudeStartMouse;

static int     insetOldVertCount   = 0;
static int     insetOldTriCount    = 0;
static int     insetNewVerts[3]    = { -1,-1,-1 };
static int     insetBaseTri        = -1;
static int     insetInnerTri       = -1;
static Vector3 insetBasePos[3];
static Vector3 insetCenter;
static Vector2 insetStartMouse;

static int     bevelOldVertCount   = 0;
static int     bevelOldTriCount    = 0;
static int     bevelBaseTri        = -1;
static Vector3 bevelA, bevelB, bevelC;
static int     bevelCorner[3][2];
static int     bevelCenterIndex    = -1;
static Vector2 bevelStartMouse;

// ============================================================
// Extrude / Inset / Face Bevel helpers
// ============================================================
static int StartExtrudeFace(int triIndex)
{
    if (g_triCount <= 0) return 0;
    if (triIndex < 0 || triIndex >= g_triCount) return 0;

    EditTri base = g_tris[triIndex];
    int ia = base.v[0];
    int ib = base.v[1];
    int ic = base.v[2];

    Vector3 pa = g_verts[ia].pos;
    Vector3 pb = g_verts[ib].pos;
    Vector3 pc = g_verts[ic].pos;

    Vector3 e1 = Vector3Subtract(pb, pa);
    Vector3 e2 = Vector3Subtract(pc, pa);
    Vector3 n  = Vector3Normalize(Vector3CrossProduct(e1, e2));
    if (Vector3LengthSqr(n) < 1e-6f) return 0;

    extrudeOldVertCount = g_vertCount;
    extrudeOldTriCount  = g_triCount;
    extrudeBaseTri      = triIndex;
    extrudeNormal       = n;

    int ia2 = AddVertex(pa);
    int ib2 = AddVertex(pb);
    int ic2 = AddVertex(pc);
    if (ia2 < 0 || ib2 < 0 || ic2 < 0) {
        g_vertCount = extrudeOldVertCount;
        g_triCount  = extrudeOldTriCount;
        return 0;
    }

    extrudeNewVerts[0] = ia2;
    extrudeNewVerts[1] = ib2;
    extrudeNewVerts[2] = ic2;

    extrudeBasePos[0] = pa;
    extrudeBasePos[1] = pb;
    extrudeBasePos[2] = pc;

    g_triHidden[triIndex] = 1;

    extrudeTopTri = AddTri(ia2, ib2, ic2);

    AddTri(ia, ib, ib2);
    AddTri(ia, ib2, ia2);

    AddTri(ib, ic, ic2);
    AddTri(ib, ic2, ib2);

    AddTri(ic, ia, ia2);
    AddTri(ic, ia2, ic2);

    return 1;
}

static int StartInsetFace(int triIndex)
{
    if (g_triCount <= 0) return 0;
    if (triIndex < 0 || triIndex >= g_triCount) return 0;

    EditTri base = g_tris[triIndex];
    int ia = base.v[0];
    int ib = base.v[1];
    int ic = base.v[2];

    Vector3 pa = g_verts[ia].pos;
    Vector3 pb = g_verts[ib].pos;
    Vector3 pc = g_verts[ic].pos;

    Vector3 center = {
        (pa.x + pb.x + pc.x)/3.0f,
        (pa.y + pb.y + pc.y)/3.0f,
        (pa.z + pb.z + pc.z)/3.0f
    };

    insetOldVertCount = g_vertCount;
    insetOldTriCount  = g_triCount;
    insetBaseTri      = triIndex;
    insetCenter       = center;

    insetBasePos[0] = pa;
    insetBasePos[1] = pb;
    insetBasePos[2] = pc;

    int ia2 = AddVertex(pa);
    int ib2 = AddVertex(pb);
    int ic2 = AddVertex(pc);

    if (ia2 < 0 || ib2 < 0 || ic2 < 0) {
        g_vertCount = insetOldVertCount;
        g_triCount  = insetOldTriCount;
        return 0;
    }

    insetNewVerts[0] = ia2;
    insetNewVerts[1] = ib2;
    insetNewVerts[2] = ic2;

    insetInnerTri = AddTri(ia2, ib2, ic2);

    AddTri(ia, ib, ib2);
    AddTri(ia, ib2, ia2);
    AddTri(ib, ic, ic2);
    AddTri(ib, ic2, ib2);
    AddTri(ic, ia, ia2);
    AddTri(ic, ia2, ic2);

    return 1;
}

static void UpdateBevelGeometry(float f)
{
    if (bevelCenterIndex < 0) return;

    f = ClampFloat(f, 0.05f, 0.45f);

    Vector3 Aab = Vector3Lerp(bevelA, bevelB, f);
    Vector3 Aac = Vector3Lerp(bevelA, bevelC, f);
    Vector3 Bbc = Vector3Lerp(bevelB, bevelC, f);
    Vector3 Bba = Vector3Lerp(bevelB, bevelA, f);
    Vector3 Cca = Vector3Lerp(bevelC, bevelA, f);
    Vector3 Ccb = Vector3Lerp(bevelC, bevelB, f);

    g_verts[bevelCorner[0][0]].pos = Aab;
    g_verts[bevelCorner[0][1]].pos = Aac;
    g_verts[bevelCorner[1][0]].pos = Bbc;
    g_verts[bevelCorner[1][1]].pos = Bba;
    g_verts[bevelCorner[2][0]].pos = Cca;
    g_verts[bevelCorner[2][1]].pos = Ccb;

    Vector3 center = {
        (Aab.x + Aac.x + Bbc.x + Bba.x + Cca.x + Ccb.x)/6.0f,
        (Aab.y + Aac.y + Bbc.y + Bba.y + Cca.y + Ccb.y)/6.0f,
        (Aab.z + Aac.z + Bbc.z + Bba.z + Cca.z + Ccb.z)/6.0f
    };
    g_verts[bevelCenterIndex].pos = center;
}

static int StartBevelFace(int triIndex)
{
    if (g_triCount <= 0) return 0;
    if (triIndex < 0 || triIndex >= g_triCount) return 0;

    EditTri base = g_tris[triIndex];
    int ia = base.v[0];
    int ib = base.v[1];
    int ic = base.v[2];

    bevelA = g_verts[ia].pos;
    bevelB = g_verts[ib].pos;
    bevelC = g_verts[ic].pos;

    bevelOldVertCount = g_vertCount;
    bevelOldTriCount  = g_triCount;
    bevelBaseTri      = triIndex;

    int Aab = AddVertex((Vector3){0});
    int Aac = AddVertex((Vector3){0});
    int Bbc = AddVertex((Vector3){0});
    int Bba = AddVertex((Vector3){0});
    int Cca = AddVertex((Vector3){0});
    int Ccb = AddVertex((Vector3){0});
    int Ctr = AddVertex((Vector3){0});

    if (Aab < 0 || Aac < 0 || Bbc < 0 || Bba < 0 ||
        Cca < 0 || Ccb < 0 || Ctr < 0)
    {
        g_vertCount = bevelOldVertCount;
        g_triCount  = bevelOldTriCount;
        return 0;
    }

    bevelCorner[0][0] = Aab;
    bevelCorner[0][1] = Aac;
    bevelCorner[1][0] = Bbc;
    bevelCorner[1][1] = Bba;
    bevelCorner[2][0] = Cca;
    bevelCorner[2][1] = Ccb;
    bevelCenterIndex  = Ctr;

    g_triHidden[triIndex] = 1;

    AddTri(ia, Aab, Aac);
    AddTri(ib, Bbc, Bba);
    AddTri(ic, Cca, Ccb);

    AddTri(Aab, Bba, Ctr);
    AddTri(Bba, Bbc, Ctr);
    AddTri(Bbc, Ccb, Ctr);
    AddTri(Ccb, Cca, Ctr);
    AddTri(Cca, Aac, Ctr);
    AddTri(Aac, Aab, Ctr);

    UpdateBevelGeometry(0.25f);
    return 1;
}

// ============================================================
// Edge bevel (rounded chamfer v1)
// ============================================================
static int Mesh_BevelEdgeRounded(const MeshTopology *topo,
                                 int edgeIndex,
                                 int segments,
                                 float radiusScale)
{
    if (!topo) return 0;
    if (edgeIndex < 0 || edgeIndex >= topo->edgeCount) return 0;

    if (segments < 1) segments = 1;
    if (segments > 6) segments = 6;
    radiusScale = ClampFloat(radiusScale, 0.05f, 0.4f);

    MeshEdge e = topo->edges[edgeIndex];
    if (e.tri0 < 0 || e.tri1 < 0) return 0; // must have 2 faces

    int v0 = e.v0;
    int v1 = e.v1;
    if (v0 < 0 || v0 >= g_vertCount) return 0;
    if (v1 < 0 || v1 >= g_vertCount) return 0;

    EditTri *t0 = &g_tris[e.tri0];
    EditTri *t1 = &g_tris[e.tri1];

    int c = -1, d = -1;
    for (int i = 0; i < 3; ++i) {
        int vi = t0->v[i];
        if (vi != v0 && vi != v1) { c = vi; break; }
    }
    for (int i = 0; i < 3; ++i) {
        int vi = t1->v[i];
        if (vi != v0 && vi != v1) { d = vi; break; }
    }
    if (c < 0 || d < 0) return 0;

    Vector3 A = g_verts[v0].pos;
    Vector3 B = g_verts[v1].pos;
    Vector3 C = g_verts[c].pos;
    Vector3 D = g_verts[d].pos;

    Vector3 n0 = Vector3Normalize(Vector3CrossProduct(
                       Vector3Subtract(B, A),
                       Vector3Subtract(C, A)));
    Vector3 n1 = Vector3Normalize(Vector3CrossProduct(
                       Vector3Subtract(A, B),
                       Vector3Subtract(D, B)));
    if (Vector3LengthSqr(n0) < 1e-6f ||
        Vector3LengthSqr(n1) < 1e-6f)
        return 0;

    Vector3 axis = Vector3Normalize(Vector3Subtract(B, A));
    if (Vector3LengthSqr(axis) < 1e-6f) return 0;

    Vector3 d0 = Vector3Subtract(n0, Vector3Scale(axis, Vector3DotProduct(n0, axis)));
    Vector3 d1 = Vector3Subtract(n1, Vector3Scale(axis, Vector3DotProduct(n1, axis)));
    if (Vector3LengthSqr(d0) < 1e-6f ||
        Vector3LengthSqr(d1) < 1e-6f)
        return 0;
    d0 = Vector3Normalize(d0);
    d1 = Vector3Normalize(d1);

    float dot = Vector3DotProduct(d0, d1);
    dot = ClampFloat(dot, -0.9999f, 0.9999f);
    float angleTotal = acosf(dot);
    if (fabsf(angleTotal) < 1e-3f) return 0;

    float lenAC = Vector3Length(Vector3Subtract(C, A));
    float lenBC = Vector3Length(Vector3Subtract(C, B));
    float lenAD = Vector3Length(Vector3Subtract(D, A));
    float lenBD = Vector3Length(Vector3Subtract(D, B));
    float minLen = lenAC;
    if (lenBC < minLen) minLen = lenBC;
    if (lenAD < minLen) minLen = lenAD;
    if (lenBD < minLen) minLen = lenBD;
    float radius = minLen * radiusScale;
    if (radius < 1e-4f) radius = minLen * 0.1f;

    int neededVerts = g_vertCount + (segments+1)*2;
    if (neededVerts > MAX_VERTS) return 0;
    int neededTris = g_triCount + segments*6;
    if (neededTris > MAX_TRIS) return 0;

    int ai[8];
    int bi[8];

    for (int s = 0; s <= segments; ++s) {
        float t = (float)s / (float)segments;
        float ang = angleTotal * t;
        Vector3 dir = RotateAroundAxis(d0, axis, ang);
        Vector3 off = Vector3Scale(dir, radius);

        int va = g_vertCount++;
        int vb = g_vertCount++;
        g_verts[va].pos = Vector3Add(A, off);
        g_verts[vb].pos = Vector3Add(B, off);
        ai[s] = va;
        bi[s] = vb;
    }

    g_triHidden[e.tri0] = 1;
    g_triHidden[e.tri1] = 1;

    for (int s = 0; s < segments; ++s) {
        int a0 = ai[s];
        int a1 = ai[s+1];
        int b0 = bi[s];
        int b1 = bi[s+1];

        AddTri(a0, b0, b1);
        AddTri(a0, b1, a1);
    }

    for (int s = 0; s < segments; ++s) {
        int a0 = ai[s];
        int a1 = ai[s+1];
        int b0 = bi[s];
        int b1 = bi[s+1];

        AddTri(v0, a0, a1);
        AddTri(v1, b1, b0);
    }

    for (int s = 0; s < segments; ++s) {
        int a0 = ai[s];
        int a1 = ai[s+1];
        int b0 = bi[s];
        int b1 = bi[s+1];

        AddTri(v0, d, a0);
        AddTri(v0, a1, d);
        AddTri(v1, b0, d);
        AddTri(v1, d, b1);
    }

    return 1;
}

// ============================================================
// OBJ Exporter (Shift+E)
// ============================================================
static int ExportOBJ(const char *path,
                     EditVertex *verts, int vertCount,
                     EditTri *tris,   int triCount)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    fprintf(f, "# Exported from YSU Mesh Edit\n");

    for (int i = 0; i < vertCount; ++i) {
        Vector3 p = verts[i].pos;
        fprintf(f, "v %f %f %f\n", p.x, p.y, p.z);
    }
    for (int i = 0; i < triCount; ++i) {
        if (g_triHidden[i]) continue;
        int a = tris[i].v[0] + 1;
        int b = tris[i].v[1] + 1;
        int c = tris[i].v[2] + 1;
        fprintf(f, "f %d %d %d\n", a, b, c);
    }

    fclose(f);
    return 1;
}

// ============================================================
// OBJ Importer (Shift+O -> import.obj)
// ============================================================
static int ImportOBJ(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    ClearMesh();

    char line[512];

    while (fgets(line, sizeof(line), f)) {
        // Vertex
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            float x, y, z;
            if (sscanf(line + 1, "%f %f %f", &x, &y, &z) == 3) {
                AddVertex((Vector3){x, y, z});
            }
        }
        // Face
        else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            int indices[8];
            int idxCount = 0;

            char *p = line + 1;
            while (*p && idxCount < 8) {
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '\0' || *p == '\n' || *p == '\r')
                    break;
                if ((*p < '0' || *p > '9') && *p != '-') break;

                int vIdx = atoi(p); // "12/..." -> 12
                indices[idxCount++] = vIdx;

                while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
                    p++;
            }

            if (idxCount >= 3) {
                for (int i = 1; i < idxCount - 1; ++i) {
                    int a = indices[0] - 1;
                    int b = indices[i] - 1;
                    int c = indices[i+1] - 1;
                    if (a < 0 || b < 0 || c < 0) continue;
                    if (a >= g_vertCount || b >= g_vertCount || c >= g_vertCount) continue;
                    AddTri(a, b, c);
                }
            }
        }
    }

    fclose(f);
    return (g_vertCount > 0 && g_triCount > 0);
}

// ============================================================
// main()
// ============================================================
int main(void)
{
    const int W = 1280;
    const int H = 720;

    InitWindow(W, H, "YSU Mesh Edit 2.0");
    SetTargetFPS(60);

    CreateCubeMesh();

    UIMode        uiMode   = MODE_EDIT;
    SelectionMode selMode  = SEL_VERTEX;
    int           selTri   = 0;
    int           selIndex = 0;

    TransformMode tmode = T_NONE;
    TransformAxis axis  = AXIS_FREE;

    Vector2 grabStartMouse = (Vector2){0};
    Vector3 grabStartPos[MAX_VERTS];

    Vector2 rotStartMouse = (Vector2){0};
    Vector3 rotStartPos[MAX_VERTS];
    Vector3 rotPivot = (Vector3){0};

    Vector2 scaleStartMouse = (Vector2){0};
    Vector3 scaleStartPos[MAX_VERTS];
    Vector3 scalePivot = (Vector3){0};

    int vertSelected[MAX_VERTS];

    int  showAddMenu = 0;

    Vector3 target = (Vector3){0.0f, 0.5f, 0.0f};
    float  dist    = 6.0f;
    float  yaw     = 0.0f;
    float  pitch   = 0.35f;

    Camera3D cam = {0};
    cam.fovy       = 60.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    Vector2 lastMouse = (Vector2){0};
    int     rotating  = 0;
    float   sens      = 0.005f;

    int showExportMessage = 0;
    int exportMessageFrames = 0;

    while (!WindowShouldClose())
    {
        Topology_Build(&g_topology, g_tris, g_triCount, g_vertCount);

        float wheel = GetMouseWheelMove();
        dist -= wheel * 0.5f;
        dist = ClampFloat(dist, 1.5f, 40.0f);

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && IsKeyDown(KEY_LEFT_ALT)) {
            Vector2 m = GetMousePosition();
            if (!rotating) {
                rotating  = 1;
                lastMouse = m;
            } else {
                Vector2 d = { m.x - lastMouse.x, m.y - lastMouse.y };
                lastMouse = m;
                yaw   -= d.x * sens;
                pitch -= d.y * sens;
                pitch = ClampFloat(pitch, -1.55f, 1.55f);
            }
        } else rotating = 0;

        cam.position.x = target.x + dist * cosf(pitch) * cosf(yaw);
        cam.position.y = target.y + dist * sinf(pitch);
        cam.position.z = target.z + dist * cosf(pitch) * sinf(yaw);
        cam.target = target;
        cam.up     = (Vector3){0,1,0};

        if (IsKeyPressed(KEY_F1)) { uiMode = MODE_EDIT;     tmode = T_NONE; }
        if (IsKeyPressed(KEY_F2)) { uiMode = MODE_VIEWPORT; tmode = T_NONE; }

        // Export OBJ (Shift+E)
        if (IsKeyDown(KEY_LEFT_SHIFT) && IsKeyPressed(KEY_E)) {
            if (ExportOBJ("export.obj", g_verts, g_vertCount, g_tris, g_triCount)) {
                showExportMessage = 1;
                exportMessageFrames = 120;
            }
        }

        // Import OBJ (Shift+O)
        if (IsKeyDown(KEY_LEFT_SHIFT) && IsKeyPressed(KEY_O)) {
            if (ImportOBJ("import.obj")) {
                selTri = 0;
                selIndex = 0;
                selMode = SEL_FACE;
            }
        }

        if (exportMessageFrames > 0) {
            exportMessageFrames--;
            if (exportMessageFrames <= 0) showExportMessage = 0;
        }

        // =====================================================
        // EDIT MODE
        // =====================================================
        if (uiMode == MODE_EDIT)
        {
            // Add mesh menu
            if (tmode == T_NONE) {
                if (IsKeyDown(KEY_LEFT_SHIFT) && IsKeyPressed(KEY_A)) {
                    showAddMenu = !showAddMenu;
                }
                if (showAddMenu) {
                    if (IsKeyPressed(KEY_ONE)) {
                        CreateCubeMesh();
                        selTri = 0; selIndex = 0; selMode = SEL_VERTEX;
                        showAddMenu = 0;
                    }
                    if (IsKeyPressed(KEY_TWO)) {
                        CreateSphereMesh();
                        selTri = 0; selIndex = 0; selMode = SEL_VERTEX;
                        showAddMenu = 0;
                    }
                    if (IsKeyPressed(KEY_THREE)) {
                        CreateCylinderMesh();
                        selTri = 0; selIndex = 0; selMode = SEL_VERTEX;
                        showAddMenu = 0;
                    }
                }
            }

            // Face pick (LMB) in FACE mode
            if (tmode == T_NONE &&
                selMode == SEL_FACE &&
                !IsKeyDown(KEY_LEFT_ALT) &&
                IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            {
                Ray ray = GetMouseRay(GetMousePosition(), cam);
                float bestDist2 = 1e30f;
                int bestTri = -1;

                for (int t = 0; t < g_triCount; ++t) {
                    if (g_triHidden[t]) continue;
                    EditTri *tri = &g_tris[t];
                    Vector3 v0 = g_verts[tri->v[0]].pos;
                    Vector3 v1 = g_verts[tri->v[1]].pos;
                    Vector3 v2 = g_verts[tri->v[2]].pos;

                    if (RayIntersectsTriangle(ray, v0, v1, v2)) {
                        Vector3 c = {
                            (v0.x + v1.x + v2.x)/3.0f,
                            (v0.y + v1.y + v2.y)/3.0f,
                            (v0.z + v1.z + v2.z)/3.0f
                        };
                        float d2 = Vector3LengthSqr(Vector3Subtract(c, cam.position));
                        if (d2 < bestDist2) {
                            bestDist2 = d2;
                            bestTri   = t;
                        }
                    }
                }

                if (bestTri >= 0) { selTri = bestTri; selIndex = 0; }
            }

            // --- Input when no transform running ---
            if (tmode == T_NONE) {
                if (IsKeyPressed(KEY_ONE))   { selMode = SEL_VERTEX; selIndex = 0; }
                if (IsKeyPressed(KEY_TWO))   { selMode = SEL_EDGE;   selIndex = 0; }
                if (IsKeyPressed(KEY_THREE)) { selMode = SEL_FACE; }

                if (g_triCount > 0) {
                    if (IsKeyPressed(KEY_TAB))
                        selTri = (selTri + 1) % g_triCount;
                    if (selMode != SEL_FACE && IsKeyPressed(KEY_E))
                        selIndex = (selIndex + 1) % 3;
                }

                if (IsKeyPressed(KEY_O) && g_triCount > 0) {
                    g_origin = ComputePivot(selTri, selMode, selIndex);
                }

                if (IsKeyPressed(KEY_M) && g_triCount > 0) {
                    EditTri *t = &g_tris[selTri];
                    int ia = t->v[0];
                    int ib = t->v[1];
                    int ic = t->v[2];

                    if (selMode == SEL_EDGE) {
                        int a,b;
                        if      (selIndex == 0) { a = ia; b = ib; }
                        else if (selIndex == 1) { a = ib; b = ic; }
                        else                    { a = ic; b = ia; }

                        Vector3 pa = g_verts[a].pos;
                        Vector3 pb = g_verts[b].pos;
                        Vector3 mid = {
                            (pa.x + pb.x)*0.5f,
                            (pa.y + pb.y)*0.5f,
                            (pa.z + pb.z)*0.5f
                        };
                        g_verts[a].pos = mid;
                        g_verts[b].pos = mid;
                    } else if (selMode == SEL_FACE) {
                        Vector3 pa = g_verts[ia].pos;
                        Vector3 pb = g_verts[ib].pos;
                        Vector3 pc = g_verts[ic].pos;
                        Vector3 mid = {
                            (pa.x + pb.x + pc.x)/3.0f,
                            (pa.y + pb.y + pc.y)/3.0f,
                            (pa.z + pb.z + pc.z)/3.0f
                        };
                        g_verts[ia].pos = mid;
                        g_verts[ib].pos = mid;
                        g_verts[ic].pos = mid;
                    }
                }

                // Bevel: face (interaktif) ve edge (chamfer)
                bool pressedB = IsKeyPressed(KEY_B) ||
                                (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_B));

                if (pressedB && g_triCount > 0) {
                    if (selMode == SEL_FACE) {
                        if (StartBevelFace(selTri)) {
                            tmode = T_BEVEL;
                            axis  = AXIS_FREE;
                            bevelStartMouse = GetMousePosition();
                        }
                    } else if (selMode == SEL_EDGE) {
                        EditTri *t = &g_tris[selTri];
                        int a,b;
                        if      (selIndex == 0) { a = t->v[0]; b = t->v[1]; }
                        else if (selIndex == 1) { a = t->v[1]; b = t->v[2]; }
                        else                    { a = t->v[2]; b = t->v[0]; }

                        int edgeIdx = Topology_FindEdge(&g_topology, a, b);
                        if (edgeIdx >= 0) {
                            Mesh_BevelEdgeRounded(&g_topology, edgeIdx, 3, 0.25f);
                        }
                    }
                }

                // Extrude (F, face)
                if (IsKeyPressed(KEY_F) && selMode == SEL_FACE && g_triCount > 0) {
                    if (StartExtrudeFace(selTri)) {
                        tmode = T_EXTRUDE;
                        axis  = AXIS_FREE;
                        extrudeStartMouse = GetMousePosition();
                        selTri   = extrudeTopTri;
                        selIndex = 0;
                    }
                }

                // Inset (I, face)
                if (IsKeyPressed(KEY_I) && selMode == SEL_FACE && g_triCount > 0) {
                    if (StartInsetFace(selTri)) {
                        tmode = T_INSET;
                        axis  = AXIS_FREE;
                        insetStartMouse = GetMousePosition();
                        selTri   = insetInnerTri;
                        selIndex = 0;
                    }
                }

                // G / R / S
                if (IsKeyPressed(KEY_G) && g_vertCount > 0 && g_triCount > 0) {
                    tmode = T_GRAB;
                    axis  = AXIS_FREE;
                    grabStartMouse = GetMousePosition();
                    for (int i = 0; i < g_vertCount; ++i)
                        grabStartPos[i] = g_verts[i].pos;
                    BuildSelectionMask(vertSelected, selMode, selTri, selIndex);
                }

                if (IsKeyPressed(KEY_R) && g_vertCount > 0 && g_triCount > 0) {
                    tmode = T_ROTATE;
                    axis  = AXIS_FREE;
                    rotStartMouse = GetMousePosition();
                    for (int i = 0; i < g_vertCount; ++i)
                        rotStartPos[i] = g_verts[i].pos;
                    rotPivot = ComputePivot(selTri, selMode, selIndex);
                    BuildSelectionMask(vertSelected, selMode, selTri, selIndex);
                }

                if (IsKeyPressed(KEY_S) && g_vertCount > 0 && g_triCount > 0) {
                    tmode = T_SCALE;
                    axis  = AXIS_FREE;
                    scaleStartMouse = GetMousePosition();
                    for (int i = 0; i < g_vertCount; ++i)
                        scaleStartPos[i] = g_verts[i].pos;
                    scalePivot = ComputePivot(selTri, selMode, selIndex);
                    BuildSelectionMask(vertSelected, selMode, selTri, selIndex);
                }
            }

            if (tmode != T_NONE) {
                if (IsKeyPressed(KEY_X)) axis = AXIS_X;
                if (IsKeyPressed(KEY_Y)) axis = AXIS_Y;
                if (IsKeyPressed(KEY_Z)) axis = AXIS_Z;
            }

            // GRAB
            if (tmode == T_GRAB) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    for (int i = 0; i < g_vertCount; ++i)
                        g_verts[i].pos = grabStartPos[i];
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    Vector2 d = { m.x - grabStartMouse.x, m.y - grabStartMouse.y };
                    float dx = d.x * 0.01f;
                    float dy = -d.y * 0.01f;

                    Vector3 forward = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
                    Vector3 right   = Vector3Normalize(Vector3CrossProduct(forward, (Vector3){0,1,0}));
                    Vector3 up      = (Vector3){0,1,0};

                    Vector3 off = {0};

                    if (axis == AXIS_FREE) {
                        off.x += right.x * dx;
                        off.z += right.z * dx;
                        off.x += up.x * dy;
                        off.y += up.y * dy;
                        off.z += up.z * dy;
                    } else if (axis == AXIS_X) {
                        off.x += dx;
                    } else if (axis == AXIS_Y) {
                        off.y += dy;
                    } else if (axis == AXIS_Z) {
                        off.x += forward.x * dy;
                        off.z += forward.z * dy;
                    }

                    for (int i = 0; i < g_vertCount; ++i) {
                        if (vertSelected[i]) {
                            Vector3 base = grabStartPos[i];
                            g_verts[i].pos = (Vector3){
                                base.x + off.x,
                                base.y + off.y,
                                base.z + off.z
                            };
                        } else {
                            g_verts[i].pos = grabStartPos[i];
                        }
                    }
                }
            }

            // ROTATE
            if (tmode == T_ROTATE) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    for (int i = 0; i < g_vertCount; ++i)
                        g_verts[i].pos = rotStartPos[i];
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dx = (m.x - rotStartMouse.x) * 0.01f;

                    Vector3 ax = {0,0,0};
                    if (axis == AXIS_FREE || axis == AXIS_Y) ax = (Vector3){0,1,0};
                    else if (axis == AXIS_X)                ax = (Vector3){1,0,0};
                    else if (axis == AXIS_Z)                ax = (Vector3){0,0,1};

                    Quaternion q = QuaternionFromAxisAngle(ax, dx);

                    for (int i = 0; i < g_vertCount; ++i) {
                        if (vertSelected[i]) {
                            Vector3 p   = rotStartPos[i];
                            Vector3 rel = Vector3Subtract(p, rotPivot);
                            Vector3 rr  = Vector3RotateByQuaternion(rel, q);
                            g_verts[i].pos = Vector3Add(rotPivot, rr);
                        } else {
                            g_verts[i].pos = rotStartPos[i];
                        }
                    }
                }
            }

            // SCALE
            if (tmode == T_SCALE) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    for (int i = 0; i < g_vertCount; ++i)
                        g_verts[i].pos = scaleStartPos[i];
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dx = (m.x - scaleStartMouse.x) * 0.01f;
                    float s = 1.0f + dx;
                    s = ClampFloat(s, 0.01f, 10.0f);

                    for (int i = 0; i < g_vertCount; ++i) {
                        if (vertSelected[i]) {
                            Vector3 base = scaleStartPos[i];
                            Vector3 rel  = Vector3Subtract(base, scalePivot);
                            Vector3 rr   = rel;

                            if (axis == AXIS_FREE)      rr = Vector3Scale(rel, s);
                            else if (axis == AXIS_X)    rr.x *= s;
                            else if (axis == AXIS_Y)    rr.y *= s;
                            else if (axis == AXIS_Z)    rr.z *= s;

                            g_verts[i].pos = Vector3Add(scalePivot, rr);
                        } else {
                            g_verts[i].pos = scaleStartPos[i];
                        }
                    }
                }
            }

            // EXTRUDE
            if (tmode == T_EXTRUDE) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    g_vertCount = extrudeOldVertCount;
                    g_triCount  = extrudeOldTriCount;
                    g_triHidden[extrudeBaseTri] = 0;
                    selTri = extrudeBaseTri;
                    tmode  = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dy = (m.y - extrudeStartMouse.y) * 0.02f;
                    float amount = -dy;

                    for (int i = 0; i < 3; ++i) {
                        int vi = extrudeNewVerts[i];
                        if (vi < 0 || vi >= g_vertCount) continue;
                        Vector3 base = extrudeBasePos[i];
                        Vector3 off  = Vector3Scale(extrudeNormal, amount);
                        g_verts[vi].pos = Vector3Add(base, off);
                    }
                }
            }

            // INSET
            if (tmode == T_INSET) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    g_vertCount = insetOldVertCount;
                    g_triCount  = insetOldTriCount;
                    selTri      = insetBaseTri;
                    tmode       = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dx = (m.x - insetStartMouse.x) * 0.01f;
                    float s = 0.3f + dx;
                    s = ClampFloat(s, 0.05f, 0.9f);

                    for (int i = 0; i < 3; ++i) {
                        int vi = insetNewVerts[i];
                        if (vi < 0 || vi >= g_vertCount) continue;
                        Vector3 base = insetBasePos[i];
                        Vector3 p = Vector3Lerp(base, insetCenter, s);
                        g_verts[vi].pos = p;
                    }
                }
            }

            // FACE BEVEL
            if (tmode == T_BEVEL) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    g_vertCount = bevelOldVertCount;
                    g_triCount  = bevelOldTriCount;
                    g_triHidden[bevelBaseTri] = 0;
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dx = (m.x - bevelStartMouse.x) * 0.01f;
                    float f = 0.25f + dx;
                    UpdateBevelGeometry(f);
                }
            }
        }

        // =====================================================
        // VIEWPORT MODE (object-level G/R/S)
// =====================================================
        if (uiMode == MODE_VIEWPORT)
        {
            if (!IsKeyDown(KEY_LEFT_ALT) &&
                IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
                g_triCount > 0)
            {
                Ray ray = GetMouseRay(GetMousePosition(), cam);
                float bestDist2 = 1e30f;
                int bestTri = -1;

                for (int t = 0; t < g_triCount; ++t) {
                    if (g_triHidden[t]) continue;
                    EditTri *tri = &g_tris[t];
                    Vector3 v0 = VpTransformPoint(g_verts[tri->v[0]].pos, vpObjPos, vpObjRotY, vpObjScale);
                    Vector3 v1 = VpTransformPoint(g_verts[tri->v[1]].pos, vpObjPos, vpObjRotY, vpObjScale);
                    Vector3 v2 = VpTransformPoint(g_verts[tri->v[2]].pos, vpObjPos, vpObjRotY, vpObjScale);

                    if (RayIntersectsTriangle(ray, v0, v1, v2)) {
                        Vector3 c = {
                            (v0.x + v1.x + v2.x)/3.0f,
                            (v0.y + v1.y + v2.y)/3.0f,
                            (v0.z + v1.z + v2.z)/3.0f
                        };
                        float d2 = Vector3LengthSqr(Vector3Subtract(c, cam.position));
                        if (d2 < bestDist2) {
                            bestDist2 = d2;
                            bestTri   = t;
                        }
                    }
                }

                if (bestTri >= 0) {
                    selTri = bestTri;
                    selMode = SEL_FACE;
                    selIndex = 0;
                }
            }

            static Vector2 vpGrabMouse     = {0};
            static Vector3 vpGrabStartPos  = {0};
            static Vector2 vpRotMouse      = {0};
            static float   vpRotStartY     = 0.0f;
            static Vector2 vpScaleMouse    = {0};
            static float   vpScaleStart    = 1.0f;

            if (tmode == T_NONE) {
                if (IsKeyPressed(KEY_G)) {
                    tmode = T_GRAB;
                    vpGrabMouse    = GetMousePosition();
                    vpGrabStartPos = vpObjPos;
                }
                if (IsKeyPressed(KEY_R)) {
                    tmode = T_ROTATE;
                    vpRotMouse  = GetMousePosition();
                    vpRotStartY = vpObjRotY;
                }
                if (IsKeyPressed(KEY_S)) {
                    tmode = T_SCALE;
                    vpScaleMouse = GetMousePosition();
                    vpScaleStart = vpObjScale;
                }
            }

            if (tmode == T_GRAB) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    vpObjPos = vpGrabStartPos;
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    Vector2 d = { m.x - vpGrabMouse.x, m.y - vpGrabMouse.y };
                    float dx = d.x * 0.01f;
                    float dy = -d.y * 0.01f;

                    Vector3 forward = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
                    Vector3 right   = Vector3Normalize(Vector3CrossProduct(forward, (Vector3){0,1,0}));
                    Vector3 up      = (Vector3){0,1,0};

                    Vector3 off = {0};
                    off.x += right.x * dx;
                    off.y += up.y    * dy;
                    off.z += right.z * dx;

                    vpObjPos.x = vpGrabStartPos.x + off.x;
                    vpObjPos.y = vpGrabStartPos.y + off.y;
                    vpObjPos.z = vpGrabStartPos.z + off.z;
                }
            }

            if (tmode == T_ROTATE) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    vpObjRotY = vpRotStartY;
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dx = (m.x - vpRotMouse.x) * 0.01f;
                    vpObjRotY = vpRotStartY + dx;
                }
            }

            if (tmode == T_SCALE) {
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER))
                    tmode = T_NONE;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_ESCAPE)) {
                    vpObjScale = vpScaleStart;
                    tmode = T_NONE;
                } else {
                    Vector2 m = GetMousePosition();
                    float dx = (m.x - vpScaleMouse.x) * 0.01f;
                    float s = vpScaleStart + dx;
                    s = ClampFloat(s, 0.05f, 20.0f);
                    vpObjScale = s;
                }
            }
        }

        // =====================================================
        // DRAW
        // =====================================================
        BeginDrawing();
        ClearBackground((Color){18,18,24,255});
        BeginMode3D(cam);

        DrawGrid(20, 1.0f);

        for (int t = 0; t < g_triCount; ++t) {
            if (g_triHidden[t]) continue;
            EditTri *tri = &g_tris[t];

            Vector3 v0, v1, v2;
            if (uiMode == MODE_VIEWPORT) {
                v0 = VpTransformPoint(g_verts[tri->v[0]].pos, vpObjPos, vpObjRotY, vpObjScale);
                v1 = VpTransformPoint(g_verts[tri->v[1]].pos, vpObjPos, vpObjRotY, vpObjScale);
                v2 = VpTransformPoint(g_verts[tri->v[2]].pos, vpObjPos, vpObjRotY, vpObjScale);
            } else {
                v0 = g_verts[tri->v[0]].pos;
                v1 = g_verts[tri->v[1]].pos;
                v2 = g_verts[tri->v[2]].pos;
            }

            if (uiMode == MODE_VIEWPORT) {
                Color fc = (t == selTri) ? (Color){250,250,250,255} : (Color){220,220,220,255};
                DrawTriangle3D(v0, v1, v2, fc);
            } else {
                Color fc = (t == selTri && selMode == SEL_FACE)
                           ? (Color){130,190,255,255}
                           : (Color){80,110,200,255};
                DrawTriangle3D(v0, v1, v2, fc);

                Color wc = (Color){240,240,240,255};
                DrawLine3D(v0, v1, wc);
                DrawLine3D(v1, v2, wc);
                DrawLine3D(v2, v0, wc);

                if (t == selTri && selMode == SEL_EDGE) {
                    Vector3 a,b;
                    if      (selIndex == 0) { a = v0; b = v1; }
                    else if (selIndex == 1) { a = v1; b = v2; }
                    else                    { a = v2; b = v0; }
                    DrawLine3D(a, b, (Color){255,80,80,255});
                }

                if (t == selTri && selMode == SEL_VERTEX) {
                    int vi = g_tris[t].v[selIndex % 3];
                    Vector3 p = g_verts[vi].pos;
                    DrawSphere(p, 0.06f, (Color){255,220,80,255});
                }
            }
        }

        if (uiMode == MODE_EDIT) {
            DrawSphere(g_origin, 0.08f, (Color){255,200,0,255});
        }

        EndMode3D();

        if (uiMode == MODE_EDIT) {
            DrawText("YSU Mesh Edit 2.0 (EDIT MODE)",            10, 10, 20, RAYWHITE);
            DrawText("F1=Edit, F2=Viewport",                     10, 34, 16, RAYWHITE);
            DrawText("ALT+LMB orbit, wheel zoom",                10, 54, 16, RAYWHITE);
            DrawText("1=V, 2=E, 3=F | TAB tri, E edge index",    10, 74, 16, RAYWHITE);
            DrawText("G/R/S (X/Y/Z) | F=Extrude, I=Inset",       10, 94, 16, RAYWHITE);
            DrawText("B=Bevel (Face=interaktif, Edge=chamfer)",  10,114, 16, RAYWHITE);
            DrawText("M=Merge, O=Origin, SHIFT+A=Add Mesh",      10,134, 16, RAYWHITE);
            DrawText("SHIFT+E=Export OBJ, SHIFT+O=Import OBJ",   10,154, 16, RAYWHITE);

            if (showAddMenu) {
                DrawText("ADD MESH: 1=Cube  2=Sphere  3=Cylinder",
                         10, 190, 18, (Color){200,220,255,255});
            }
        } else {
            DrawText("YSU Viewport (OBJECT MODE)",               10, 10, 20, RAYWHITE);
            DrawText("F1=Edit, F2=Viewport",                     10, 34, 16, RAYWHITE);
            DrawText("ALT+LMB orbit, wheel zoom",                10, 54, 16, RAYWHITE);
            DrawText("LMB face pick | G/R/S = object transform", 10, 74, 16, RAYWHITE);
        }

        if (showExportMessage) {
            DrawText("Exported to export.obj", W - 280, 10, 18, (Color){100,255,130,255});
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
