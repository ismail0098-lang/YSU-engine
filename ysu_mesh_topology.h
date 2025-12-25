#ifndef YSU_MESH_TOPOLOGY_H
#define YSU_MESH_TOPOLOGY_H

#include "raylib.h"
#include "raymath.h"

// Buralar hem topology hem edit tarafında kullanılıyor
// Sayılar şimdilik rahat olsun, sonra istersen arttırırız.
#define MAX_VERTS  8000
#define MAX_TRIS   4000
#define YSU_MAX_EDGES   (MAX_TRIS * 3)

// Edit tarafında kullandığımız vertex / tri tipleri
typedef struct {
    Vector3 pos;
} EditVertex;

typedef struct {
    int v[3];
} EditTri;

// Edge = iki vertex + en fazla iki üçgen (tri0 & tri1)
typedef struct {
    int v0, v1;
    int tri0;
    int tri1;
} MeshEdge;

typedef struct {
    MeshEdge edges[YSU_MAX_EDGES];
    int      edgeCount;
} MeshTopology;

// Triangle listesinden edge listesi çıkar
void Topology_Build(MeshTopology *topo,
                    EditTri *tris, int triCount,
                    int vertCount);

// (v0,v1) edge'inin topo içindeki index'i, yoksa -1
int Topology_FindEdge(const MeshTopology *topo,
                      int v0, int v1);

// Blender-vari bevel (tek segmentlik basit chamfer):
// - yeni vertexler ekler (pVertCount artar)
// - yeni üçgenler ekler (pTriCount artar)
// - edge çevresinde 4 üçgenlik bevel bandı oluşturur
int Mesh_BevelEdge(const MeshTopology *topo,
                   int edgeIndex,
                   int segments,
                   float amount,
                   EditVertex *verts,
                   int *pVertCount,
                   EditTri *tris,
                   int *pTriCount);

#endif // YSU_MESH_TOPOLOGY_H