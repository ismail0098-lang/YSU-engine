#include "ysu_mesh_topology.h"

// Küçük helper: (a,b) sırasız pair -> sıralı hale getir
static void SortPair(int *a, int *b)
{
    if (*a > *b) {
        int t = *a;
        *a = *b;
        *b = t;
    }
}

void Topology_Build(MeshTopology *topo,
                    EditTri *tris, int triCount,
                    int vertCount)
{
    (void)vertCount; // Şimdilik kullanmıyoruz

    topo->edgeCount = 0;

    for (int ti = 0; ti < triCount; ++ti) {
        EditTri *t = &tris[ti];
        int v[3] = { t->v[0], t->v[1], t->v[2] };

        for (int e = 0; e < 3; ++e) {
            int a = v[e];
            int b = v[(e + 1) % 3];
            if (a < 0 || b < 0) continue;

            SortPair(&a, &b);

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

int Topology_FindEdge(const MeshTopology *topo,
                      int v0, int v1)
{
    SortPair(&v0, &v1);
    for (int i = 0; i < topo->edgeCount; ++i) {
        const MeshEdge *e = &topo->edges[i];
        if (e->v0 == v0 && e->v1 == v1) return i;
    }
    return -1;
}

// Asıl olay burada: Blender-vari edge bevel (tek segment)
// (v0,v1) edge'ini alıp, iki ucuna yeni vertex ekleyip
// tri0 ve tri1'i 4 üçgene bölüyoruz.
int Mesh_BevelEdge(const MeshTopology *topo,
                   int edgeIndex,
                   int segments,
                   float amount,
                   EditVertex *verts,
                   int *pVertCount,
                   EditTri *tris,
                   int *pTriCount)
{
    if (!topo || !verts || !pVertCount || !tris || !pTriCount) return 0;
    if (edgeIndex < 0 || edgeIndex >= topo->edgeCount) return 0;

    MeshEdge e = topo->edges[edgeIndex];

    int vertCount = *pVertCount;
    int triCount  = *pTriCount;

    // Şimdilik sadece 2 yüzü olan interior edge'ler için bevel yapıyoruz
    if (e.tri0 < 0 || e.tri1 < 0) {
        // border edge için bevel'i sonra ekleriz
        return 0;
    }

    if (vertCount + 2 > MAX_VERTS) return 0; // iki yeni vertex
    if (triCount  + 2 > MAX_TRIS)  return 0; // iki yeni üçgen

    int v0 = e.v0;
    int v1 = e.v1;

    // tri0 için karşı vertex (c)
    EditTri *t0 = &tris[e.tri0];
    int c = -1;
    for (int i = 0; i < 3; ++i) {
        int vi = t0->v[i];
        if (vi != v0 && vi != v1) {
            c = vi;
            break;
        }
    }

    // tri1 için karşı vertex (d)
    EditTri *t1 = &tris[e.tri1];
    int d = -1;
    for (int i = 0; i < 3; ++i) {
        int vi = t1->v[i];
        if (vi != v0 && vi != v1) {
            d = vi;
            break;
        }
    }

    if (c < 0 || d < 0) {
        // topoloji bozuksa abort
        return 0;
    }

    Vector3 p0 = verts[v0].pos;
    Vector3 p1 = verts[v1].pos;
    Vector3 pc = verts[c].pos;
    Vector3 pd = verts[d].pos;

    // İki yüzün normalini hesapla
    Vector3 n0 = Vector3Normalize(Vector3CrossProduct(
                      Vector3Subtract(p1, p0),
                      Vector3Subtract(pc, p0)));

    Vector3 n1 = Vector3Normalize(Vector3CrossProduct(
                      Vector3Subtract(p0, p1),
                      Vector3Subtract(pd, p1)));

    Vector3 nAvg = Vector3Add(n0, n1);
    if (Vector3Length(nAvg) < 1e-6f) {
        nAvg = n0; // yedek
    }
    nAvg = Vector3Normalize(nAvg);

    if (segments < 1) segments = 1; // şimdilik sadece 1 segment
    float t = amount;

    // Edge uçları için bevel pozisyonları
    Vector3 p0b = Vector3Add(p0, Vector3Scale(nAvg, t));
    Vector3 p1b = Vector3Add(p1, Vector3Scale(nAvg, t));

    // İki yeni vertex ekle
    int bv0 = vertCount++;
    int bv1 = vertCount++;

    verts[bv0].pos = p0b;
    verts[bv1].pos = p1b;

    // Eski iki üçgeni yeniden yaz + iki yeni üçgen ekle
    // Toplamda 4 üçgen ile edge çevresinde bir chamfer bandı
    // oluşuyor:

    // tri0: (v0, c, bv0)  ve yeni tri: (bv0, c, bv1)
    // tri1: (v1, d, bv1)  ve yeni tri: (bv1, d, bv0)

    // tri0'yı güncelle
    t0->v[0] = v0;
    t0->v[1] = c;
    t0->v[2] = bv0;

    // tri1'i güncelle
    t1->v[0] = v1;
    t1->v[1] = d;
    t1->v[2] = bv1;

    // yeni tri2
    EditTri *t2 = &tris[triCount++];
    t2->v[0] = bv0;
    t2->v[1] = c;
    t2->v[2] = bv1;

    // yeni tri3
    EditTri *t3 = &tris[triCount++];
    t3->v[0] = bv1;
    t3->v[1] = d;
    t3->v[2] = bv0;

    *pVertCount = vertCount;
    *pTriCount  = triCount;

    return 1;
}