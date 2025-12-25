#include "ysu_obj_exporter.h"

int ExportOBJ(const char *path,
              EditVertex *verts, int vertCount,
              EditTri *tris,   int triCount)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    fprintf(f, "# Exported from YSU Mesh Edit\n");

    // vertexler
    for (int i = 0; i < vertCount; i++) {
        Vector3 p = verts[i].pos;
        fprintf(f, "v %f %f %f\n", p.x, p.y, p.z);
    }

    // üçgen yüzler
    for (int i = 0; i < triCount; i++) {
        int a = tris[i].v[0] + 1; // OBJ 1-based index
        int b = tris[i].v[1] + 1;
        int c = tris[i].v[2] + 1;
        fprintf(f, "f %d %d %d\n", a, b, c);
    }

    fclose(f);
    return 1;
}
