# Experimental "Gerçek Sıçrama" Modülleri

Bu klasör **çekirdeğe dokunmadan** eklenmiş, ciddi şekilde yazılmış deneysel modülleri içerir.

## 1) AVX2 Ray Packet / Triangle Packet
Dosyalar:
- `ysu_packet.h`
- `ysu_packet.c`

Sunulanlar:
- `YSU_Ray8`: 8-wide ray packet (SoA)
- `YSU_Tri8`: 8-wide triangle packet (SoA, v0 + edges)
- `ysu_intersect_ray8_tri1`: 8 ray vs 1 triangle (Möller–Trumbore, AVX2)
- `ysu_intersect_ray1_tri8`: 1 ray vs 8 triangles (AVX2), en yakın hit seçer

> Not: Bu modül şu an BVH leaf'e bağlanmıyor. Ama leaf'te 8 tri bloklarına geçince doğrudan kullanılır.

## 2) Wavefront Path Tracing Skeleton
Dosyalar:
- `ysu_wavefront.h`
- `ysu_wavefront.c`

Bu bir "pipeline iskeleti":
- active queue (rays/paths)
- intersect callback
- shade/scatter callback
- next queue

Bu sayede:
- batching açılır
- cache locality artar
- packet intersection kullanımı kolaylaşır

## Derleme
AVX2 açık olsun:
- GCC/Clang: `-mavx2 -O3`
- MSVC: `/arch:AVX2`

Örnek (sadece modülü testlemek için):
`gcc -O3 -mavx2 -std=c11 experimental/ysu_packet.c vec3.c ray.c -o packet_test.exe`

## Entegrasyon Planı (Doğru Sıra)
1) BVH leaf'te triangle listelerini 8'li bloklara düzenle
2) Leaf intersection: `ysu_intersect_ray1_tri8` (ray scalar, tri packet)
3) Sonra wavefront: primary rays -> intersect batch -> shade batch

Bunları istersen ben senin mevcut BVH ve primitive layout'una göre **tam entegre** edecek şekilde bağlarım.
