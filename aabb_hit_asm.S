.intel_syntax noprefix
.text
.globl aabb_hit_asm

.equ R_OX, 0
.equ R_OY, 4
.equ R_OZ, 8
.equ R_DX, 12
.equ R_DY, 16
.equ R_DZ, 20

.equ B_MINX, 0
.equ B_MINY, 4
.equ B_MINZ, 8
.equ B_MAXX, 12
.equ B_MAXY, 16
.equ B_MAXZ, 20

aabb_hit_asm:
    movss xmm0, DWORD PTR [rdx + R_DX]
    cvtss2sd xmm0, xmm0
    movapd xmm1, xmm0
    andpd  xmm1, XMMWORD PTR [rip + ABSMASK]
    comisd xmm1, QWORD PTR [rip + EPS]
    jb .x_parallel

    movsd xmm7, QWORD PTR [rip + ONE]
    divsd xmm7, xmm0

    movss xmm4, DWORD PTR [rdx + R_OX]
    cvtss2sd xmm4, xmm4

    movss xmm5, DWORD PTR [rcx + B_MINX]
    cvtss2sd xmm5, xmm5
    subsd xmm5, xmm4
    mulsd xmm5, xmm7

    movss xmm6, DWORD PTR [rcx + B_MAXX]
    cvtss2sd xmm6, xmm6
    subsd xmm6, xmm4
    mulsd xmm6, xmm7

    comisd xmm5, xmm6
    jbe .x_no_swap
    movapd xmm0, xmm5
    movapd xmm5, xmm6
    movapd xmm6, xmm0
.x_no_swap:
    maxsd xmm2, xmm5
    minsd xmm3, xmm6
    comisd xmm3, xmm2
    jbe .miss
    jmp .y_axis

.x_parallel:
    movss xmm4, DWORD PTR [rdx + R_OX]
    cvtss2sd xmm4, xmm4
    movss xmm5, DWORD PTR [rcx + B_MINX]
    cvtss2sd xmm5, xmm5
    movss xmm6, DWORD PTR [rcx + B_MAXX]
    cvtss2sd xmm6, xmm6
    comisd xmm4, xmm5
    jb .miss
    comisd xmm4, xmm6
    ja .miss

.y_axis:
    movss xmm0, DWORD PTR [rdx + R_DY]
    cvtss2sd xmm0, xmm0
    movapd xmm1, xmm0
    andpd  xmm1, XMMWORD PTR [rip + ABSMASK]
    comisd xmm1, QWORD PTR [rip + EPS]
    jb .y_parallel

    movsd xmm7, QWORD PTR [rip + ONE]
    divsd xmm7, xmm0

    movss xmm4, DWORD PTR [rdx + R_OY]
    cvtss2sd xmm4, xmm4

    movss xmm5, DWORD PTR [rcx + B_MINY]
    cvtss2sd xmm5, xmm5
    subsd xmm5, xmm4
    mulsd xmm5, xmm7

    movss xmm6, DWORD PTR [rcx + B_MAXY]
    cvtss2sd xmm6, xmm6
    subsd xmm6, xmm4
    mulsd xmm6, xmm7

    comisd xmm5, xmm6
    jbe .y_no_swap
    movapd xmm0, xmm5
    movapd xmm5, xmm6
    movapd xmm6, xmm0
.y_no_swap:
    maxsd xmm2, xmm5
    minsd xmm3, xmm6
    comisd xmm3, xmm2
    jbe .miss
    jmp .z_axis

.y_parallel:
    movss xmm4, DWORD PTR [rdx + R_OY]
    cvtss2sd xmm4, xmm4
    movss xmm5, DWORD PTR [rcx + B_MINY]
    cvtss2sd xmm5, xmm5
    movss xmm6, DWORD PTR [rcx + B_MAXY]
    cvtss2sd xmm6, xmm6
    comisd xmm4, xmm5
    jb .miss
    comisd xmm4, xmm6
    ja .miss

.z_axis:
    movss xmm0, DWORD PTR [rdx + R_DZ]
    cvtss2sd xmm0, xmm0
    movapd xmm1, xmm0
    andpd  xmm1, XMMWORD PTR [rip + ABSMASK]
    comisd xmm1, QWORD PTR [rip + EPS]
    jb .z_parallel

    movsd xmm7, QWORD PTR [rip + ONE]
    divsd xmm7, xmm0

    movss xmm4, DWORD PTR [rdx + R_OZ]
    cvtss2sd xmm4, xmm4

    movss xmm5, DWORD PTR [rcx + B_MINZ]
    cvtss2sd xmm5, xmm5
    subsd xmm5, xmm4
    mulsd xmm5, xmm7

    movss xmm6, DWORD PTR [rcx + B_MAXZ]
    cvtss2sd xmm6, xmm6
    subsd xmm6, xmm4
    mulsd xmm6, xmm7

    comisd xmm5, xmm6
    jbe .z_no_swap
    movapd xmm0, xmm5
    movapd xmm5, xmm6
    movapd xmm6, xmm0
.z_no_swap:
    maxsd xmm2, xmm5
    minsd xmm3, xmm6
    comisd xmm3, xmm2
    jbe .miss

    mov eax, 1
    ret

.z_parallel:
    movss xmm4, DWORD PTR [rdx + R_OZ]
    cvtss2sd xmm4, xmm4
    movss xmm5, DWORD PTR [rcx + B_MINZ]
    cvtss2sd xmm5, xmm5
    movss xmm6, DWORD PTR [rcx + B_MAXZ]
    cvtss2sd xmm6, xmm6
    comisd xmm4, xmm5
    jb .miss
    comisd xmm4, xmm6
    ja .miss

    mov eax, 1
    ret

.miss:
    xor eax, eax
    ret

.section .rodata
.align 16
ONE:    .double 1.0
EPS:    .double 1.0e-12
ABSMASK:
    .quad 0x7fffffffffffffff
    .quad 0x7fffffffffffffff
