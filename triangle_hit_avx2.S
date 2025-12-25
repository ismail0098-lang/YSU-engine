.intel_syntax noprefix
.text
.globl ysu_hit_triangle_avx2

# int ysu_hit_triangle_avx2(
#   RCX = Triangle* tri
#   RDX = Ray* r
#   XMM0 = t_min
#   XMM1 = t_max
#   R8  = float* out_t
#   R9  = float* out_u
#   [rsp+40] = float* out_v
# )

ysu_hit_triangle_avx2:
    push rbp
    mov rbp, rsp
    sub rsp, 32

    mov r10, [rbp+40]          # out_v

    vmovups xmm2, [rcx]        # p0
    vmovups xmm3, [rcx+16]     # p1
    vmovups xmm4, [rcx+32]     # p2

    vsubps xmm5, xmm3, xmm2    # e1
    vsubps xmm6, xmm4, xmm2    # e2

    vmovups xmm7, [rdx+16]     # dir

    # pvec = cross(dir, e2)
    vmovaps xmm8, xmm7
    vshufps xmm8, xmm8, xmm8, 0b11001001
    vmovaps xmm9, xmm6
    vshufps xmm9, xmm9, xmm9, 0b11010010
    vmulps xmm8, xmm8, xmm9

    vmovaps xmm10, xmm7
    vshufps xmm10, xmm10, xmm10, 0b11010010
    vmovaps xmm11, xmm6
    vshufps xmm11, xmm11, xmm11, 0b11001001
    vmulps xmm10, xmm10, xmm11

    vsubps xmm8, xmm8, xmm10   # pvec

    # det = dot(e1, pvec)
    vmulps xmm12, xmm5, xmm8
    vhaddps xmm12, xmm12, xmm12
    vhaddps xmm12, xmm12, xmm12

    vandps xmm12, xmm12, XMMWORD PTR [rip + abs_mask]
    vcomiss xmm12, DWORD PTR [rip + eps]
    jb miss

    vmovss xmm13, DWORD PTR [rip + one]
    vdivss xmm13, xmm13, xmm12

    # tvec = origin - p0
    vmovups xmm14, [rdx]
    vsubps xmm14, xmm14, xmm2

    # u
    vmulps xmm15, xmm14, xmm8
    vhaddps xmm15, xmm15, xmm15
    vhaddps xmm15, xmm15, xmm15
    vmulss xmm15, xmm15, xmm13

    vcomiss xmm15, DWORD PTR [rip + zero]
    jb miss
    vcomiss xmm15, DWORD PTR [rip + one]
    ja miss

    # qvec = cross(tvec, e1)
    vmovaps xmm8, xmm14
    vshufps xmm8, xmm8, xmm8, 0b11001001
    vmovaps xmm9, xmm5
    vshufps xmm9, xmm9, xmm9, 0b11010010
    vmulps xmm8, xmm8, xmm9

    vmovaps xmm10, xmm14
    vshufps xmm10, xmm10, xmm10, 0b11010010
    vmovaps xmm11, xmm5
    vshufps xmm11, xmm11, xmm11, 0b11001001
    vmulps xmm10, xmm10, xmm11

    vsubps xmm8, xmm8, xmm10   # qvec

    # v
    vmulps xmm9, xmm7, xmm8
    vhaddps xmm9, xmm9, xmm9
    vhaddps xmm9, xmm9, xmm9
    vmulss xmm9, xmm9, xmm13

    vcomiss xmm9, DWORD PTR [rip + zero]
    jb miss

    vaddss xmm10, xmm15, xmm9
    vcomiss xmm10, DWORD PTR [rip + one]
    ja miss

    # t
    vmulps xmm11, xmm6, xmm8
    vhaddps xmm11, xmm11, xmm11
    vhaddps xmm11, xmm11, xmm11
    vmulss xmm11, xmm11, xmm13

    vcomiss xmm11, xmm0
    jb miss
    vcomiss xmm11, xmm1
    ja miss

    vmovss DWORD PTR [r8], xmm11
    vmovss DWORD PTR [r9], xmm15
    vmovss DWORD PTR [r10], xmm9

    mov eax, 1
    vzeroupper
    jmp done

miss:
    xor eax, eax
    vzeroupper

done:
    add rsp, 32
    pop rbp
    ret

.section .rodata
.align 16
eps:      .float 1e-8
one:      .float 1.0
zero:     .float 0.0
abs_mask: .int 0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff
