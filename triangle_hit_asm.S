.intel_syntax noprefix
.text
.globl ysu_hit_triangle_asm

# int ysu_hit_triangle_asm(
#   RCX = Triangle* tri
#   RDX = Ray* r
#   XMM0 = t_min
#   XMM1 = t_max
#   R8  = float* out_t
#   R9  = float* out_u
#   [rsp+40] = float* out_v
# )

ysu_hit_triangle_asm:
    push rbp
    mov rbp, rsp
    sub rsp, 32

    mov r10, [rbp+40]          # out_v

    # load p0,p1,p2 (Vec3 are assumed 16B: x,y,z,pad)
    movups xmm2, [rcx]         # p0
    movups xmm3, [rcx+16]      # p1
    movups xmm4, [rcx+32]      # p2

    # e1 = p1 - p0
    movaps xmm5, xmm3
    subps xmm5, xmm2

    # e2 = p2 - p0
    movaps xmm6, xmm4
    subps xmm6, xmm2

    # dir
    movups xmm7, [rdx+16]

    # pvec = cross(dir, e2)
    movaps xmm8, xmm7
    shufps xmm8, xmm8, 0b11001001
    movaps xmm9, xmm6
    shufps xmm9, xmm9, 0b11010010
    mulps xmm8, xmm9

    movaps xmm10, xmm7
    shufps xmm10, xmm10, 0b11010010
    movaps xmm11, xmm6
    shufps xmm11, xmm11, 0b11001001
    mulps xmm10, xmm11

    subps xmm8, xmm10          # pvec

    # det = dot(e1, pvec)
    movaps xmm12, xmm5
    mulps xmm12, xmm8
    haddps xmm12, xmm12
    haddps xmm12, xmm12        # det in xmm12 low

    # abs(det)
    andps xmm12, XMMWORD PTR [rip + abs_mask]
    comiss xmm12, DWORD PTR [rip + eps]
    jb miss

    # invDet = 1/det
    movss xmm13, DWORD PTR [rip + one]
    divss xmm13, xmm12

    # tvec = origin - p0
    movups xmm14, [rdx]        # origin
    subps xmm14, xmm2

    # u = dot(tvec,pvec) * invDet
    movaps xmm15, xmm14
    mulps xmm15, xmm8
    haddps xmm15, xmm15
    haddps xmm15, xmm15
    mulss xmm15, xmm13         # u in xmm15 low

    comiss xmm15, DWORD PTR [rip + zero]
    jb miss
    comiss xmm15, DWORD PTR [rip + one]
    ja miss

    # qvec = cross(tvec, e1)
    movaps xmm8, xmm14
    shufps xmm8, xmm8, 0b11001001
    movaps xmm9, xmm5
    shufps xmm9, xmm9, 0b11010010
    mulps xmm8, xmm9

    movaps xmm10, xmm14
    shufps xmm10, xmm10, 0b11010010
    movaps xmm11, xmm5
    shufps xmm11, xmm11, 0b11001001
    mulps xmm10, xmm11

    subps xmm8, xmm10          # qvec

    # v = dot(dir,qvec) * invDet
    movaps xmm9, xmm7
    mulps xmm9, xmm8
    haddps xmm9, xmm9
    haddps xmm9, xmm9
    mulss xmm9, xmm13          # v in xmm9 low

    comiss xmm9, DWORD PTR [rip + zero]
    jb miss

    movaps xmm10, xmm15
    addss xmm10, xmm9
    comiss xmm10, DWORD PTR [rip + one]
    ja miss

    # t = dot(e2,qvec) * invDet
    movaps xmm11, xmm6
    mulps xmm11, xmm8
    haddps xmm11, xmm11
    haddps xmm11, xmm11
    mulss xmm11, xmm13         # t in xmm11 low

    comiss xmm11, xmm0
    jb miss
    comiss xmm11, xmm1
    ja miss

    # write outputs (NO mem->mem movss)
    movss DWORD PTR [r8], xmm11
    movss DWORD PTR [r9], xmm15
    movss DWORD PTR [r10], xmm9

    mov eax, 1
    jmp done

miss:
    xor eax, eax

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
