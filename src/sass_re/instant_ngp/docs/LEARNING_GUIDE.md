# Instant-NGP SASS Kernels — Learning Guide

*For anyone who wants to understand GPU optimization from the ground up. No GPU knowledge needed — just curiosity.*

---

## Table of Contents

1. [Start Here: What Even IS Assembly?](#start-here-what-even-is-assembly)
2. [Why GPUs Are Different (And Why That Matters)](#why-gpus-are-different)
3. [The Memory Problem: Why Fast Code Is Hard](#the-memory-problem)
4. [Reading Real SASS: A Guided Tour](#reading-real-sass-a-guided-tour)
5. [The NeRF Pipeline](#the-nerf-pipeline)
6. [Kernel 1: Hash Grid Encoding — Deep Dive](#kernel-1-hash-grid-encoding)
7. [Kernel 2: MLP Forward Pass — Deep Dive](#kernel-2-mlp-forward-pass)
8. [Kernel 3: Volume Rendering — Deep Dive](#kernel-3-volume-rendering)
9. [How to Think About Performance](#how-to-think-about-performance)
10. [Key Concepts Glossary](#key-concepts-glossary)

---

## Start Here: What Even IS Assembly?

### Every computer runs on numbers

At the very bottom layer, a computer processor (CPU or GPU) doesn't understand Python, C++, or English. It understands **numbers**. Specifically, it reads a stream of binary instructions — patterns of 1s and 0s — and each pattern means "do something specific."

For example, the number `0x823F0004000780FF` might mean:

> "Multiply the value in register 0 by the value in register 1, add the value in register 2, and store the result back in register 0."

That's one **machine instruction**. Your GPU might execute 10 billion of these per second.

### What's a register?

A **register** is a tiny storage slot *inside* the processor itself. Think of it like a sticky note on your desk — you can read it and write on it instantly (0 cycles), but you only have a few of them.

On our GPU (RTX 4070 Ti Super), each thread gets up to **255 registers**, each holding one 32-bit number (an integer or a floating-point number like `3.14`).

Contrast this with **memory** (VRAM), which is a huge storage space (16 GB) but takes 200-400 clock cycles to access — imagine walking to a warehouse across town to get a piece of paper, vs reading the sticky note right in front of you.

### So what is "assembly language"?

Assembly is a human-readable version of those machine instructions. Instead of remembering that `0x823F...` means "multiply-add," we write:

```
FFMA R0, R1, R2, R0     ← "R0 = R1 * R2 + R0"
```

Same instruction. Same binary. Just written with names instead of numbers so humans can read it.

**SASS** (Shader ASSembly) is NVIDIA's name for their GPU assembly language. Every GPU has its own — Intel calls theirs "EU ISA," AMD calls theirs "RDNA ISA." They're all the same idea: the real instructions the hardware executes.

### The compilation chain

When you write a program in a high-level language, it goes through layers of translation:

```
┌───────────────────────────────────────────────────────────────────┐
│ Level 1: CUDA C++ (what humans write)                             │
│                                                                   │
│   float result = a * b + c;                                       │
│                                                                   │
│         │  nvcc compiler (translates to PTX)                      │
│         ▼                                                         │
│ Level 2: PTX (virtual assembly — portable across GPU generations) │
│                                                                   │
│   fma.rn.f32   %f3, %f1, %f2, %f3;                               │
│                                                                   │
│         │  ptxas assembler (translates to SASS)                   │
│         ▼                                                         │
│ Level 3: SASS (real assembly — what the silicon actually runs)    │
│                                                                   │
│   FFMA R3, R1, R2, R3;                                            │
│                                                                   │
│         │  hardware decoder                                       │
│         ▼                                                         │
│ Level 4: Binary (electrical signals in silicon)                   │
│                                                                   │
│   0x823F0004000780FF                                              │
└───────────────────────────────────────────────────────────────────┘
```

Most programmers work at Level 1 and never look lower. We work at Levels 2 and 3 — writing PTX by hand and checking the SASS output to make sure the hardware is doing exactly what we want.

### Why would anyone write assembly by hand?

**Because the compiler isn't always right.**

The compiler is a program that translates high-level code to machine instructions. It's *very* good — decades of research have gone into it. But it has to be **safe**. It won't make assumptions. It won't take risks. It's like a translator who always uses formal, correct, but sometimes clunky phrasing.

A human who understands the hardware can sometimes write instructions that are:
- **More efficient**: using specialized instructions the compiler doesn't know about
- **Better scheduled**: ordering instructions to keep the hardware busy instead of idle
- **Leaner**: cutting out defensive code the compiler adds "just in case"

That's what this project does. Three specific GPU programs (kernels), rewritten at the assembly level, running faster than the compiler's version.

> **Think about it**: Every game, every AI model, every video you watch on your GPU runs SASS instructions. The kernel you're about to learn about executes ~770 SASS instructions per thread, 262,144 threads in parallel. That's 200 million instructions per frame. Understanding what those instructions actually *do* is understanding what the GPU actually *does*.

---

## Why GPUs Are Different

### CPU vs GPU — The Factory Analogy

A **CPU** is like a factory with 8-16 extremely skilled workers. Each worker can handle any task — complex logic, branching decisions, sequential work — but there are only a few of them.

A **GPU** is like a factory with **thousands** of simple workers. Each one can only do basic math, but there are *so many* of them that they can collectively process massive amounts of data in parallel.

The RTX 4070 Ti Super has **66 SMs** (Streaming Multiprocessors — think of them as departments), each with **128 math units**. That's 8,448 math units running simultaneously.

### Threads, Warps, and Blocks

When you write GPU code (a "kernel"), you launch it with millions of **threads**. Each thread runs the same code but on different data (think: thread 0 processes pixel 0, thread 1 processes pixel 1, etc.).

Threads are grouped:
- **Warp** (32 threads): The hardware *always* executes 32 threads in lockstep. They all run the same instruction at the same time but on different data. This is called **SIMT** (Single Instruction, Multiple Threads).
- **Block** (64-1024 threads): A group of warps that share fast "shared memory" and can synchronize with each other.
- **Grid**: All blocks launched for a kernel.

> **Think about it**: When you hear "my GPU has 8,448 cores," that doesn't mean 8,448 independent processors like a CPU. It means 264 warps can execute simultaneously, each running 32 threads in lockstep. One instruction controls 32 threads at once. That's fundamentally different from a CPU, and it's why GPU programming requires a different way of thinking.

### What makes a GPU fast (and what makes it slow)

A GPU is **unbelievably fast** at one thing: doing the same math on millions of data points simultaneously. If you need to multiply 262,144 numbers by 2, a GPU does it in microseconds.

A GPU is **terrible** at:
- **Branching** (if/else): when some threads in a warp take the `if` path and others take the `else`, *both* paths execute — the threads on the wrong path just sit idle. This is called **divergence**.
- **Sequential work**: each individual thread is much slower than a CPU core. The power comes from parallelism.
- **Random memory access**: accessing scattered, unpredictable memory locations is the single biggest performance killer.

Every optimization in this project tackles one of these weaknesses.

---

## The Memory Problem

This is **the single most important concept** for understanding GPU optimization. If you understand this section, you understand 80% of why our code is fast.

### The speed gap

```
                Access Time     Size          Analogy
                ───────────     ────          ──────
Registers       ~0 cycles       256 KB/SM     Sticky note on your desk
Shared Memory   ~23 cycles      100 KB/SM     Filing cabinet in your office
L1 Cache        ~33 cycles      128 KB/SM     Bookshelf down the hall
L2 Cache        ~200 cycles     48 MB         Library across campus
Global Memory   ~400 cycles     16 GB         Warehouse in another city
```

Look at those numbers. A register access is **instant**. A global memory access takes **400 cycles**. During those 400 cycles, the math unit could have performed **400 multiply-add operations**, but instead it's just... waiting.

This is like asking a brilliant mathematician to solve problems, but every time they need a number, they have to drive to a warehouse 2 hours away and drive back. They spend 99% of their time driving, 1% doing math.

### Latency hiding: the GPU's trick

The GPU's solution is genius: **don't wait. Work on something else.**

A GPU doesn't have 1 mathematician. It has 48 mathematicians (warps) squeezed into each department (SM). When mathematician #1 is waiting for data, the GPU switches to mathematician #2, who already has their data. When #2 needs data, switch to #3. By the time it cycles back to #1, their data has arrived.

This is called **latency hiding** — keeping the hardware busy by interleaving work from many threads.

```
Warp 0:  load A ────── waiting 200 cycles ──────── use A
Warp 1:         load B ────── waiting 200 cycles ──────── use B
Warp 2:                load C ────── waiting 200 cycles ──────── use C
...                                                              
         ↑                                                       
         The GPU switches between warps every cycle               
         At least one warp always has data ready!                 
```

**But here's the catch**: this only works if the compiler is allowed to *reorder* instructions within each warp — issuing loads early, doing math while waiting, and consuming data later. If you prevent reordering (like we did with `asm volatile` in v1), the GPU *cannot hide latency*, and performance collapses.

> **Think about it**: Our hash grid kernel loads 768 bytes from L2 cache per thread. At 200 cycles per load, that's ~19,200 cycles of waiting if done serially. But the total math (XOR, multiply, interpolation) is only ~850 cycles. The kernel is 95% waiting, 5% computing. Any optimization that reduces that waiting time — even a little — has a huge impact.

### Memory-bound vs compute-bound

Every GPU kernel has a **bottleneck** — the thing that limits its speed:

- **Memory-bound**: The kernel spends most of its time waiting for data. More math wouldn't help. Faster memory would. *Example: our hash grid kernel.*
- **Compute-bound**: The kernel spends most of its time doing math. The data arrives fast enough, but there's just too much computation. Faster math units would help. *Example: our MLP neural network kernel.*

Knowing which bottleneck you're dealing with determines **which optimizations are useful**:

| Bottleneck | What helps | What doesn't help |
|-----------|-----------|-------------------|
| Memory-bound | Fewer loads, wider loads, better caching, load reordering | More math tricks (math is free time already) |
| Compute-bound | Fewer instructions, ILP (instruction-level parallelism), hardware special functions | Memory tricks (data arrives fast enough) |

Our three kernels have different bottlenecks — that's why each one needs different optimizations.

---

## Reading Real SASS: A Guided Tour

This is where many guides lose people — they show you SASS and expect you to "just get it." Let's go slow. We'll take real instructions from our kernels and decode them piece by piece.

### Anatomy of a SASS instruction

Here's a real instruction from our MLP kernel:

```
/*0090*/   FFMA R8, R12, R5, R8;
```

Let's break down every piece:

| Part | Meaning |
|------|---------|
| `/*0090*/` | **Address** — where this instruction lives in memory (byte offset `0x0090` = instruction #18). Like a line number. |
| `FFMA` | **Opcode** — what to do. FFMA = **F**used **F**loat **M**ultiply-**A**dd |
| `R8` | **Destination** — where to store the result (register 8) |
| `R12` | **Source 1** — first input (register 12) |
| `R5` | **Source 2** — second input (register 5) |
| `R8` | **Source 3** — third input (register 8, same as destination!) |

So this instruction means: **R8 = R12 × R5 + R8**

That's one neuron's accumulation step: multiply a weight (`R12`) by an input (`R5`), add it to the running total (`R8`).

Notice `R8` appears twice — as both destination and source 3. That's an **accumulator pattern**: the register accumulates (adds up) results across many iterations.

### A longer example — loading from memory

```
/*0048*/   LDG.E.64.CONSTANT R22, [R4.64+0x100];
```

| Part | Meaning |
|------|---------|
| `LDG` | **L**oa**D** from **G**lobal memory |
| `.E` | **E**xtended address (64-bit pointer) |
| `.64` | Load 64 bits (8 bytes = two floats) |
| `.CONSTANT` | Use the CONSTANT cache path (optimized for read-only data) |
| `R22` | Destination — result goes into `R22` (and `R23`, since it's 64 bits) |
| `[R4.64+0x100]` | Address: take the 64-bit pointer in `R4:R5`, add offset `0x100` (256 bytes) |

This loads two floats from the hash table. The `.CONSTANT` hint tells the cache "this data is read-only" so it can cache it more aggressively.

**Why does 64-bit load fill TWO registers?** Because each register is 32 bits. A 64-bit load needs two registers side by side: `R22` gets the first float, `R23` gets the second. The hardware always uses consecutive register pairs for wide loads.

### The logic instruction — LOP3

```
/*0070*/   LOP3.LUT R16, R10, R14, R18, 0x96, !PT;
```

| Part | Meaning |
|------|---------|
| `LOP3` | **L**ogic **OP**eration with **3** inputs |
| `.LUT` | Uses a **L**ook**U**p **T**able to define the operation |
| `R16` | Destination |
| `R10, R14, R18` | Three input registers |
| `0x96` | The truth table — this specific value means XOR(a, XOR(b, c)) |
| `!PT` | Predicate: always execute (no condition) |

**The `0x96` magic number**: LOP3 can compute *any* 3-input boolean function using an 8-bit truth table. `0x96` in binary is `10010110`. If you write out the XOR truth table for 3 inputs, you get exactly that pattern. It's a clever hardware trick — instead of having separate AND, OR, XOR instructions, one instruction handles all of them depending on the look-up table value.

```
Inputs (a,b,c) → XOR result → Truth table bit
   0,0,0       →     0       → bit 0 = 0
   0,0,1       →     1       → bit 1 = 1
   0,1,0       →     1       → bit 2 = 1
   0,1,1       →     0       → bit 3 = 0
   1,0,0       →     1       → bit 4 = 1
   1,0,1       →     0       → bit 5 = 0
   1,1,0       →     0       → bit 6 = 0
   1,1,1       →     1       → bit 7 = 1
                               ─────────
                Reading bottom-to-top: 10010110 = 0x96
```

So `LOP3.LUT R16, R10, R14, R18, 0x96` means: **R16 = R10 XOR R14 XOR R18** — but any boolean function could be selected just by changing `0x96` to a different number.

### Special functions — MUFU

```
/*00B0*/   MUFU.EX2 R20, R19;
```

| Part | Meaning |
|------|---------|
| `MUFU` | **M**ulti-**F**unction **U**nit — hardware special math |
| `.EX2` | Compute 2^x (two to the power of x) |
| `R20` | Destination |
| `R19` | Input |

So: **R20 = 2^(R19)**

The MUFU unit is a dedicated piece of silicon on the GPU that computes transcendental functions (2^x, log2, sin, cos, reciprocal) in hardware. It's approximate (~22 bits of precision instead of 24), but it's **incredibly fast** — one instruction instead of 20-30 instructions for a software implementation.

We use this to compute `exp(-x)` by rewriting it as `2^(-x × log2(e))`. One multiply + one MUFU.EX2 = 2 instructions. The standard `expf()` function needs ~20+ instructions because it does a polynomial approximation in software.

### Predicates — conditional execution without branching

```
/*00D0*/   FSETP.LT.AND P0, PT, R6, 0.001, PT;
/*00D8*/   @P0 BRA 0x120;
```

| Part | Meaning |
|------|---------|
| `FSETP` | **F**loat **SET** **P**redicate — compare two floats and set a true/false flag |
| `.LT` | **L**ess **T**han |
| `.AND` | Combine with another predicate using AND |
| `P0` | Destination predicate register |
| `PT` | "True" predicate (second output, usually unused) |
| `R6` | The float to test (transmittance) |
| `0.001` | The threshold |
| `PT` | Pre-condition: always (PT = always true) |
| `@P0 BRA 0x120` | If P0 is true, jump (branch) to instruction at address 0x120 |

This is our **early exit**: "If transmittance < 0.001, skip to the end." Predicate registers (`P0` through `P6`) are 1-bit true/false flags. You set them with comparison instructions, then use `@P0` to conditionally execute any instruction.

> **Think about it**: After reading these examples, go back and look at the instruction table below. Each instruction should now feel familiar — not just "FFMA does multiply-add" but "FFMA R8, R12, R5, R8 accumulates a weight×input product into a running sum, and P0 can conditionally skip it." That's the difference between *reading* SASS and *understanding* it.

### The complete SASS instruction reference (for this project)

| SASS Instruction | What it does | Latency | Example |
|-----------------|-------------|---------|---------|
| **FFMA** | Fused float multiply-add: `d = a*b + c` | ~4.5 cycles | Dot products, interpolation |
| **LDG.E.64** | Load 64 bits (2 floats) from global memory | ~200 cycles (L2) | Hash table lookups |
| **LDS** | Load from shared memory | ~23 cycles | Weight matrix access |
| **STG** | Store to global memory | ~200 cycles | Writing results |
| **LOP3** | 3-input logic operation (AND, OR, XOR — any combo) | ~4 cycles | Hash function XOR |
| **IMAD** | Integer multiply-add: `d = a*b + c` | ~4 cycles | Address calculation, hashing |
| **MUFU.EX2** | Hardware 2^x approximation (special function unit) | ~6 cycles | Fast exp(), sigmoid |
| **FMNMX** | Float min/max | ~4 cycles | ReLU activation |
| **FSETP** | Float compare → set predicate flag | ~4 cycles | Early exit test |
| **IADD3** | 3-input integer add | ~4 cycles | Address arithmetic |
| **SHF** | Funnel shift (bit manipulation) | ~4 cycles | Address computation |
| **F2I** | Float → integer conversion | ~4 cycles | Grid coordinate floor |
| **SELP** | Select based on predicate | ~4 cycles | Conditional value pick |

---

## The NeRF Pipeline

NeRF (Neural Radiance Fields) turns photographs into 3D scenes. The rendering pipeline has three stages:

```
  3D Point Position (x,y,z)
        │
        ▼
 ┌──────────────────┐
 │ 1. HASH GRID     │  "Encode position into features"
 │    ENCODING       │  Input:  3 floats (x,y,z)
 │                   │  Output: 24 floats (features)
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐
 │ 2. MLP FORWARD   │  "Decode features into color"
 │    (Neural Net)   │  Input:  27 floats (24 features + 3 view dir)
 │                   │  Output: 4 floats (R, G, B, density)
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐
 │ 3. VOLUME        │  "Composite samples into pixel color"
 │    RENDERING      │  Input:  64 samples × (RGBA, dt) per ray
 │                   │  Output: 1 pixel color (RGBA)
 └──────────────────┘
```

For a 512×512 image, that's **262,144 rays**, each sampling ~64 points along the ray. That's **16.7 million** hash grid + MLP evaluations per frame.

---

## Kernel 1: Hash Grid Encoding

### What it does

Given a 3D point (x, y, z), encode it into a 24-dimensional feature vector by looking up learned features at multiple resolution levels.

### The Algorithm (step by step)

For each of 12 resolution levels:

```
1. SCALE:  Multiply (x,y,z) by the level's resolution
           Level 0: ×16,  Level 1: ×24,  Level 2: ×36, ... Level 11: ×1388

2. FLOOR:  Find which grid cell the point falls in
           grid_x = floor(scaled_x)
           grid_y = floor(scaled_y)
           grid_z = floor(scaled_z)

3. FRACT:  How far into the cell? (0.0 to 1.0)
           wx = scaled_x - floor(scaled_x)    ← used for interpolation later

4. HASH:   Find the 8 corners of the cube the point is inside
           For each corner (±1 in x,y,z):
             hash = (corner_x × 73856093) XOR (corner_y × 19349663) XOR (corner_z × 83492791)
             index = hash mod 131072          ← index into hash table

5. LOAD:   Read 2 learned features from hash table for each corner
           8 corners × 2 features = 16 values loaded

6. LERP:   Trilinear interpolation (weighted average based on distance)
           4 lerps on X-axis → 2 lerps on Y-axis → 1 lerp on Z-axis
           Result: 2 features for this level
```

After all 12 levels: 12 × 2 = 24 output features.

### Why it's hard to optimize

This kernel is **memory-bound**. Each level does 8 random lookups into a 12MB hash table. "Random" means the addresses are scattered (not sequential), so the GPU cache can't help much. Each lookup takes ~200 cycles to reach L2 cache.

The compute is cheap (a few multiplies, some XORs, interpolation). The bottleneck is waiting for data.

> **Think about it**: 12 levels × 8 corners × 200 cycles = 19,200 cycles just waiting for memory *if* done one at a time. The total math work is ~850 cycles. That means the hardware spends **95% of its time sitting idle**, waiting for data. The actual calculations take only 5% of the time. This is why "making the math faster" wouldn't help this kernel — the math is already basically free.

### Our optimizations explained

**1. float2 vectorized loads (16 × LDG.32 → 8 × LDG.64)**

Each hash table entry has 2 floats. Instead of loading them one at a time:
```
load float[0]     ← 1 instruction, 4 bytes
load float[1]     ← 1 instruction, 4 bytes
```
We load them together:
```
load float2       ← 1 instruction, 8 bytes
```
Same data, half the instructions. The GPU's memory bus naturally handles 64-bit loads at the same throughput as 32-bit loads — you get double the data per instruction for free.

**In SASS**: `LDG.E.64.CONSTANT` instead of two `LDG.E.CONSTANT`

**2. Non-volatile asm (the v1 disaster and fix)**

In v1, we used `asm volatile` on every instruction. The `volatile` keyword tells the compiler: "Don't reorder this instruction — execute it exactly here."

Why that's devastating for memory-bound code:
```
// v1 (volatile): compiler MUST keep this order
load A          ← wait 200 cycles for data
load B          ← wait 200 cycles
XOR             ← 4 cycles
multiply        ← 4 cycles
load C          ← wait 200 more cycles
```

What we want the compiler to do (and what v2-v3 allows):
```
// v2+ (non-volatile): compiler CAN reorder
load A          ← start loading (200 cycles)
load B          ← start loading (200 cycles) — overlaps with A!
load C          ← start loading (200 cycles) — overlaps with A and B!
XOR             ← 4 cycles (fills time while waiting for loads)
multiply        ← 4 cycles (fills time while waiting for loads)
use A           ← by now, A has arrived
use B           ← by now, B has arrived
```

This is called **latency hiding** — doing useful work while waiting for slow memory.

**3. Software pipelining across level pairs**

We process two levels at the same time:
```
Phase 1: Hash + issue loads for Level 0 AND Level 1 (16 loads in flight)
Phase 2: Trilinear interpolation for Level 0 AND Level 1
         (by now, loads from Phase 1 have returned)
```

This doubles the number of outstanding memory requests, giving the memory system more chances to serve data while we compute.

**4. LOP3 — 3-input XOR in one instruction**

The hash function does `a XOR b XOR c`. On older GPUs, that requires two instructions:
```
temp = a XOR b
hash = temp XOR c
```

Ada Lovelace (SM 8.9) has **LOP3** — a single instruction that computes ANY 3-input boolean function. We use truth table `0x96` (which encodes XOR-XOR):
```
hash = LOP3(a, b, c, 0x96)    ← 1 instruction!
```

### The Full Optimization Journey: v1 → v2 → v3

This is the most instructive part of the project — we went from **slower than the compiler** to **11% faster**, and the lessons along the way teach more than the final result.

#### v1: The volatile disaster (0.69x — 31% SLOWER than the compiler)

Our first version wrapped every single inline PTX instruction in `asm volatile(...)`. Here's what that looked like:

```c
// v1: EVERY operation was volatile
asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(hash) : "r"(a), "r"(b), "r"(c));
asm volatile("ld.global.v2.f32 {%0,%1}, [%2];" : "=f"(f0), "=f"(f1) : "l"(addr));
asm volatile("fma.rn.f32 %0, %1, %2, %3;"     : "=f"(r)    : "f"(w), "f"(a), "f"(b));
```

**Why `volatile` kills memory-bound kernels:**

The GPU can have ~20-30 memory loads in-flight simultaneously per warp. This is how it hides the 200-cycle L2 latency — issue a load, then do something else for 200 cycles, and the data will be ready when you need it (this is called **latency hiding**).

`asm volatile` creates a **scheduling barrier**. The compiler must execute that instruction at that exact point — it cannot move it earlier or later. When every load is volatile, the pipeline collapses:

```
Cycle 0:     load_volatile(A)   ← issue load
Cycle 1-199:  ... 199 cycles of NOTHING (stall!) ...
Cycle 200:   use A
Cycle 201:   load_volatile(B)   ← issue load
Cycle 202-400: ... 199 more cycles of NOTHING ...
Cycle 401:   use B
```

The compiler's version (no volatile) freely reorders:
```
Cycle 0:     load(A)       ← issue (non-blocking)
Cycle 1:     load(B)       ← issue (non-blocking, overlaps with A)
Cycle 2:     load(C)       ← issue (non-blocking, overlaps with A and B)
Cycle 3-6:   XOR, IMAD...  ← do math while all 3 loads are in flight
...
Cycle ~200:  use A          ← data arrived!
Cycle ~201:  use B          ← also arrived!
```

**We serialized what should have been parallel.** The GPU was idle 95% of the time.

#### v2: Removing the barriers (1.03x — back to parity)

The fix was conceptually simple but required rethinking our approach:

| Component | v1 (broken) | v2 (fixed) | Why |
|-----------|-------------|------------|-----|
| Hash XOR | `asm volatile("lop3...")` | `asm("lop3...")` | Non-volatile: compiler CAN reorder |
| Trilinear interp | `asm volatile("fma...")` | `a + t*(b-a)` (C code) | Compiler generates FMA, full optimization freedom |
| Feature loads | `asm volatile("ld.global...")` | `__ldg()` intrinsic | Compiler knows it's a load, schedules optimally |
| Feature stores | `asm volatile("st.global...")` | `output[i] = val` (C) | Compiler handles store scheduling |

Key insight: **non-volatile `asm` still guarantees the instruction is emitted** (compiler won't delete it), but lets the compiler freely reorder it relative to other instructions. This is the sweet spot — you get your custom instruction AND the compiler's scheduler.

We kept `asm("lop3...")` for the 3-input XOR because C has no equivalent (two C XORs = 2 SASS instructions vs LOP3 = 1). Everything else went back to C or intrinsics.

Result: 1.03x — we'd undone the self-inflicted damage and caught up to the compiler.

#### v3: Beating the compiler (1.11x)

Two optimizations pushed us past:

**Optimization 1: float2 vectorized loads**

Each hash table entry is 2 floats (8 bytes). The compiler loaded them as two separate 32-bit loads:
```
// Compiler generates:
LDG.E.CONSTANT R22, [R4.64]        // 1st float, 4 bytes
LDG.E.CONSTANT R26, [R4.64+0x4]    // 2nd float, 4 bytes
```

We forced a single 64-bit load:
```c
float2 v = __ldg((const float2*)(level_ptr + hash_index * 2));
// Compiles to:
// LDG.E.64.CONSTANT R22, [R22.64]    // both floats, 8 bytes, 1 instruction
```

Same data, half the instructions. The memory bus handles 64-bit loads at the same throughput — you get twice the data per instruction.

**Across the whole kernel**: 96 loads instead of ~192. That's 96 fewer instructions for decode/issue/retire.

**Optimization 2: Software pipelining across level pairs**

Instead of processing one level at a time, we interleave two:
```
// Process levels in pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
for (pair = 0; pair < 6; pair++) {
    // Issue 16 loads simultaneously (8 for level A + 8 for level B)
    hash_and_load(level_A);
    hash_and_load(level_B);
    
    // The hash computation for level B = ~100 cycles of ALU work
    // By now, level A's loads (issued first) have had time to arrive
    
    // Trilinear interpolation for both
    trilinear(level_A);   // loads arrived ~200 cycles ago
    trilinear(level_B);   // loads arrived ~100 cycles ago
}
```

16 simultaneous memory requests (vs 8) means the memory controller has more work to pipeline. And the ALU work for level B's hash computation acts as free latency hiding for level A's loads.

#### The scoreboard

| Version | Speedup | Key Change | SASS Signature |
|---------|---------|------------|----------------|
| **v1** | **0.69x** | All `asm volatile` → serialized loads | Interleaving destroyed |
| **v2** | **1.03x** | Non-volatile asm + C trilinear + `__ldg()` | Load reordering restored |
| **v3** | **1.11x** | float2 loads + level-pair pipelining at `-O2` | 96 × LDG.E.64.CONSTANT |
| *ref* | *1.00x* | Compiler's auto-generated code | ~192 × LDG.E.32 |

**The lesson**: For memory-bound kernels, the compiler's instruction scheduler is your *ally*, not your enemy. Use inline asm surgically (for instructions C can't express), and let the compiler handle the rest.

> **Think about it**: This journey is the most important lesson in the entire project. It's tempting to think "I'll write everything in assembly and it'll be faster." v1 proves that's wrong. The compiler's scheduler is an incredibly powerful tool — it was written by people who've spent decades optimizing GPU instruction ordering. The winning strategy isn't "replace the compiler" — it's "help the compiler where it's weak, and stay out of its way everywhere else." We added exactly 2 things the compiler can't do (LOP3 and float2 loads) and let it handle the other ~700 instructions.

---

## Kernel 2: MLP Forward Pass

### What it does

A tiny fully-connected neural network:
```
Input (27 values) → Layer 0 (64 neurons, ReLU) → Layer 1 (64 neurons, ReLU) → Output (4 values, sigmoid)
```

Each layer computes: `output = activation(weight_matrix × input + bias)`

### The math

For one neuron in Layer 0, the computation is a **dot product** of 27 inputs with 27 weights, plus a bias:

```
result = bias + w0*x0 + w1*x1 + w2*x2 + ... + w26*x26
```

That's 27 multiply-adds for one neuron. Layer 0 has 64 neurons → 64 × 27 = **1,728 multiply-adds**.
Layer 1: 64 × 64 = **4,096 multiply-adds**.
Output: 4 × 64 = **256 multiply-adds**.

Total: ~6,080 multiply-adds per sample. At 262K samples, that's 1.6 billion operations.

> **Think about it**: 1.6 billion operations sounds insane, but the RTX 4070 Ti Super can do 337,920 multiply-adds *per clock cycle* across all SMs. At 2640 MHz boost clock, that's ~892 billion multiply-adds per second. So 1.6 billion takes about 1.8 milliseconds — *if* the pipeline is fully utilized. The question is: can we keep all those math units fed?

### Why this kernel is fast (3.16x speedup)

This kernel is **compute-bound** — the bottleneck is raw math throughput, not memory. That's where manual optimization shines.

**1. 8-wide ILP (Instruction Level Parallelism)**

The GPU's FFMA unit has a 4.5-cycle latency. If you do one multiply-add at a time, you waste 3.5 cycles waiting between each one:

```
// Naive: 1 accumulator (serial dependency chain)
acc += w0 * x0       ← 4.5 cycles (then wait...)
acc += w1 * x1       ← 4.5 cycles (then wait...)
acc += w2 * x2       ← 4.5 cycles (then wait...)
// Throughput: 1 FFMA per 4.5 cycles
```

Instead, we compute 8 neurons simultaneously with 8 independent accumulators:

```
// Our approach: 8 independent accumulators
acc[0] += w0_0 * x0     ← cycle 0
acc[1] += w0_1 * x0     ← cycle 1 (doesn't depend on acc[0]!)
acc[2] += w0_2 * x0     ← cycle 2
acc[3] += w0_3 * x0     ← cycle 3
acc[4] += w0_4 * x0     ← cycle 4
acc[5] += w0_5 * x0     ← cycle 5 (acc[0] result ready by now!)
acc[6] += w0_6 * x0     ← cycle 6
acc[7] += w0_7 * x0     ← cycle 7
acc[0] += w1_0 * x1     ← cycle 8 (no stall! acc[0] was ready at cycle 5)
// Throughput: 1 FFMA per cycle — fully pipelined!
```

The compiler *could* theoretically do this but doesn't — it's conservative about register usage and instruction reordering. We manually force it.

> **Think about it**: Why doesn't the compiler do 8-wide ILP on its own? Because it's a **tradeoff**. 8 accumulators means 8 registers just for those running totals, plus registers for weights, inputs, and temporaries. If you use too many registers, the SM can't fit as many threads, which hurts latency hiding for memory loads. The compiler plays it safe — it uses fewer registers to keep occupancy high. But for this kernel, the bottleneck is math throughput, not memory latency, so the tradeoff is worth it. This is exactly the kind of judgment call a human can make but a compiler can't — we know *which* bottleneck matters.

**In SASS**: The kernel generates 6,249 FFMA instructions — nearly everything is fused multiply-add.

**2. Shared memory weight tiling**

The weight matrix (6,212 floats = ~24 KB) is loaded into **shared memory** at the start of each block. Shared memory is ~8.7x faster than global memory (23 vs 200 cycles).

All 128 threads in a block cooperatively load the weights, then everyone reads from shared memory during computation:
```
// Cooperative loading (all threads work together)
for (int i = threadIdx; i < weight_count; i += block_size)
    shared_weights[i] = global_weights[i];
__syncthreads();  // wait for everyone to finish

// Now all 128 threads read from fast shared memory
for (int j = 0; j < 27; j++)
    acc += shared_weights[neuron][j] * input[j];  // LDS.32 (~23 cycles)
```

**In SASS**: 1,553 LDS (shared memory loads) instead of global memory loads.

**3. FMNMX ReLU — one instruction instead of a branch**

ReLU is `max(x, 0)`. A naive if/else:
```
if (x > 0) result = x;
else result = 0;
```
That's a branch, which causes thread divergence on a GPU (some threads in the warp go one way, others go another — both paths execute serially).

Our approach — single instruction, no branch:
```asm
FMNMX result, x, RZ, !PT    // max(x, 0) — RZ is the zero register
```

**In SASS**: 130 FMNMX instructions (64 neurons × 2 layers + 2 extras).

**4. MUFU.EX2 fast sigmoid**

The output layer uses sigmoid: `1 / (1 + exp(-x))`. Standard exp() implementation is ~30 instructions.

We rewrite it using the hardware's fast 2^x unit (MUFU.EX2):
```
sigmoid(x) = 1 / (1 + 2^(-x × log2(e)))
```

That's: 1 FMUL + 1 MUFU.EX2 + 1 FADD + 1 MUFU.RCP = **4 instructions** total.

**In SASS**: 14 MUFU instructions across the kernel.

### Results

**3.16x speedup** over the reference CUDA kernel, with max error of only 1.19 × 10^-7 (effectively identical results — the tiny error comes from MUFU.EX2 being an approximation).

---

## Kernel 3: Volume Rendering

### What it does

For each ray (one per pixel), combine all the samples along the ray into one final pixel color. This is **alpha compositing**, the same technique used in video games, Photoshop layers, and movie VFX.

### The algorithm

```python
transmittance = 1.0    # starts fully transparent (light passes through)
color = (0, 0, 0)      # starts black

for each sample along the ray:
    alpha = 1 - exp(-density * step_size)          # how opaque is this sample?
    weight = transmittance * alpha                  # how much does it contribute?
    color += weight * sample_color                  # add weighted color
    transmittance *= (1 - alpha)                    # reduce light passing through
    
    if transmittance < 0.001:                       # already opaque?
        break                                       # stop — nothing behind matters
```

### Our optimizations explained

**1. MUFU.EX2 fast exponential (2 instructions instead of ~20)**

The `exp(-x)` computation is the inner-loop bottleneck. We rewrite it as:
```
exp(-x) = 2^(-x × log2(e))
```
And use the GPU's hardware exponent unit:
```asm
FMUL    neg_xl2e, x, -1.4427    // multiply by -log2(e)
MUFU.EX2 result, neg_xl2e       // hardware 2^x (built-in approximation)
```

Two instructions, ~8 cycles. The standard library `expf()` compiles to ~20+ instructions.

**2. Reusing the exponential result**

Notice: `(1 - alpha) = exp(-sigma * dt)`, which we already computed for alpha!
```
neg_exp = exp(-sigma * dt)
alpha   = 1 - neg_exp
...
transmittance *= neg_exp    // reuse! instead of computing (1 - alpha) again
```

This eliminates a subtraction in the hot loop.

**3. Predicated early exit**

When transmittance drops below 0.001, further samples contribute < 0.1% to the pixel. We stop:
```asm
FSETP.LT  P0, _, transmittance, 0.001  // set predicate if T < threshold
SELP      done, 1, 0, P0               // convert to integer boolean
// then check 'done' and break
```

This is done without divergent branches — the predicate register is set and read without a branch instruction.

**4. Vectorized load/store**

Input RGBA is loaded as `float4` (128-bit, one instruction) and output is stored the same way:
```asm
LDG.E.128  {r, g, b, sigma}, [ptr]    // load 4 floats at once
STG.E.128  [out], {r, g, b, alpha}    // store 4 floats at once
```

### Results

**1.53x speedup**, max error 2.98 × 10^-7 (from MUFU.EX2 approximation error accumulating over 64 steps — still visually identical).

> **Think about it**: The volume renderer's 1.53x speedup is "free performance" — we didn't change the algorithm at all. Same front-to-back compositing, same early exit, same formula. We just replaced the *implementation* of `exp()` with a 2-instruction hardware version and reused a value we already computed. This is pure assembly-level optimization: the algorithm is identical, but the instructions are better chosen.

---

## How to Think About Performance

If you've read this far, you understand each kernel's optimizations. But how do you develop the *instinct* to know what to optimize? Here's the mental framework.

### Step 1: Find the bottleneck

Before optimizing anything, ask: **"What is the hardware waiting on?"**

- Is it waiting for **data from memory**? → Memory-bound. Optimize loads.
- Is it waiting for **math to finish**? → Compute-bound. Optimize instructions.
- Is it waiting for **threads to reconverge after a branch**? → Divergence-bound. Remove branches.

You can often figure this out with napkin math:

```
Bytes loaded per thread × number of threads = total memory traffic
Total memory traffic ÷ memory bandwidth = minimum time (memory)

Math ops per thread × number of threads = total compute
Total compute ÷ peak compute throughput = minimum time (compute)

Whichever minimum is LARGER → that's your bottleneck
```

For the hash grid:
- Memory: 768 bytes × 262,144 threads = 201 MB. At 672 GB/s bandwidth → 0.30 ms
- Compute: 852 ops × 262,144 threads = 223M ops. At 892 billion ops/s → 0.00025 ms
- Memory time is **1,200x larger** → clearly memory-bound

For the MLP:
- Memory: 124 bytes × 262,144 threads = 32.5 MB → 0.048 ms
- Compute: 6,080 ops × 262,144 threads = 1.6B ops → 1.8 ms
- Compute time is **37x larger** → clearly compute-bound

### Step 2: Know what the hardware can do

Once you know the bottleneck, look at what hardware features you're *not* using:

- **Not using wide loads?** Switch from `float` to `float2`/`float4`.
- **Not using shared memory?** If many threads read the same data, load it into shared memory once.
- **Not using special functions?** MUFU can do `exp`, `log`, `sin`, `cos`, `sqrt`, `rcp` in 1-2 instructions.
- **Not using ILP?** If your code is a long serial dependency chain, break it into independent chains.
- **The compiler missing a trick?** Check the SASS — is it using 2 instructions where 1 would do (like LOP3 for 3-way XOR)?

### Step 3: Check the SASS

The final truth is always in the SASS. Compile your kernel, run `cuobjdump -sass`, and look at what the hardware will actually execute:

```powershell
# Generate just the binary
nvcc -arch=sm_89 -O2 -cubin -o kernel.cubin source.cu

# Disassemble to readable SASS
cuobjdump -sass kernel.cubin > output.sass
```

Things to look for:
- **LDG count**: How many global loads? Can any be combined (32→64 bit)?
- **FFMA count**: Is the math fully fused? (separate MUL + ADD = missed opportunity)
- **LDL/STL**: These are "local memory" spills — the compiler ran out of registers and is dumping to slow memory. Very bad.
- **BRA instructions**: Branches. Every branch is a potential divergence point.
- **Instruction interleaving**: Are LDG instructions clustered at the top (good: loads issued early) or scattered between uses (bad: loads issued just-in-time)?

### Step 4: Measure, don't guess

Always benchmark. Our v1 hash grid *felt* like it should be fast (we wrote every instruction!), but it was 31% slower. The only way to know is to measure:

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
for (int i = 0; i < 100; i++)
    my_kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Average: %.3f ms\n", ms / 100.0f);
```

100 iterations, averaged. Discard 10 warmup runs (the GPU needs time to ramp up its clock speed). Compare against the reference kernel compiled in the *same binary* with the *same flags* — otherwise you're measuring compiler differences, not code differences.

> **Think about it**: The entire optimization process is a cycle: **profile → identify bottleneck → hypothesize fix → implement → measure → verify in SASS → repeat**. At no point do you guess. You look at the hardware, look at the numbers, and make a targeted change based on evidence. That discipline — not raw knowledge — is what separates a GPU optimization engineer from someone who just knows what SASS instructions are.

---

## Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **SASS** | The GPU's native binary instruction set (like x86 for CPUs) |
| **PTX** | An intermediate assembly language that maps ~1:1 to SASS |
| **Kernel** | A function that runs on the GPU across thousands of threads |
| **Thread** | One worker executing the kernel code on one piece of data |
| **Warp** | 32 threads that execute in lockstep (always together) |
| **Block** | A group of warps (64-1024 threads) that share fast memory |
| **SM** | Streaming Multiprocessor — one "department" of the GPU |
| **FFMA** | Fused Float Multiply-Add: `d = a*b + c` in one instruction |
| **LDG** | Load from Global memory (slow, ~200 cycles) |
| **LDS** | Load from Shared memory (fast, ~23 cycles) |
| **MUFU** | Multi-Function Unit — hardware exp, log, sin, cos, reciprocal |
| **LOP3** | 3-input Logic Operation — any boolean combo in 1 instruction |
| **ILP** | Instruction Level Parallelism — keeping the pipeline full |
| **Latency hiding** | Doing useful work while waiting for slow memory |
| **Register** | Fastest storage — each thread has its own (0 cycle access) |
| **Shared memory** | Fast storage shared by all threads in a block (23 cycles) |
| **Global memory** | GPU VRAM — large but slow (200-400 cycles) |
| **Compute-bound** | Limited by math speed (like MLP: lots of FFMA) |
| **Memory-bound** | Limited by data transfer speed (like hash grid: lots of LDG) |
| **asm volatile** | Tells compiler "do NOT reorder this instruction" |
| **Software pipelining** | Starting the next batch of work before finishing current |
| **Trilinear interpolation** | Weighted average across 8 corners of a 3D cube |
| **Hash table** | Array where the index is computed by a hash function |
| **Alpha compositing** | Blending colors front-to-back with opacity |
| **Sigmoid** | S-shaped function that maps any value to 0-1 range |
| **ReLU** | `max(x, 0)` — sets negative values to zero |
| **Neural network** | Layers of multiply-add + activation (learned from data) |

---

## Where to Go From Here

If you've read and understood this guide, you now know more about GPU assembly than 99% of programmers. Here's how to keep going:

### Practice: read SASS yourself

1. Write a simple CUDA kernel (add two arrays, multiply a matrix).
2. Compile it: `nvcc -arch=sm_89 -O2 -cubin -o test.cubin test.cu`
3. Disassemble: `cuobjdump -sass test.cubin > test.sass`
4. Read the SASS. Match each instruction back to your source code.
5. Try changing your code and see how the SASS changes.

This builds the muscle memory of "when I write *this* C code, the GPU executes *those* instructions."

### Resources

- **NVIDIA PTX ISA docs**: Search "PTX ISA 8.x" — the official reference for every PTX instruction. Dense but authoritative.
- **CUDA C Programming Guide**: Start with chapters on shared memory, occupancy, and the execution model.
- **Instant-NGP paper**: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" (Muller et al., 2022) — the algorithm we optimized.
- **NeRF paper**: "NeRF: Representing Scenes as Neural Radiance Fields" (Mildenhall et al., 2020) — the original technique.
- **Our SASS RE toolkit**: See `src/sass_re/` in this repo — 9 probe kernels and microbenchmarks for measuring real instruction latencies on your own GPU.

### The most important thing

Assembly isn't about memorizing opcodes. It's about building a **mental model** of what the hardware is doing — so when you look at high-level code, you can see the instructions underneath, the memory accesses, the stalls, the wasted cycles. Once you have that mental model, you can see optimizations that are invisible to everyone else.

You don't need to write assembly every day. But understanding it changes how you write *everything*.

---

*Built by Umut Korkmaz. RTX 4070 Ti Super, CUDA 13.1, March 2026.*
