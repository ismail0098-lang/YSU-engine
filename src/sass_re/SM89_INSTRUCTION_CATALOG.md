# Ada Lovelace SM 8.9 SASS Instruction Reference

Manifest-backed inventory of SASS mnemonics observed on NVIDIA Ada Lovelace
SM 8.9 (RTX 4070 Ti), with measured latencies and compilation flag requirements.

- Generated: 2026-03-20
- Hardware: NVIDIA GeForce RTX 4070 Ti (AD104, SM 8.9, 60 SMs, 2625 MHz)
- Compiler: CUDA 13.1 (nvcc V13.1.115)
- Probes: 349 recursive `probe_*.cu` files in `src/sass_re/probes/`
- Manifest non-skip entries: 344
- Compile-enabled entries in current manifest: 343
- Compile-enabled entries in latest full refresh: 341
- Microbenchmarks: 15 files in `src/sass_re/microbench/`
- Canonical sweep model: recursive manifest + 6 comparison lanes
- Latencies: measured via 512-deep dependent chains, ncu cross-validated
- Latest aggregate run:
  `src/sass_re/results/runs/full_recursive_20260320_182500`
- Latest full-corpus flag sweep:
  `src/sass_re/results/runs/flag_sweep_postfix_parallel6x4_20260320_233059`
- Canonical optimized frontier: 379 raw mnemonics
- Strongest discovery-lane frontier: 382 raw mnemonics (`--maxrregcount=32`)
- Full-corpus verified flag movers:
  - `--maxrregcount=32` -> `UISETP.EQ.U32.XOR`, `UISETP.GE.U32.AND`,
    `UISETP.GT.AND`
  - `--restrict` -> `I2F.S8`, `LDG.E.U16.CONSTANT`, `LDG.E.U8.CONSTANT`,
    `LDL.LU`
  - `-Xptxas -dlcm=cg` -> `LD.E.64.STRONG.GPU`,
    `LDG.E.128.STRONG.GPU`, `LDG.E.64.STRONG.GPU`,
    `LDG.E.S16.STRONG.GPU`, `LDG.E.S8.STRONG.GPU`,
    `LDG.E.U16.STRONG.GPU`, `LDG.E.U8.STRONG.GPU`

Mnemonic naming note:
- `raw_sass` is the exact `cuobjdump -sass` spelling.
- Some semantic families alias to different raw spellings on this toolchain.
  Example: logical warp reduction AND is emitted as bare `REDUX`.

Latency notation:
- Exact values (e.g., "4.53 cy") are from direct measurement
- Approximate values (e.g., "~4.53 cy") are inferred from same-pipeline instructions
- Empty latency = not directly measured (SASS-only probe, no timing chain)
- Section heading counts may lag the exact row inventory during active probing;
  when they disagree, treat the table rows and the total count below as the
  source of truth.

Library-mined provisional note:
- The canonical inventory below is still organized around direct local
  `sm_89` probe observations.
- A separate architecture-filtered cuDNN mining pass over packaged `sm_86`
  cubins on this machine is recorded under
  `src/sass_re/results/runs/cudnn_library_sm86_mining_20260320_103900`.
- A direct local confirmation tranche under
  `src/sass_re/results/runs/direct_confirm_20260320` has now promoted
  `HFMA2.RELU`, `HMMA.1688.F32.TF32`, `LDSM.16.MT88.4`,
  `LDGSTS.E.LTC128B.128`, `LDGSTS.E.LTC128B.128.ZFILL`,
  `LDGSTS.E.BYPASS.LTC128B.128`, `LDGSTS.E.BYPASS.LTC128B.128.ZFILL`, and
  `HMNMX2.NAN`, `F2FP.BF16.F32.PACK_AB`, and
  `F2FP.RELU.BF16.F32.PACK_AB` into the canonical direct `sm_89` inventory.
- A dedicated alias follow-up under
  `src/sass_re/results/runs/f2fp_alias_followup_20260320` now shows that even
  the closest local cuDNN-like source shape, packing a scalar with a compile-
  time zero lane followed by `STG.E.U16`, still renders as
  `F2FP.F16.F32.PACK_AB` or `F2FP.BF16.F32.PACK_AB` in both `cuobjdump` and
  `nvdisasm`.
- The remaining higher-value library-only candidates are therefore currently
  best grouped as:
  - likely library/disassembler aliases adjacent to direct local forms:
    `F2FP.PACK_AB`, `F2FP.BF16.PACK_AB`
  - still-unreproduced from direct source/IR predicate/uniform variants:
    `P2R.B1`, `P2R.B2`, `P2R.B3`
  - already directly observed but adjacent to the same frontier:
    `R2P`, `USHF.L.U64.HI`, `ULOP3.LUT`
- Follow-up direct probes `probe_p2r_mov_pack_inline_ptx.cu` and
  `probe_tensor_uniform_predicate.cu` now provide sharpened negative evidence:
  the local compiler still lowers those patterns to `ISETP`/`SEL`/`LOP3` or
  `@!UPT UIADD3` scaffolding instead of surfacing `P2R.B*` or `UPLOP3.LUT`.
- A stricter uniform follow-up under
  `src/sass_re/results/runs/p2r_uniform_strict_20260320` now directly confirms
  `ULEA.HI.X.SX32` on local `sm_89`, but still only reaches `USHF.L.U32` plus
  GPR-space `SHF.L.U64.HI` rather than fully uniform `USHF.L.U64.HI`.
- The latest follow-up under
  `src/sass_re/results/runs/uplop3_p2r_followup_20260320` sharpens those last
  two gaps further. `probe_p2r_vector_pack_inline_ptx.cu` uses
  predicate-derived byte lanes plus `mov.b32` vector packing before merging
  into a live destination register, but still lowers to
  `ISETP`/`SEL`/`PRMT`/`LOP3` glue instead of `P2R.B1/B2/B3`. Meanwhile
  `probe_uniform_async_tensor_pipeline.cu` reproduces the local cuDNN-like
  uniform logic neighborhood by combining
  `LDGSTS.E.BYPASS.LTC128B.128`, `HMMA.1688.F32.TF32`, `@!UPT UIADD3`, and
  dense `PLOP3.LUT` with `0x40`/`0x80` immediates. Direct local `ULOP3.LUT`
  is already observed elsewhere in the uniform-path corpus, but the exact
  library-mined `UPLOP3.LUT` spelling still does not appear verbatim in local
  direct probes.
- A stricter follow-up under
  `src/sass_re/results/runs/p2r_uplop3_stage_followup_20260320` pushes both
  motifs harder again. `probe_p2r_banked_reload.cu` mirrors a longer-lived
  banked predicate-save lifecycle, but still lowers to
  `ISETP`/`SEL`/`LOP3`/`PRMT`/`IMAD` glue rather than `P2R.B1/B2/B3`.
  `probe_uniform_stage_toggle_pipeline.cu` rebuilds a tighter software-
  pipelined tensor mainloop and directly reproduces
  `HMMA.1688.F32.TF32`, `LDGSTS.E.BYPASS.LTC128B.128`,
  `LDGSTS.E.BYPASS.LTC128B.128.ZFILL`, dense `PLOP3.LUT`, and repeated
  `@!UPT UIADD3` / `R2UR` scaffolding, but still does not emit raw
  `UPLOP3.LUT`.
- An exact predicate/uniform tranche under
  `src/sass_re/results/runs/predicate_uniform_frontier_20260321_031500`
  reconciles an important erratum: base `R2P` is already directly observed in
  `probe_transcendentals.sass` and `probe_fp64_transcendentals.sass`. The new
  exact probes still do not surface `P2R.B1/B2/B3`, `UPLOP3.LUT`, or
  `USHF.L.U64.HI`, but they do reproduce stronger nearby local forms such as
  `ULOP3.LUT`, `USEL`, `UISETP.NE.U32.AND`, `ULDC.64`,
  `HMMA.1688.F32.TF32`, and `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`.
- A tighter same-carrier predicate follow-up under
  `src/sass_re/results/runs/p2r_two_stage_bank_20260321_110000` now directly
  reproduces `P2R R0, PR, R0, 0x7f` on local `sm_89`. That closes the broader
  full-mask same-carrier predicate-pack shape and leaves the byte-qualified
  `P2R.B1/B2/B3` family as the real remaining direct-local gap in this corner.
- A newer literal byte-bank extension under
  `src/sass_re/results/runs/p2r_b1_literal_cudnn_20260321_115500` and
  `src/sass_re/results/runs/p2r_b23_literal_cudnn_20260321_122400` pushes that
  same-carrier neighborhood across byte-one, byte-two, and byte-three rewrite
  shapes. All three still lower through `ISETP` + `SEL` + `LOP3.LUT` glue
  rather than surfacing `P2R.B1`, `P2R.B2`, or `P2R.B3`, which tightens the
  boundary around the whole byte-qualified family.
- A final uniform-u64 follow-up under
  `src/sass_re/results/runs/ushf_u64_hi_final_20260320` tests cleaner
  constant-backed and parameter-backed 64-bit shift families. It directly
  confirms `ULEA` and `ULEA.HI.X` in the uniform address path, but still does
  not reproduce `USHF.L.U64.HI`; the compiler continues to use GPR-space
  `SHF.L.U64.HI` or `SHF.R.U64` for the actual 64-bit-high shift.
- A direct prefix cleanup tranche under
  `src/sass_re/results/runs/prefix_direct_followup_20260320` now promotes
  `LDC.U8` and `LDC.S8` into the canonical direct local inventory. The same
  tranche also tested a wider `UISETP` sweep, but those source shapes still
  lower to ordinary `ISETP` plus `SEL` rather than surfacing the broader
  library-mined `UISETP.*` family.
- A final CUTLASS-like tensor follow-up under
  `src/sass_re/results/runs/cutlass_predicate_pipeline_20260320` now produces
  the strongest direct local optimized neighborhood around the unresolved
  predicate cluster: base `P2R`, `PLOP3.LUT`, `HMMA.1688.F32.TF32`,
  `LDGSTS.E.BYPASS.LTC128B.128*`, and `@!UPT UIADD3`. Even there, it still does
  not emit `P2R.B1/B2/B3` or `UPLOP3.LUT`.
- A final direct-local cluster follow-up under
  `src/sass_re/results/runs/final_cluster_followup_20260320` adds a
  reconvergence-heavy predicate-save path and a tighter uniform stage-FSM path
  with loop-carried U64 rebasing. A newer exact follow-up under
  `predicate_uniform_frontier_20260321_024500` finally closes one of those
  gaps by directly confirming `USHF.L.U64.HI` on local `sm_89`, while
  `P2R.B1/B2/B3` and `UPLOP3.LUT` remain unreproduced.
- The strongest still-provisional prefix-driven candidates from library mining
  are now:
  - direct source/IR gap: `P2R.B1`, `P2R.B2`, `P2R.B3`, `UPLOP3.LUT`
  - cubin-side materialization now exists for `P2R.B1`, `P2R.B2`, and
    `P2R.B3`; `UPLOP3.LUT` remains the stronger unresolved non-cubin frontier
  - `LDC.S8`, `LDC.U8` are no longer provisional; they are now direct local
    confirmations
  - second-tier candidates worth future direct probing include `BRXU`,
    `USEL`, `LDG.E.STRONG.GPU`, `LDG.E.128.STRONG.GPU`, `LDG.E.EF.128`,
    `LDL.LU.64`, and the wider `UISETP.*` family

---


### Atomic Global + Reduction (24 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `ATOM.E.ADD.64.STRONG.GPU` |  | atomicAdd(global) | default | |
| `ATOMG.E.ADD.STRONG.GPU` |  | atomicXxx_system() | barrier probes (-O3 --restrict) | Global atomic add with strong ordering (G = global scope) |
| `ATOM.E.ADD.F32.FTZ.RN.STRONG.GPU` |  | atomicAdd(global) | default | |
| `ATOM.E.ADD.STRONG.GPU` |  | atomicAdd(global) | default | |
| `ATOM.E.AND.STRONG.GPU` |  |  | default | |
| `ATOM.E.CAS.STRONG.GPU` |  |  | default | |
| `ATOM.E.EXCH.STRONG.GPU` |  |  | default | |
| `ATOM.E.MAX.S32.STRONG.GPU` |  |  | default | |
| `ATOM.E.MIN.S32.STRONG.GPU` |  |  | default | |
| `ATOM.E.OR.STRONG.GPU` |  |  | default | |
| `ATOM.E.XOR.STRONG.GPU` |  |  | default | |
| `ATOMG.E.AND.STRONG.GPU` |  | atomicXxx_system() | default | |
| `ATOM.E.CAS.64.STRONG.GPU` |  |  | -G debug + expanded probes | 64-bit global CAS with strong GPU ordering |
| `ATOMG.E.CAS.64.STRONG.GPU` |  | atomicXxx_system() | edge atomics probes | **64-bit global CAS with strong ordering.** For lock-free double-precision shared atomic accumulation. |
| `ATOMG.E.CAS.STRONG.SYS` |  | atomicXxx_system() | system-scope probes | **System-scope CAS** (cross-GPU/CPU visible atomics) |
| `ATOMG.E.EXCH.STRONG.SYS` |  | atomicXxx_system() | system-scope probes | **System-scope exchange** (CPU-visible atomic swap) |
| `ATOMG.E.CAS.STRONG.GPU` |  | atomicXxx_system() | default | |
| `ATOMG.E.EXCH.STRONG.GPU` |  | atomicXxx_system() | default | |
| `ATOMG.E.MAX.S32.STRONG.GPU` |  | atomicXxx_system() | default | |
| `ATOMG.E.MIN.S32.STRONG.GPU` |  | atomicXxx_system() | default | |
| `ATOMG.E.OR.STRONG.GPU` |  | atomicXxx_system() | default | |
| `ATOMG.E.XOR.STRONG.GPU` |  | atomicXxx_system() | default | |
| `RED.E.ADD.64.STRONG.GPU` |  | atomicAdd() [reduction, no return] | shared atomics probes | 64-bit global reduction add with strong memory ordering |
| `RED.E.ADD.F32.FTZ.RN.STRONG.GPU` |  | atomicAdd() [reduction, no return] | default | |
| `RED.E.ADD.STRONG.GPU` |  | atomicAdd() [reduction, no return] | default | |
| `RED.E.MAX.S32.STRONG.GPU` |  | atomicAdd() [reduction, no return] | shared atomics probes | Global reduction max (signed INT32, strong ordering) |
| `RED.E.ADD.F32.FTZ.RN.STRONG.SYS` |  | atomicAdd() [reduction, no return] | system-scope probes | **System-scope float reduction** (cross-GPU/CPU visible FP32 atomicAdd) |
| `RED.E.DEC.STRONG.GPU` |  | atomicAdd() [reduction, no return] | atomic sweep probes | **Global reduction wrapping decrement** with strong ordering |
| `RED.E.INC.STRONG.GPU` |  | atomicAdd() [reduction, no return] | atomic sweep probes | **Global reduction wrapping increment** with strong ordering |
| `RED.E.MIN.S32.STRONG.GPU` |  | atomicAdd() [reduction, no return] | shared atomics probes | Global reduction min (signed INT32, strong ordering) |
| `REDUX` |  | __reduce_and_sync() / redux.sync.and.b32 | atomic sweep probes | Bare raw SASS spelling for logical AND on this CUDA 13.1 toolchain. Treat semantic family as `REDUX.AND`. |
| `REDUX.MAX` |  | __reduce_max_sync() | atomic sweep probes | Warp-level max (bare, without .S32 suffix) |
| `REDUX.MIN` |  | __reduce_min_sync() | atomic sweep probes | Warp-level min (bare, without .S32 suffix) |
| `REDUX.MAX.S32` | ~60 cy | __reduce_max_sync() | default | |
| `REDUX.MIN.S32` | ~60 cy | __reduce_min_sync() | default | |
| `REDUX.OR` |  | __reduce_or_sync() | default | |
| `REDUX.SUM` | 60.01 cy (2.6x faster than SHFL tree) | __reduce_add_sync() | default | |
| `REDUX.SUM.S32` | 60.01 cy | __reduce_add_sync() | default | |
| `REDUX.XOR` |  | asm redux.sync.xor | shared atomics probes | **Warp-level XOR reduction.** Single-instruction bitwise parity across 32 lanes. Enables warp-wide parity/checksum in 1 cycle. Not ADD/MIN/MAX -- bitwise XOR fold. |
| `ATOM.E.ADD.64.STRONG.SYS` |  | atomicAdd_system(u64) | -G debug | 64-bit system-scope atomic add |
| `ATOM.E.ADD.F32.FTZ.RN.STRONG.SYS` |  | atomicAdd_system(float) | -G debug | System-scope FP32 atomic add with FTZ |
| `ATOM.E.ADD.STRONG.SYS` |  | atomicAdd_system(int) | -G debug | System-scope INT32 atomic add |
| `ATOM.E.AND.STRONG.SYS` |  | atomicAnd_system() | -G debug | System-scope bitwise AND |
| `ATOM.E.CAS.STRONG.SYS` |  | atomicCAS_system() | -G debug | System-scope compare-and-swap |
| `ATOM.E.DEC.STRONG.GPU` |  | atomicDec(global) | -G debug | Global wrapping decrement with GPU ordering |
| `ATOM.E.EXCH.STRONG.SYS` |  | atomicExch_system() | -G debug | System-scope exchange |
| `ATOM.E.INC.STRONG.GPU` |  | atomicInc(global) | -G debug | Global wrapping increment with GPU ordering |
| `ATOM.E.MAX.S32.STRONG.SYS` |  | atomicMax_system(int) | -G debug | System-scope signed max |
| `ATOM.E.MAX.STRONG.GPU` |  | atomicMax(unsigned,global) | -G debug | Unsigned max with GPU ordering |
| `ATOM.E.MAX.STRONG.SYS` |  | atomicMax_system(unsigned) | -G debug | System-scope unsigned max |
| `ATOM.E.MIN.S32.STRONG.SYS` |  | atomicMin_system(int) | -G debug | System-scope signed min |
| `ATOM.E.MIN.STRONG.GPU` |  | atomicMin(unsigned,global) | -G debug | Unsigned min with GPU ordering |
| `ATOM.E.MIN.STRONG.SYS` |  | atomicMin_system(unsigned) | -G debug | System-scope unsigned min |
| `ATOM.E.OR.STRONG.SYS` |  | atomicOr_system() | -G debug | System-scope bitwise OR |
| `ATOM.E.XOR.STRONG.SYS` |  | atomicXor_system() | -G debug | System-scope bitwise XOR |
| `RED.E.ADD.S32.STRONG.GPU` |  | atomicAdd(global) [red path] | -G debug | Signed INT32 reduction add |
| `RED.E.ADD.S32.STRONG.SYS` |  | atomicAdd_system() [red path] | -G debug | System-scope signed reduction add |
| `RED.E.AND.STRONG.GPU` |  | atomicAnd(global) [red path] | -G debug | Reduction AND (no return value) |
| `RED.E.MAX.S32.STRONG.SYS` |  | atomicMax_system() [red] | -G debug | System-scope reduction max |
| `RED.E.MIN.S32.STRONG.SYS` |  | atomicMin_system() [red] | -G debug | System-scope reduction min |
| `RED.E.MIN.STRONG.GPU` |  | atomicMin(unsigned) [red] | -G debug | Unsigned reduction min |
| `RED.E.OR.STRONG.GPU` |  | atomicOr(global) [red path] | -G debug | Reduction OR (no return value) |
| `RED.E.XOR.STRONG.GPU` |  | atomicXor(global) [red path] | -G debug | Reduction XOR (no return value) |

### Atomic Shared (7 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `ATOMS.CAS` |  | atomicCAS(shared) | shared atomics probes | **Shared memory compare-and-swap (32-bit).** Used for lock-free CAS loops on smem. Generates ATOMS.CAS instruction (distinct from ATOM.E.CAS for global). |
| `ATOMS.CAS.64` |  | atomicCAS(shared) | shared atomics probes | **Shared memory 64-bit CAS.** For 64-bit lock-free updates in smem. |
| `ATOMS.CAST.SPIN` |  | atomicCAS(shared) | -G | Debug spin-lock CAS loop |
| `ATOMS.CAST.SPIN.64` |  | atomicCAS(shared) | -G | Debug 64-bit spin-lock |
| `ATOMS.DEC` |  | atomicDec(shared) | atomic sweep probes | **Shared memory wrapping decrement.** atomicDec: (old==0 or old>limit) ? limit : old-1. |
| `ATOMS.EXCH` |  | atomicExch(shared) | shared atomics probes | **Shared memory exchange.** Atomically replaces smem value, returns old value. |
| `ATOMS.INC` |  | atomicInc(shared) | atomic sweep probes | **Shared memory wrapping increment.** atomicInc: (old>=limit) ? 0 : old+1. |
| `ATOMS.MIN.S32` | 4.37 cy (single thread) | atomicMin(shared) | atomic sweep probes | **Shared memory signed minimum.** Nearly same latency as ATOMS.ADD (4.27 cy). |
| `ATOMS.POPC.INC.32` |  |  | default | |

### Barrier/Sync (8 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `B2R.RESULT` |  | __syncthreads_count() [result] | barrier probes | **Barrier-to-register.** Reads BAR.RED reduction result into a GPR. |
| `BAR.ARV` |  | bar.arrive (PTX) | barrier probes | **Split-phase arrive-without-wait.** |
| `BAR.RED.AND.DEFER_BLOCKING` |  | __syncthreads_and() | barrier probes | **Barrier + AND predicate reduction** across all threads in block. |
| `BAR.RED.OR.DEFER_BLOCKING` |  | __syncthreads_or() | barrier probes | **Barrier + OR predicate reduction.** |
| `BAR.RED.POPC.DEFER_BLOCKING` |  | __syncthreads_count() | barrier probes | **Barrier + population count reduction** (count threads with pred=true). |
| `BAR.SYNC` | 35.01 cy | __syncthreads() | default | Block-level barrier. |
| `BAR.SYNC.DEFER_BLOCKING` |  | __syncthreads() | default | |
| `DEPBAR.LE` |  | dependency barrier | default | |

Recent `mbarrier` lowering note:
- `cuda_awbarrier_primitives.h` probes on Ada SM89 lower through
  `BAR.SYNC.DEFER_BLOCKING`, `MEMBAR.ALL.CTA`, `ATOMS.ARRIVE.64`, and
  `ATOMS.POPC.INC.32` rather than exposing a distinct raw `MBAR*` mnemonic in
  `cuobjdump -sass` on this CUDA 13.1 toolchain.
- The safe init/arrive/wait and arrive-drop flows run correctly in the dedicated
  runner. `__mbarrier_try_wait()` compiles, but the emitted path behaves like a
  trap-style negative control during live launch and is tracked as disassembly-only.

### Bit Manipulation (25 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `BMSK` |  | mask generation | -G | |
| `BREV` | 17.51 cy (SFU) | __brev() | default | |
| `FLO.U32` |  | __clz()/__ffs() | default | |
| `FLO.U32.SH` |  | __clz()/__ffs() | default | |
| `LOP3.LUT` | 4.53 cy | __funnelshift / bitwise | default | |
| `PLOP3.LUT` |  |  | default | |
| `POPC` | ~7-8 cy (multi-cycle INT, corrected from 23.52) | __popc() | default | |
| `PRMT` | 4.52 cy | __byte_perm() | default | |
| `PRMT.B4E` |  | __byte_perm() mode B4E | bitmanip probes | **Byte-4 extract** permute mode. |
| `PRMT.ECL` |  | __byte_perm() mode ECL | bitmanip probes | **Edge clamp left** permute mode. |
| `PRMT.ECR` |  | __byte_perm() mode ECR | bitmanip probes | **Edge clamp right** permute mode. |
| `PRMT.F4E` |  | __byte_perm() mode F4E | bitmanip probes | **Four-byte extract** (funnel shift by bytes). |
| `PRMT.RC16` |  | __byte_perm() mode RC16 | bitmanip probes | **Replicate 16-bit** to both halves. |
| `PRMT.RC8` |  | __byte_perm() mode RC8 | bitmanip probes | **Replicate 8-bit** to all 4 bytes. |
| `SGXT` |  | sign extend | default | |
| `SGXT.U32` |  | sign extend | default | |
| `SHF.L.U32` | 4.52 cy | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.L.U32.HI` | ~4.52 (similar) | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.L.U64.HI` |  | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.L.W.U32` | ~4.52 cy | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.L.W.U32.HI` | ~~4.52 (similar) | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.R.S32.HI` | 4.52 cy | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.R.S64` | ~4.52 cy | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.R.U32` | ~4.52 (similar) | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.R.U32.HI` | 4.52 cy | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.R.U64` |  | __funnelshift_l()/__funnelshift_r() | default | |
| `SHF.R.W.U32` |  | __funnelshift_l()/__funnelshift_r() | default | |
| `SHFL.BFLY` | 24.96 cy | __shfl_xor_sync() | default | |
| `SHFL.DOWN` | ~24.96 cy | __shfl_down_sync() | default | |
| `SHFL.IDX` | ~24.96 cy | __shfl_sync() | default | |
| `SHFL.UP` | ~24.96 cy | __shfl_up_sync() | default | |

### Cache Control (5 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `CCTL.E.PF1` |  | cache control | default | |
| `CCTL.E.PF2` |  | cache control | default | |
| `CCTL.IVALL` |  | cache control | default | |
| `QSPC.E.G` |  | query address space | -G | |
| `QSPC.E.S` |  | query address space | -G | |

### Control Flow (14 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `BMOV.32` |  |  | -G | |
| `BMOV.32.CLEAR` |  |  | -G | |
| `BRA` |  | if/else/loop branch | default | |
| `BRA.CONV` |  | if/else/loop branch | barrier probes | **Convergent branch** (compiler marks branches where all threads take same path) |
| `BRA.DIV` |  | if/else/loop branch | barrier probes | **Divergent branch** (compiler marks branches where threads may diverge) |
| `BRA.CONV` |  | if/else/loop branch | default | |
| `BRX` |  |  | default | |
| `BSSY` |  | compiler: divergence entry | default | |
| `BSYNC` |  | compiler: reconvergence | default | |
| `CALL.ABS.NOINC` |  | function call | -G | |
| `CALL.REL.NOINC` |  | function call | default | |
| `EXIT` | ~0 cy (thread termination) | return/kernel end | default | |
| `NOP` | 0 cy (pipeline filler) | pipeline filler | default | |
| `RET.ABS.NODEC` |  | function return | -G | |
| `RET.REL.NODEC` |  | function return | default | |
| `YIELD` | 2676.61 cy (=NANOSLEEP) | __nanosleep(0) | default | |

### Conversion (38 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `F2F.BF16.F32` |  |  | default | |
| `F2F.F16.F32` |  |  | default | |
| `F2F.F16.F32.RM` |  |  | default | |
| `F2F.F16.F32.RP` |  |  | default | |
| `F2F.F16.F32.RZ` |  |  | default | |
| `F2F.F32.F64` |  |  | default | |
| `F2F.F64.F32` |  |  | default | |
| `F2FP.BF16.F32.PACK_AB` | ~8.54 cy (BF16 round-trip) | format conversion | default | Direct local `sm_89` form. Alias follow-up still does not collapse this to shorter library-mined `F2FP.BF16.PACK_AB` locally. |
| `F2FP.RELU.BF16.F32.PACK_AB` |  | asm `cvt.rn.relu.bf16x2.f32` | direct confirm tranche | Packed BF16 convert with ReLU clamp from forced inline PTX. |
| `F2FP.F16.E4M3.UNPACK_B` | ~6 cy (FP8 decode) | format conversion | default | |
| `F2FP.F16.E5M2.UNPACK_B` | ~6 cy (FP8 E5M2 decode) | format conversion | default | |
| `F2FP.F16.F32.PACK_AB` | ~10.54 cy (FP16 round-trip) | format conversion | default | Direct local `sm_89` form. Alias follow-up still does not collapse this to shorter library-mined `F2FP.PACK_AB` locally. |
| `F2FP.F16.F32.PACK_AB.RZ` | ~~10.54 (similar) | format conversion | default | |
| `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | ~18.54 cy (FP8 round-trip) | format conversion | default | |
| `F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C` | ~18.53 cy (FP8 E5M2 round-trip) | format conversion | default | |
| `F2I.CEIL.NTZ` |  | float-to-int cast | default | |
| `F2I.F64` |  | (int)double | default | |
| `F2I.F64.TRUNC` |  | (int)double [truncate] | -G debug | **FP64->INT32 truncation** |
| `F2I.FLOOR.NTZ` |  | (int)floorf() | default | |
| `F2I.S64.F64.TRUNC` |  | (long long)double | -G debug | **FP64->INT64 truncation** |
| `F2I.U32.F64.TRUNC` |  | (unsigned)fabs(double) | -G debug | **FP64->UINT32 truncation** |
| `F2I.FTZ.CEIL.NTZ` |  | float-to-int cast | default | |
| `F2I.FTZ.FLOOR.NTZ` |  | float-to-int cast | default | |
| `F2I.FTZ.NTZ` |  | float-to-int cast | default | |
| `F2I.FTZ.TRUNC.NTZ` |  | float-to-int cast | default | |
| `F2I.FTZ.U32.NTZ` |  | float-to-int cast | default | |
| `F2I.FTZ.U32.TRUNC.NTZ` |  | float-to-int cast | default | |
| `F2I.NTZ` |  | float-to-int cast | default | |
| `F2I.TRUNC.NTZ` | ~12.03 cy (F2I+I2F round-trip) | float-to-int cast | default | |
| `F2I.U32.NTZ` |  | float-to-int cast | default | |
| `F2I.U32.TRUNC.NTZ` |  | float-to-int cast | default | |
| `F2I.U64.TRUNC` |  | float-to-int cast | INT64 tiling probes | **Float to unsigned 64-bit with truncation.** New 64-bit conversion variant. |
| `I2F.F64` |  | int-to-float cast | default | |
| `I2F.F64.S64` |  | int-to-float cast | default | |
| `I2F.RM` |  | int-to-float cast | default | |
| `I2F.RP` |  | int-to-float cast | default | |
| `I2F.S16` | ~44.51 cy (INT16 I2F+F2I chain) | int-to-float cast | default | |
| `I2F.S8` | ~6 cy (direct byte-to-float) | int-to-float cast | --restrict | |
| `I2F.U16` |  | int-to-float cast | default | |
| `I2F.U32.RP` |  | int-to-float cast | default | |
| `I2F.U64.RP` |  | int-to-float cast | INT64 tiling probes | **Unsigned 64-bit to float with round-toward-positive.** New FP64 conversion variant. |
| `I2FP.F32.S32` | ~6 cy | int-to-float (packed) | default | |
| `I2FP.F32.S32.RZ` | ~~6 (similar) | int-to-float (packed) | default | |
| `I2FP.F32.U32` |  | int-to-float (packed) | default | |

### Data Movement (2 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `MOV` | ~2 cy (register move) | register assignment | default | |
| `SEL` | ~4.53 cy | ternary select | default | |

### FP16/BF16 Packed + Tensor Core Float (14 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `HADD2` | 4.54 cy | __hadd2() | default | |
| `HADD2.F32` | ~4.54 cy | __hadd2() | default | |
| `HFMA2` | 4.54 cy | __hfma2() | default | |
| `HFMA2.RELU` |  | asm `fma.rn.relu.f16x2` | direct confirm tranche | Direct local `sm_89` confirmation from forced inline PTX. |
| `HFMA2.BF16_V2` | 4.01 cy (FASTEST FMA on Ada) | __hfma2(bfloat162) | default | |
| `HMMA.16816.F16` | 42.14 cy/WMMA (fastest float TC) | wmma::mma_sync() [tensor core] | default | |
| `HMMA.16816.F32` | 66.28 cy/WMMA (256 FMA) | wmma::mma_sync() [tensor core] | default | |
| `HMMA.16816.F32.BF16` | 66.33 cy/WMMA (=FP16->FP32) | wmma::mma_sync() [tensor core] | default | |
| `HMMA.1684.F32.TF32` | 66.66 cy/2xHMMA (TF32 via 2 instructions) | wmma::mma_sync() [tensor core] | default | |
| `HMMA.1688.F32.TF32` |  | asm `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | direct confirm tranche | Direct local `sm_89` confirmation of the single-instruction TF32 m16n8k8 form. |
| `HMNMX2` |  | __hmax2()/__hmin2() | default | |
| `HMNMX2.NAN` |  | asm `min.NaN/max.NaN.f16x2` | direct confirm tranche | Direct local `sm_89` confirmation of NaN-aware packed half2 min/max lowering. |
| `HSETP2.GTU.AND` |  | __hgt2()/__hle2() | edge atomics probes | **Half2 packed comparison set-predicate** (greater-than-unordered, AND combiner). FP16 packed comparison -- first observation. |
| `HSETP2.LE.AND` |  | __hgt2()/__hle2() | -G debug + expanded probes | Half2 packed less-or-equal comparison. Second FP16 packed comparison variant. |
| `HMUL2` | ~4.54 cy (only with -fmad=false) | __hmul2() (-fmad=false) | -fmad=false | |

### FP32 Arithmetic (42 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `FADD` | 4.52 cy | a + b | default | |
| `FADD.FTZ` | ~4.52 cy | --use_fast_math: a + b | default | |
| `FADD.FTZ.RZ` | ~4.52 cy | --use_fast_math: a + b | default | |
| `FADD.RZ` | ~4.52 cy | a + b | default | |
| `FADD.SAT` | 4.52 cy (SAT=FREE) | __saturatef(a+b) | default | |
| `FCHK` |  | __isnanf()/__isinff() | default | |
| `FFMA` | 4.53 cy | fmaf(a,b,c) | default | |
| `FFMA.FTZ` | 4.51 cy (FTZ=FREE) | --use_fast_math: fmaf() | default | |
| `FFMA.RM` | ~4.53 cy | fmaf(a,b,c) | default | |
| `FFMA.RP` | ~4.53 cy | fmaf(a,b,c) | default | |
| `FFMA.RZ` | ~4.53 cy | fmaf(a,b,c) | default | |
| `FFMA.SAT` | ~4.53 (similar) | saturated FMA | default | |
| `FMNMX` |  | fminf()/fmaxf() | default | |
| `FMNMX.FTZ` |  | fminf()/fmaxf() | default | |
| `FMUL` | 4.52 cy | a * b | default | |
| `FMUL.D8` | ~4.52 (similar) | a * b | default | |
| `FMUL.FTZ` | ~4.52 (similar) | a * b | default | |
| `FMUL.FTZ.D8` | ~4.52 cy | a * b | default | |
| `FMUL.RZ` | ~4.52 (similar) | a * b | default | |
| `FMUL.SAT` | ~4.52 (similar) | a * b | default | |
| `FSEL` | 8.52 cy | ternary ? : | default | |
| `FSET.BF.GT.AND` |  |  | default | |
| `FSET.BF.GT.FTZ.AND` |  |  | default | |
| `FSETP.EQ.AND` |  | a > b (float comparison) | default | |
| `FSETP.EQ.FTZ.AND` |  | a > b (float comparison) | default | |
| `FSETP.EQ.OR` |  | a > b (float comparison) | default | |
| `FSETP.GE.AND` | ~8.52 cy | a > b (float comparison) | default | |
| `FSETP.GE.FTZ.AND` |  | a > b (float comparison) | default | |
| `FSETP.GEU.AND` |  | a > b (float comparison) | default | |
| `FSETP.GEU.FTZ.AND` |  | a > b (float comparison) | default | |
| `FSETP.GEU.OR` |  | a > b (float comparison) | default | |
| `FSETP.GT.AND` | ~8.52 cy | a > b (float comparison) | default | |
| `FSETP.GT.FTZ.AND` |  | a > b (float comparison) | default | |
| `FSETP.GTU.AND` |  | a > b (float comparison) | default | |
| `FSETP.GTU.FTZ.AND` | ~8.52 cy | a > b (float comparison) | default | |
| `FSETP.GTU.OR` |  | a > b (float comparison) | default | |
| `FSETP.LE.AND` |  | a > b (float comparison) | default | |
| `FSETP.LE.FTZ.AND` |  | a > b (float comparison) | default | |
| `FSETP.LT.AND` |  | a > b (float comparison) | default | |
| `FSETP.LT.FTZ.AND` |  | a > b (float comparison) | default | |
| `FSETP.NEU.AND` | ~8.52 cy | a > b (float comparison) | default | |
| `FSETP.NEU.FTZ.AND` |  | a > b (float comparison) | default | |

### FP64 Arithmetic (19 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `DADD` | 48.47 cy | double a + b | default | |
| `DFMA` | 54.48 cy | fma(double) | default | |
| `DFMA.RM` | ~54.48 (similar) | fma(double) | default | |
| `DFMA.RP` | ~54.48 (similar) | fma(double) | default | |
| `DMUL` | 48.47 cy | double a * b | default | |
| `DMUL.RP` | ~48.47 (similar) | double a * b | default | |
| `DSETP.EQ.AND` |  |  | default | |
| `DSETP.EQU.AND` |  |  | default | |
| `DSETP.GE.AND` |  |  | default | |
| `DSETP.GEU.AND` |  |  | default | |
| `DSETP.GT.AND` | ~38 cy |  | default | |
| `DSETP.GTU.AND` | ~38 cy |  | default | |
| `DSETP.LE.AND` |  |  | default | |
| `DSETP.LT.AND` |  |  | default | |
| `DSETP.MAX.AND` |  |  | default | |
| `DSETP.MIN.AND` |  |  | default | |
| `DSETP.NAN.AND` |  |  | default | |
| `DSETP.NE.AND` |  |  | default | |
| `DSETP.NEU.AND` |  |  | default | |

### Fence (5 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `MEMBAR.ALL.GPU` | 205.25 cy (__threadfence) | __threadfence() | default | |
| `MEMBAR.SC.CTA` |  | __threadfence_block() | default | |
| `MEMBAR.SC.GPU` |  |  | default | |
| `MEMBAR.SC.SYS` | 2583.37 cy (system-scope) | __threadfence_system() | default | |
| `MEMBAR.SC.VC` |  | debug-lane memory ordering | -G dedicated follow-up | Reproduced by `probe_membar_sc_vc_debug.cu` and `probe_r2ur_debug_path.cu` in `-O0 -G`; absent from the optimized lane for the same kernels. |

### Integer Arithmetic (24 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `IABS` | 0.26 cy/pair (sub-cycle modifier) | abs(int) | default | |
| `IADD3` | 2.52 cy | a + b (int32) | default | |
| `IADD3.X` | ~2.59 cy (carry propagation) | a + b (int32) | default | |
| `IDP.4A.S8.S8` | 4.53 cy (4x effective INT8) | __dp4a() / dp4a.s32.s32 | default + signedness probe | Signed x signed packed INT8 dot product. |
| `IDP.4A.S8.U8` | 4.53 cy (same pipe as S8.S8) | dp4a.s32.u32 | signedness probe | Signed x unsigned packed INT8 dot product. |
| `IDP.4A.U8.S8` | 4.53 cy (same pipe as S8.S8) | dp4a.u32.s32 | signedness probe | Unsigned x signed packed INT8 dot product. |
| `IDP.4A.U8.U8` | 4.53 cy (same pipe as S8.S8) | dp4a.u32.u32 | signedness probe | Unsigned x unsigned packed INT8 dot product. |
| `IMAD` | 4.53 cy | a * b + c (int32) | default | |
| `IMAD.HI` | ~4.53 cy | a * b + c (int32) | default | |
| `IMAD.HI.U32` | ~4.53 cy | a * b + c (int32) | default | |
| `IMAD.IADD` | ~4.53 cy | a * b + c (int32) | default | |
| `IMAD.MOV` | ~4.53 (similar) | a * b + c (int32) | default | |
| `IMAD.MOV.U32` | ~4.53 cy | a * b + c (int32) | default | |
| `IMAD.SHL` | ~4.53 (similar) | a * b + c (int32) | default | |
| `IMAD.SHL.U32` | ~4.53 cy | a * b + c (int32) | default | |
| `IMAD.U32` | ~4.53 (similar) | a * b + c (int32) | default | |
| `IMAD.WIDE` | 2.59 cy (INT64 ADD via carry chain) | a * b + c (int32) | default | |
| `IMAD.WIDE.U32` | ~4.53 cy | a * b + c (int32) | default | |
| `IMAD.WIDE.U32.X` | ~4.53 (similar) | a * b + c (int32) | default | |
| `IMAD.X` | ~4.53 cy | a * b + c (int32) | default | |
| `IMNMX` | ~4.53 cy | min(int)/max(int) | default | |
| `IMNMX.U32` | ~4.53 cy | min(int)/max(int) | default | |
| `LEA` |  |  | default | |
| `LEA.HI` |  |  | default | |
| `LEA.HI.SX32` |  |  | default | |
| `LEA.HI.X` |  |  | default | |
| `LEA.HI.X.SX32` |  |  | default | |

### Integer Comparison (29 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `ISETP.EQ.AND` | ~4.53 cy | a > b (int comparison) | default | |
| `ISETP.EQ.AND.EX` | ~~4.53 (similar) | a > b (int comparison) | default | |
| `ISETP.EQ.OR` |  | a > b (int comparison) | default | |
| `ISETP.EQ.U32.AND` |  | a > b (int comparison) | default | |
| `ISETP.EQ.U32.AND.EX` |  | a > b (int comparison) | default | |
| `ISETP.GE.AND` | ~4.53 cy | a > b (int comparison) | default | |
| `ISETP.GE.AND.EX` | ~~4.53 (similar) | a > b (int comparison) | default | |
| `ISETP.GE.OR` |  | a > b (int comparison) | default | |
| `ISETP.GE.U32.AND` |  | a > b (int comparison) | default | |
| `ISETP.GE.U32.OR` |  | a > b (int comparison) | tiling probes (-O3 --restrict) | Unsigned GE with OR predicate combiner |
| `ISETP.GE.U32.AND.EX` |  | a > b (int comparison) | default | |
| `ISETP.GT.AND` | ~4.53 cy | a > b (int comparison) | default | |
| `ISETP.GT.AND.EX` | ~~4.53 (similar) | a > b (int comparison) | default | |
| `ISETP.GT.U32.AND` |  | a > b (int comparison) | default | |
| `ISETP.GT.U32.AND.EX` |  | a > b (int comparison) | default | |
| `ISETP.GT.U32.OR` |  | a > b (int comparison) | default | |
| `ISETP.LE.AND` |  | a > b (int comparison) | default | |
| `ISETP.LE.U32.AND` |  | a > b (int comparison) | default | |
| `ISETP.LE.U32.AND.EX` |  | a > b (int comparison) | default | |
| `ISETP.LT.AND` | ~4.53 cy | a > b (int comparison) | default | |
| `ISETP.LT.AND.EX` | ~~4.53 (similar) | a > b (int comparison) | default | |
| `ISETP.LT.OR` |  | a > b (int comparison) | default | |
| `ISETP.LT.U32.AND` |  | a > b (int comparison) | default | |
| `ISETP.LT.U32.AND.EX` |  | a > b (int comparison) | default | |
| `ISETP.LT.U32.OR.EX` |  | a > b (int comparison) | default | |
| `ISETP.NE.AND` | ~4.53 cy | a > b (int comparison) | default | |
| `ISETP.NE.AND.EX` | ~~4.53 (similar) | a > b (int comparison) | default | |
| `ISETP.NE.OR` |  | a > b (int comparison) | default | |
| `ISETP.NE.U32.AND` |  | a > b (int comparison) | default | |

### Memory Constant (8 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `LDC` | 70.57 cy (constant cache chain) | constant memory (__constant__) | default | |
| `LDC.64` |  | constant memory (__constant__) | default | |
| `LDC.S8` |  | constant memory (__constant__) | direct prefix follow-up | Direct local signed 8-bit constant-memory load from `probe_ldc_subword.cu`. |
| `LDC.U16` |  | constant memory (__constant__) | -G debug + expanded probes | Unsigned 16-bit constant memory load (sub-word constant cache) |
| `LDC.U8` |  | constant memory (__constant__) | direct prefix follow-up | Direct local unsigned 8-bit constant-memory load from `probe_ldc_subword.cu`. |
| `ULDC` |  |  | default | |
| `ULDC.64` |  |  | default | |
| `ULDC.S8` |  | `probe_uniform_exotic.cu` | direct confirm tranche | Uniform signed 8-bit constant-memory load from a warp-uniform constant-table index. |
| `ULDC.U8` |  | `probe_uniform_exotic.cu` | direct confirm tranche | Uniform unsigned 8-bit constant-memory load from a warp-uniform constant-table index. |

### Memory Global (51 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `LD.E` |  |  | default | |
| `LD.E.128` |  |  | default | |
| `LD.E.64` |  |  | default | |
| `LD.E.64.STRONG.SYS` |  |  | default | |
| `LD.E.STRONG.GPU` |  |  | default | |
| `LD.E.STRONG.SYS` |  |  | default | |
| `LD.E.U16` |  |  | default | |
| `LD.E.U16.STRONG.SYS` |  |  | default | |
| `LD.E.U8` |  |  | default | |
| `LDG.E` | 92-123 cy (L2 hit, varies by SKU) | global load (default) | default | |
| `LDG.E.128` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.128.CONSTANT` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.64` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.64.CONSTANT` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.64.STRONG.SYS` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.CONSTANT` | ~33 cy (L1 read-only cache) | __ldg() | default | |
| `LDG.E.EF` |  | ld.global.cs (streaming) | cache probes | **Global load evict-first.** Streaming read hint, matching STG.E.EF for stores. |
| `LDG.E.LU` |  | ld.global.lu (last-use) | cache probes | **Global load last-use.** Evict from cache after this read. |
| `LDG.E.S16` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.S8` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.STRONG.SYS` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.U16` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.U16.CONSTANT` | ~92-123 (similar) | global load (default) | --restrict | |
| `LDG.E.U16.STRONG.SYS` | ~92-123 (similar) | global load (default) | default | |
| `LDG.E.U8` | ~44.99 cy (byte pointer chase) | global load (default) | default | |
| `LDG.E.U8.CONSTANT` | ~92-123 (similar) | global load (default) | --restrict | |
| `LDGDEPBAR` | ~0 cy (barrier token, not execution) | __pipeline_commit() | default | |
| `LDGSTS.E` | 363.28 cy/iter (async copy with sync) | cp.async / __pipeline_memcpy_async() | default | |
| `LDGSTS.E.64.ZFILL` | ~363.28 (similar) | cp.async / __pipeline_memcpy_async() | -G | |
| `LDGSTS.E.BYPASS.128` | ~363.28 (similar) | cp.async ignore-src lowering | explicit cp.async probe | Predicated non-`ZFILL` ignore-src lowering. |
| `LDGSTS.E.BYPASS.LTC128B.128` |  | asm `cp.async.cg.shared.global.L2::128B ..., 16` | direct confirm tranche | Direct local `sm_89` confirmation of the `.cg` L2 prefetch-hint path. |
| `LDGSTS.E.BYPASS.LTC128B.128.ZFILL` |  | asm `cp.async.cg.shared.global.L2::128B ..., 16, 8` | direct confirm tranche | `.cg` L2 prefetch-hint path with zero fill. |
| `LDGSTS.E.BYPASS.128.ZFILL` | ~363.28 (similar) | cp.async.cg.shared.global ..., 16, 8 | explicit cp.async probe | 16-byte bypass path with zero fill. |
| `LDGSTS.E.LTC128B.128` |  | asm `cp.async.ca.shared.global.L2::128B ..., 16` | direct confirm tranche | Direct local `sm_89` confirmation of the `.ca` L2 prefetch-hint path. |
| `LDGSTS.E.LTC128B.128.ZFILL` |  | asm `cp.async.ca.shared.global.L2::128B ..., 16, 8` | direct confirm tranche | `.ca` L2 prefetch-hint path with zero fill. |
| `LDGSTS.E.ZFILL` | ~363.28 (similar) | cp.async.ca.shared.global ..., 4, 2 | explicit cp.async probe | 4-byte zero-fill path. |
| `ST.E` |  |  | default | |
| `ST.E.128` |  |  | default | |
| `ST.E.64` |  |  | default | |
| `ST.E.64.STRONG.SYS` |  |  | default | |
| `ST.E.STRONG.SYS` |  |  | default | |
| `ST.E.U16` |  |  | default | |
| `ST.E.U16.STRONG.SYS` |  |  | default | |
| `ST.E.U8` |  |  | default | |
| `STG.E` | ~~4 (similar) | global store | default | |
| `STG.E.128` |  | global store | default | |
| `STG.E.64` |  | global store | default | |
| `STG.E.64.STRONG.SYS` |  | global store | default | |
| `STG.E.EF` | ~4 cy issue (evict-first, async) | __stcs() (evict-first) | default | |
| `STG.E.STRONG.SM` |  | global store (SM scope) | cache probes | **SM-scope strong store.** Weaker than GPU-scope, only visible within the SM. |
| `STG.E.STRONG.SYS` |  | global store | default | |
| `STG.E.U16` |  | global store | default | |
| `STG.E.U16.STRONG.SYS` |  | global store | default | |
| `STG.E.U8` |  | global store | default | |

### Memory Local (Spill) (5 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `LDL` |  | local memory load (spill) | -G | |
| `LDL.64` |  | local memory load (spill) | default | |
| `LDL.LU` |  | local memory load (spill) | --restrict | |
| `STL` |  | local memory store (spill) | --restrict / -G | |
| `STL.64` |  | local memory store (spill) | default | |

### Memory Shared (11 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `LDS` | 28.03 cy (shared memory) | shared memory load | default | |
| `LDS.128` |  | shared memory load | default | |
| `LDS.64` |  | shared memory load | default | |
| `LDS.S8` |  | shared memory load | INT8 tiling probes | **Signed 8-bit shared memory load.** Sign-extends byte to 32-bit register. First sub-byte signed smem load observed. |
| `LDS.S16` |  | shared memory load | INT16 tiling probes | **Signed 16-bit shared memory load.** Sign-extends short to 32-bit register. |
| `LDS.U8` |  | shared memory load | bitops tiling probes | Unsigned 8-bit shared memory load. |
| `LDS.U16` |  | shared memory load | edge atomics probes | Unsigned 16-bit shared memory load. |
| `STS` |  | shared memory store | default | |
| `STS.U8` |  | shared memory byte store | data movement probes | **Unsigned 8-bit shared memory store.** Sub-byte smem write. |
| `STS.64` |  | shared memory store | -O3 no-restrict | 64-bit shared memory store (without --restrict) |
| `STS.U16` |  | shared memory store | default | |

### SIMD Video (2 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `VABSDIFF4.U8` |  | __vabsdiffu4() | SIMD video probes | **Packed INT8x4 absolute difference.** Baseline packed-video opcode on Ada. Most other SIMD video intrinsics still decompose to standard integer SASS. |
| `VABSDIFF4.U8.ACC` |  | inline PTX `vabsdiff4...add`, __vsadu4() lowering pattern | forced inline PTX video probe | **Packed INT8x4 absolute difference with accumulate.** Observed when `vabsdiff4` is forced with an explicit add/accumulate operand. |

SIMD-video lowering note from the recursive Ada corpus:
- `__vsadu4()` still reaches native video hardware, but only through
  `VABSDIFF4.U8` followed by integer horizontal-byte accumulation.
- The trap-resistant inline-PTX probe `probe_video_isa_inline_ptx.cu` confirms:
  `vadd2/vsub2 -> IADD3 + LOP3.LUT`, `vadd4/vsub4 -> LOP3.LUT + IMAD.IADD`,
  `vavrg2 -> LOP3.LUT + SHF + IMAD.IADD`, `vavrg4 -> PRMT + SHF + IMAD.IADD`,
  `vmin2/vmax2 -> IMNMX.U32`, `vmin2.s32/vmax2.s32 -> PRMT + IMNMX`,
  `vmin4/vmax4 -> PRMT + IMNMX`, `vset2/vset4 -> PRMT + ISETP + LOP3.LUT`,
  `vset2.s32/vset4.s32 -> PRMT + ISETP + LOP3.LUT`, and
  `vabsdiff4...add -> VABSDIFF4.U8.ACC`.
- The follow-on probes `probe_video_scalar_isa_inline_ptx.cu` and
  `probe_video_variant_isa_inline_ptx.cu` confirm that scalar PTX video
  instructions also synthesize on Ada:
  `vadd/vsub -> IMAD.IADD`, `vabsdiff -> IADD3 + IADD3.X + ISETP`,
  `vmin/vmax -> PRMT + ISETP`, `vshl/vshr -> SHF + predicate/setp plumbing`,
  `vmad -> PRMT + shifts/adds/multiply helpers`, `vset -> PRMT + ISETP`,
  and `vabsdiff2 -> IABS + shift/add glue`.
- `__vadd2`, `__vsub2`, `__vadd4`, `__vsub4`, `__vmax*`, `__vmin*`,
  `__viaddmax*`, `__viaddmin*`, `__vset2`, and `__vset4` do **not** emit extra
  `V*` raw SASS on this toolkit. They lower to combinations of `IADD3`,
  `LOP3.LUT`, `PRMT`, `IMNMX`, and `ISETP`.
- Recompiling the entire `simd_video` slice under both `-O0 -G` and
  `-O3 -Xptxas -O3` does not add any new raw packed-video `V*` spellings.
  `-G` changes the surrounding support code and surfaces debug/control mnemonics
  such as `BPT.TRAP`, `BSSY`, `BSYNC`, `LDC`, `LDL`, and `STL`, but not new
  packed-video ALU opcodes.
- Recompiling the scalar/variant inline-PTX tranches under both `-O0 -G` and
  `-O3 -Xptxas -O3` also does not add any new raw `V*` spellings.
- The selector-heavy tranche `probe_video_selector_isa_inline_ptx.cu`
  confirms the same result for merge selectors and packed subfield forms:
  `.b0`, `.h1`, `.h10`, `.b3210`, `vset*.add`, and `vabsdiff2` still lower to
  `PRMT`/`ISETP`/`SHF`/`IMAD` glue rather than exposing new raw packed-video
  `V*` opcodes.
- `PLOP3.LUT` remains a mainline mnemonic, not a debug appendix artifact.
  The follow-up bundle `predicate_logic_followup_20260320_091700` shows that a
  minimal direct `-O3 -Xptxas -O3` build of `probe_predicate_pressure.cu` does
  not retain `PLOP3.LUT`, but the full compile-profile lane used by the main
  corpus (`--use_fast_math --restrict --extra-device-vectorization` with the
  resolver-selected `nvcc` language mode; `-std=c++20` locally on CUDA 13.1)
  does reproduce `PLOP3.LUT` and `P2R` in optimized code.
- Across the completed recursive run, the only observed raw `V*` mnemonics are
  `VABSDIFF4.U8`, `VABSDIFF4.U8.ACC`, `VOTE.ALL`, `VOTE.ANY`, and `VOTEU.ANY`.

### Other (6 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `BPT.TRAP` |  | debug breakpoint | default | |
| `ERRBAR` |  | error barrier | default | |
| `FRND` |  | roundf() | default | |
| `FRND.FLOOR` |  | floorf() | default | |
| `FRND.TRUNC` |  | truncf() | default | |
| `NANOSLEEP` | 2685 cy | __nanosleep() | default | Timer-independent warp reschedule overhead |

### SFU (MUFU) (9 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `MUFU.COS` | ~23.50 cy | __cosf() | default | |
| `MUFU.EX2` | 17.55 cy | __expf() / __exp2f() | default | |
| `MUFU.LG2` | 39.53 cy | __logf() / __log2f() | default | |
| `MUFU.RCP` | 41.53 cy | __frcp_rn() / 1.0f/x | default | |
| `MUFU.RCP64H` | 17.54 cy | __frcp_rn() / 1.0f/x | default | |
| `MUFU.RSQ` | 39.53 cy | rsqrtf() | default | |
| `MUFU.RSQ64H` | ~17.54 cy (FP64 rsqrt approx) | 1.0/sqrt(double) [approx] | default | |
| `MUFU.SIN` | 23.50 cy | __sinf() | default | |
| `MUFU.SQRT` | ~17.5 cy (only with --use_fast_math) | sqrtf() [--use_fast_math] | --use_fast_math | |

### Special Register (7 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `CS2R` |  |  | default | |
| `CS2R.32` |  |  | default | |
| `P2R` |  | predicate spill to register | --use_fast_math | |
| `R2P` |  | predicate reload | default | |
| `R2UR` |  | debug-lane register-to-uniform transfer | -G dedicated follow-up | Reproduced by `probe_r2ur_debug_path.cu` and the debug lane of `probe_membar_sc_vc_debug.cu`; not observed in the optimized lane for the same kernels. |
| `S2R` |  | threadIdx/blockIdx/clock | default | |
| `S2UR` |  |  | default | |

### Tensor Core Integer + Data (8 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `IMMA.16816.S8.S8` | 34.06 cy/WMMA (INT8, 2x faster than float) | wmma::mma_sync() [INT TC] | default | |
| `IMMA.16816.S8.S8.SAT` | ~34.06 (similar) | wmma::mma_sync() [INT TC] | -G | |
| `IMMA.8832.S4.S4` | 28.05 cy/WMMA (INT4, fastest TC) | wmma::mma_sync() [INT TC] | default | |
| `IMMA.8832.S4.S4.SAT` | ~28.05 (similar) | wmma::mma_sync() [INT TC] | -G | |
| `IMMA.8832.U4.U4` | 28.05 cy/WMMA (=INT4 S4) | wmma::mma_sync() [INT TC] | default | |
| `IMMA.8832.U4.U4.SAT` | ~28.05 (similar) | wmma::mma_sync() [INT TC] | -G | |
| `LDSM.16.M88.4` | ~28 cy (shared memory matrix load) | wmma::load_matrix_sync() [from smem] | default | |
| `LDSM.16.MT88.4` |  | asm `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | direct confirm tranche | Direct local `sm_89` confirmation of the transposed x4 matrix-load form. |
| `MOVM.16.MT88` |  | register assignment | -G | |

### Texture/Surface (15 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `SULD.D.BA.1D.STRONG.SM` |  | surface load | default | |
| `SULD.D.BA.1D.STRONG.SM.IGN` |  | surface load | default | |
| `SULD.D.BA.1D.STRONG.SM.TRAP` |  | surface load | default | |
| `SULD.D.BA.2D.STRONG.SM` |  | surface load | TMU behavior probe | 2D surface load used by boundary-mode validation. |
| `SULD.D.BA.2D.STRONG.SM.IGN` |  | surface load | TMU behavior probe | 2D surface load with ignore/zero boundary behavior. |
| `SUST.D.BA.1D.STRONG.SM` |  | surface store | default | |
| `SUST.D.BA.1D.STRONG.SM.IGN` |  | surface store | default | |
| `SUST.D.BA.1D.STRONG.SM.TRAP` |  | surface store | default | |
| `SUST.D.BA.2D.STRONG.SM.TRAP` |  | surface store | TMU behavior probe | 2D surface store path. |
| `TEX.B.LL` |  | tex1D/2D/3D fetch | default | |
| `TEX.SCR.B.LL` |  | tex1D/2D/3D fetch | default | |
| `TEX.SCR.LL` |  | tex1D/2D/3D fetch | default | |
| `TLD.SCR.B.LZ` |  | texture load | default | |
| `TLD.SCR.LZ` |  | texture load | default | |

Texture/TMU note:
- The dedicated TMU runner validates point vs linear filtering, clamp/border/
  wrap/mirror address modes, and 1D/2D/3D interpolation against a CPU oracle.
- Manual bilinear/trilinear kernels show dense `FFMA` clusters that are absent
  from the hardware-filtered `TEX` kernels, which is the evidence for TMU
  interpolation offload.

### Uniform Datapath (22 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `UFLO.U32` |  |  | default | |
| `UIADD3` |  |  | default | |
| `UIADD3.X` |  |  | default | |
| `UIMAD` |  |  | default | |
| `UIMAD.WIDE` |  |  | tiling probes (-O3 --restrict) | **Uniform IMAD with 64-bit result** (for uniform 64-bit address computation) |
| `UIMAD.WIDE.U32` |  |  | default | |
| `UISETP.GE.AND` |  |  | INT tiling probes | Uniform unsigned greater-or-equal with AND combiner |
| `UISETP.GE.U32.AND` |  |  | --maxrregcount | |
| `UISETP.GT.AND` |  |  | --maxrregcount | |
| `UISETP.NE.U32.AND` |  |  | barrier probes | Uniform unsigned not-equal set-predicate |
| `USEL` |  |  | barrier probes | **Uniform conditional select** (uniform datapath equivalent of SEL) |
| `ULOP3.LUT` |  |  | default | |
| `UMOV` |  |  | default | |
| `UPOPC` |  |  | default | |
| `USHF.L.U32` |  |  | default | |
| `UPRMT` |  |  | INT8 tiling probes | **Uniform byte permute.** Uniform datapath equivalent of PRMT. Used for warp-uniform byte shuffle in INT8 pack/unpack patterns. |
| `USHF.R.S32.HI` |  |  | default | |
| `USHF.R.U32.HI` |  |  | bitops tiling probes | Uniform unsigned right shift high |
| `ULEA` | ~2.5 cy (uniform pipeline) |  | tiling probes | Uniform load effective address. Opcode 0x7891. |
| `ULEA.HI` |  |  | data movement probes | **Uniform LEA high.** Upper 32 bits of 64-bit uniform address computation. |
| `ULEA.HI.X` |  | warp-uniform 64-bit address carry | final uniform-u64 follow-up | Direct local `sm_89` confirmation from `probe_uniform_ushf_u64_hi_final.cu`. |
| `ULEA.HI.X.SX32` |  | warp-uniform signed byte-offset address generation | strict uniform follow-up | Direct local `sm_89` confirmation from `probe_uniform_u64_strict.cu`. |

### Warp Vote/Match/Redux (7 mnemonics)

| Mnemonic | Measured Latency | CUDA/PTX Intrinsic | Flags | Notes |
|---|---|---|---|---|
| `MATCH.ALL` | ~25 cy | __match_all_sync() | default | |
| `MATCH.ANY` |  | __match_any_sync() | default | |
| `VOTE.ALL` |  | __all_sync() | default | |
| `VOTE.ANY` |  | __any_sync()/__ballot_sync() | default | |
| `VOTEU.ANY` |  |  | default | |
| `WARPSYNC` |  | __syncwarp() | default | |
| `WARPSYNC.EXCLUSIVE` |  | __syncwarp() | -G | |

Warp-vote note:
- `VOTE.*` belongs to warp vote/ballot, not the packed SIMD-video family.
- In the current Ada corpus, `VOTE.ALL`, `VOTE.ANY`, and `VOTEU.ANY` are the
  only non-video `V*` raw spellings observed.

---

**Total: 470 unique SASS mnemonics across 25 categories.**

All latencies measured on RTX 4070 Ti (SM 8.9, 2625 MHz, 60 SMs).
See `SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md` for measurement methodology, ncu cross-validation,
and corrected values.
