// file: ysu_ipc_v1_4.y
// YSU OS — Canonical IPC Spec v1.4
// Features: Split WC/WB Memory Model, @gpu_uncached, Auto Cache-Line Alignment
// Changes from v1.3:
//   - data[] mapped as Write-Combining (GPU-visible, no CPU atomics)
//   - head/tail stay in Write-Back (CPU atomics valid here)
//   - @gpu_uncached attribute enforces this split at compile time
//   - mem.sfence() after every WC write to flush the 64-byte WC buffer
//   - @align(cache_line) automated via target.cache_line_size()

import std.mmu;       // map_gpu_portal, WC/WB split
import std.sync;      // @atomic, acquire/release
import hardware.nvda; // GPU DMA coherency

// ─── COMPILER HINT: TARGET CACHE LINE ────────────────────────────────────────
// Resolved at compile time per target triple.
// x86_64 → 64 bytes
// ARM Cortex-X / Apple M-series → 128 bytes
// Prevents false sharing on ALL targets without hardcoding.
const CACHE_LINE = target.cache_line_size();

// ─── RING BUFFER TYPE ─────────────────────────────────────────────────────────
//
// MEMORY SPLIT:
//   @gpu_uncached data[]  → Write-Combining (WC) memory
//                           CPU writes bypass L1/L2, flush to RAM in 64B bursts
//                           GPU DMA reads directly from RAM — coherent
//                           FORBIDDEN: atomic ops, pointer aliasing
//
//   @atomic head/tail     → Write-Back (WB) cached memory
//                           CPU atomic acquire/release ops valid here
//                           GPU reads these through a separate mapped WB region
//
// @gpu_uncached is a COMPILER ENFORCED attribute:
//   - Allocates field in WC/UC physical memory region
//   - Emits compile error if atomic op is attempted on this field
//   - Emits compile error if the field is accessed outside a chisel block
//     without an explicit mem.sfence() after the last write

type RingBuffer<T, SIZE> struct {
    // WC region — GPU sees this. No atomics allowed here.
    @gpu_uncached
    data: [T; SIZE],

    // WB region — CPU atomics live here. Auto cache-line padded.
    @align(CACHE_LINE) @atomic head: u64,
    @align(CACHE_LINE) @atomic tail: u64,
}

// ─── MMU SETUP: THE SPLIT PORTAL ─────────────────────────────────────────────
// Called once at system init by the chisel layer.
// Maps data[] as WC and index region as WB into GPU address space.

func setup_gpu_portal(rb: *RingBuffer<f32, 1024>) {
    chisel {
        // WC mapping — GPU DMA reads audio samples from here.
        // CPU writes to data[] will bypass L1/L2 and land in RAM directly
        // (after sfence flush).
        mmu.map_gpu_portal(
            ptr:    &rb.data,
            size:   size_of(rb.data),
            policy: .write_combining
        );

        // WB mapping — GPU reads head/tail indices through this region.
        // Normal cached memory — GPU can use its own cache coherency protocol
        // (NVLink or PCIe snooping) to stay in sync.
        mmu.map_gpu_portal(
            ptr:    &rb.head,
            size:   2 * CACHE_LINE, // head + tail, each on own cache line
            policy: .write_back
        );

        print("YSU IPC: GPU portal established. WC data, WB indices.");
    }
}

// ─── PRODUCER (CPU) ───────────────────────────────────────────────────────────
// acquire/release on WB head/tail — valid, atomic
// write to WC data[] — valid, but requires sfence before index release

func produce(rb: *RingBuffer<f32, 1024>, sample: f32) -> bool {
    chisel {
        let cur_head = rb.head.acquire();
        let cur_tail = rb.tail.acquire();
        let next_head = (cur_head + 1) % 1024;

        // Buffer full — drop sample (correct for real-time audio)
        // For physics/simulation channels: change to .block or .overwrite
        if next_head == cur_tail {
            return false;
        }

        // Write to WC memory.
        // WARNING: WC buffer accumulates writes — does NOT flush per-write.
        // A 4-byte f32 write fills 1/16th of the 64-byte WC buffer.
        rb.data[cur_head] = sample;

        // SFENCE: Force-flush the WC buffer to RAM NOW.
        // Without this, the GPU DMA may read the old poison value
        // because the sample is still sitting in the CPU WC buffer.
        // Cost: ~10-20 cycles on x86. Acceptable for audio paths.
        mem.sfence();

        // RELEASE store on head — signals GPU/consumer that data is ready.
        // Happens AFTER sfence, so data is guaranteed in RAM before
        // the index update is visible.
        rb.head.release(next_head);

        return true;
    }
}

// ─── CONSUMER (CPU PATH) ─────────────────────────────────────────────────────
// Used when CPU is the consumer (e.g. monitoring, Shield-side readback).
// GPU consumer path is handled by DMA — see ysu_gpu_consumer.cu

func consume(rb: *RingBuffer<f32, 1024>) -> Option<f32> {
    chisel {
        let cur_tail = rb.tail.acquire();
        let cur_head = rb.head.acquire();

        if cur_tail == cur_head {
            return Option.None;
        }

        // Read from WC memory. CPU reads from WC are uncached —
        // always fetch from RAM. No stale cache risk here.
        let sample = rb.data[cur_tail];

        // RELEASE tail — signals producer that slot is free.
        rb.tail.release((cur_tail + 1) % 1024);

        return Option.Some(sample);
    }
}

// ─── COMPILE-TIME INVARIANTS ─────────────────────────────────────────────────
// Y compiler enforces these as static assertions at build time.

@static_assert(is_power_of_two(1024),
    "RingBuffer SIZE must be power-of-two. Modulo compiles to AND only for 2^N.");

@static_assert(size_of(@atomic u64) <= CACHE_LINE,
    "Atomic index larger than cache line — padding insufficient.");

// ─── KNOWN REMAINING RISKS ───────────────────────────────────────────────────
//
// [RISK-1] SFENCE COST ON BATCH WRITES
//   Calling sfence() per sample at 192kHz = 192,000 sfences/sec.
//   Each sfence is ~10-20 cycles. At 4GHz: ~0.5ms/sec overhead.
//   Acceptable for audio. For bulk transfers (physics sim, NeRF training),
//   batch writes and sfence once per batch instead:
//
//     for i in 0..batch_size { rb.data[head + i] = batch[i]; }
//     mem.sfence();  // one flush for the whole batch
//     rb.head.release(head + batch_size);
//
// [RISK-2] MPSC / MPMC NOT SAFE
//   This spec is SPSC (one producer, one consumer) only.
//   Multiple producers on head without CAS → data race.
//   YSU fix: per-core ring buffers + work-stealing consumer (v2.0 target).
//
// [RISK-3] GPU CONSUMER MUST NOT USE C11 ATOMICS ON WC MEMORY
//   The GPU consumer (CUDA kernel) reads rb.data[] via DMA, not atomic ops.
//   Any attempt to use atomicLoad() on a WC-mapped address from CUDA
//   is undefined behavior. GPU side uses __ldg() (read-only cache) or
//   raw pointer dereference only.
//
// [RISK-4] PORTAL TEARDOWN ORDER
//   On shutdown, unmap WC data region BEFORE WB index region.
//   If indices are unmapped first, a concurrent GPU DMA read of data[]
//   may fault trying to validate the (now-unmapped) head index.
//   Correct teardown: quiesce GPU → unmap WC data → unmap WB indices.
