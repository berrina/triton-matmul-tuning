# Triton Matmul Kernel: A Tuning Study

A from-scratch GPU matrix multiplication kernel written in Triton, tuned through
systematic experimentation rather than guesswork — with two GPU-specific failure
modes discovered and explained along the way.

## What this is

I implemented tiled matrix multiplication as a custom GPU kernel (using Triton,
the Python-based kernel language), verified it against PyTorch/cuBLAS for
correctness, then ran a series of controlled experiments to understand what
actually makes a GPU kernel fast — tile shape, thread/warp count, and memory
access patterns — on an NVIDIA T4.

The point of this project wasn't to beat cuBLAS (a professional, decades-tuned
library) — it was to understand *why* GPU kernels perform the way they do, and
to practice reasoning about performance with real measurements instead of
intuition.

## Setup

- Environment: Google Colab, NVIDIA T4 GPU
- Framework: [Triton](https://triton-lang.org/) (installed via pip)
- Inputs: fp16, 1024×1024×1024 matmul, fp32 accumulation
- Timing: `triton.testing.do_bench`, warmed up, median of repeated runs
- [Colab notebook](https://colab.research.google.com/drive/10mo0tFX47731Pof0ue7cDWO47yoaWEwC?hl=en)

## Correctness

Verified against `torch.matmul` (cuBLAS): max absolute error of 0.062,
passing `allclose` at `atol=1e-1`.

This isn't a bug — it's expected floating-point divergence. Inputs are fp16
(~3 decimal digits of precision), each output element sums 1024 products, and
Triton and cuBLAS sum those products in different orders. Floating-point
addition isn't associative, so different summation orders produce slightly
different rounding — this is normal numerical fuzz, not an error in the kernel.

Accumulation is done in fp32 even though inputs are fp16: summing 1024 values
in fp16 would compound rounding error and risks overflow (fp16 maxes out at
65,504). Loading in fp16 (cheap on memory bandwidth) and accumulating in fp32
(numerically safe) is the standard pattern — it's also how tensor cores are
built to operate.

## Experiment 1: Tile size sweep

**Question:** which tile shape (BLOCK_M × BLOCK_N × BLOCK_K) is fastest, and why?

**Method:** swept all combinations of BLOCK_M, BLOCK_N ∈ {32, 64, 128} and
BLOCK_K ∈ {32, 64} — 18 configurations total, same kernel, same problem size.

| BM  | BN  | BK | ms     | TFLOPS |
|-----|-----|----|--------|--------|
| 32  | 64  | 32 | 0.755  | **2.8** |
| 32  | 64  | 64 | 0.787  | 2.7 |
| 64  | 32  | 64 | 0.946  | 2.3 |
| 64  | 32  | 32 | 0.967  | 2.2 |
| 64  | 64  | 32 | 1.477  | 1.5 |
| 32  | 128 | 32 | 1.408  | 1.5 |
| 128 | 32  | 32 | 1.728  | 1.2 |
| 32  | 32  | 64 | 1.871  | 1.1 |
| 32  | 32  | 32 | 2.067  | 1.0 |
| 32  | 128 | 64 | 11.257 | 0.2 |
| 64  | 64  | 64 | 10.670 | 0.2 |
| 128 | 32  | 64 | 10.304 | 0.2 |
| 64  | 128 | 64 | 20.833 | 0.1 |
| 64  | 128 | 32 | 21.802 | 0.1 |
| 128 | 64  | 32 | 22.054 | 0.1 |
| 128 | 64  | 64 | 20.286 | 0.1 |
| 128 | 128 | 32 | 35.347 | 0.1 |
| 128 | 128 | 64 | 34.990 | 0.1 |

**Finding 1 — measuring beats guessing.** The best configuration (32×64×32,
2.8 TFLOPS) wasn't my original hand-picked one (64×64×32, which ranked 5th).
Sweeping the full space found a 2.3x improvement over the initial guess —
the exact reason autotuning exists in production kernel libraries.

**Finding 2 — the silent cliff.** Several large configurations didn't error
out — they just became 10-28x slower while still producing correct results.
This is consistent with register/shared-memory spilling: when a tile needs
more on-chip storage than the GPU has, the compiler silently falls back to
slow global memory instead of failing. A kernel can be *correct* and
*catastrophically slow* at the same time, with no exception ever thrown —
a real GPU performance trap worth knowing about.

**Finding 3 — wide beats tall (unconfirmed).** Mirror-pair configurations
consistently favor a wider BLOCK_N over a taller BLOCK_M (e.g. 32×64 → 2.8
TFLOPS vs. 64×32 → 2.2 TFLOPS). Hypothesis: memory accesses along rows of B
and C are contiguous, so a wider tile means longer contiguous (coalesced)
loads. Not yet confirmed with a profiler — noted as a next step.

New baseline after this experiment: **2.8 TFLOPS** (~19% of cuBLAS, up from
the initial 8%).

## Experiment 2: Warps and pipelining sweep

**Question:** does overlapping memory loads with computation (pipelining),
or changing the number of parallel warps per tile, improve throughput further?

**Method:** swept `num_warps` ∈ {2, 4, 8} × `num_stages` ∈ {1..5} on the
winning 32×64×32 tile shape from Experiment 1.

| Warps | Stages | ms    | TFLOPS |
|-------|--------|-------|--------|
| 4     | 2      | 0.583 | **3.7** |
| 4     | 4      | 0.582 | 3.7 |
| 4     | 1      | 0.589 | 3.6 |
| 4     | 3      | 0.599 | 3.6 |
| 4     | 5      | 0.592 | 3.6 |
| 8     | 2      | 0.725 | 3.0 |
| 8     | 5      | 0.724 | 3.0 |
| 8     | 3      | 0.732 | 2.9 |
| 8     | 1      | 0.729 | 2.9 |
| 8     | 4      | 0.732 | 2.9 |
| 2     | 4      | 1.678 | 1.3 |
| 2     | 1      | 1.678 | 1.3 |
| 2     | 5      | 1.683 | 1.3 |
| 2     | 2      | 1.681 | 1.3 |
| 2     | 3      | 1.679 | 1.3 |

**Finding 0 — caught a benchmarking bug.** The warps=4/stages=1 case should
have reproduced Experiment 1's 2.8 TFLOPS baseline, but measured 3.6 instead.
Root cause: the earlier benchmark allocated the output tensor *inside* the
timed region, so it was partly timing memory allocation, not just the kernel.
The corrected true baseline for this tile shape is 3.6 TFLOPS — Experiment
1's *relative* ranking of tile shapes still holds, since all 18 configs were
measured with the same (flawed) harness consistently. Lesson: a benchmark
compares harnesses as much as kernels — only numbers measured identically
are directly comparable.

**Finding A — pipelining bought almost nothing, for a hardware reason.**
Throughput was flat (3.6–3.7 TFLOPS) across 1 to 5 pipeline stages. This
traces back to hardware: multi-stage pipelining depends on asynchronous
memory copies (`cp.async`), a feature introduced in NVIDIA's Ampere
architecture. The T4 is Turing (2018) and predates it — so extra pipeline
stages mostly consume shared memory without unlocking real overlap. A
missing hardware feature, visible directly as a flat line in a benchmark.

**Finding B — warp count matters a lot (3x spread).** 4 warps reached 3.7
TFLOPS, 8 warps only 3.0, and 2 warps just 1.3. Two warps (64 threads) can't
generate enough in-flight work to hide memory latency for this tile size;
eight warps oversubscribe it, with too little work per warp to go around.
The optimal warp count depends on tile size — these tuning knobs interact,
they aren't independent.

Final baseline reached: **3.7 TFLOPS** (32×64×32, num_warps=4, num_stages=2)
— roughly 25% of cuBLAS's 14.7 TFLOPS. Honest attribution: tile shape drove
almost all of the improvement; warp/stage tuning added a modest ~3% on top
of the corrected baseline.

## Summary

| Stage | Config | TFLOPS | % of cuBLAS |
|---|---|---|---|
| Initial (hand-picked) | 64×64×32 | 1.2 | 8% |
| After tile sweep | 32×64×32 | 2.8 | 19% |
| After warp/stage sweep | 32×64×32, warps=4, stages=2 | 3.7 | 25% |

## What I'd do next

- **Grouped tile ordering** to improve L2 cache reuse across thread blocks —
  parked as the next lever after tile shape and launch params were exhausted.
- **Nsight Compute profiling** to directly confirm the coalesced-memory
  hypothesis from Finding 3, rather than relying on it as an educated guess.
- Extend the same tile/warp sweep methodology to a second kernel (e.g.
  softmax or a simple attention op) to see how much of this transfers.

## What I learned

Kernel performance is dominated by a small number of structural decisions
(tile shape) more than fine-grained tuning knobs (warp count, pipeline
stages) — but both matter, and neither is safe to guess. Two GPU-specific
failure modes showed up in this project alone: silent performance cliffs
from memory spilling, and a hardware generation gap that made an entire
optimization technique (pipelining) ineffective on this GPU. Both were only
visible because every claim here was checked against a measurement, including
catching a mistake in my own benchmark.
