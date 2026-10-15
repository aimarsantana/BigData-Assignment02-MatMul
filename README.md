# Assignment 2 — Optimised Dense Multiplication and Sparse Matrices (C)

**Big Data · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027**
Author: Aimar Daniel Alejandro Santana

Research questions:

* **A.** *How much do loop reordering, loop unrolling and cache blocking speed up the sequential
  dense `float64` multiplication relative to the basic triple loop of Assignment 1, and how does
  that change with the matrix size n?*
* **B.** *For which tested sparsity levels and input structures does a CSR (Gustavson) sparse
  product beat dense multiplication on the same logical matrices, once conversion, indexing and
  fill-in of the product are taken into account?*

The report is `../Assignment02_AlejandroSantana_Report.pdf` (source: `report/build_report.py`).

## What is in this folder

```
src/c/matmul2.c            all kernels (C, float64, single thread)
                             dense : ijk (baseline, reused from Assignment 1), ikj, unroll4, blocked
                             sparse: csr (own Gustavson SpGEMM on CSR), convert (dense -> CSR)
scripts/make_config.py     writes config/benchmark_config.json (inputs + experiments + budgets)
scripts/generate_inputs.py shared dense inputs (same seeds as Assignment 1) and sparse inputs
scripts/run_benchmarks.py  benchmark harness (one fresh process per configuration, budgets, stop rule)
scripts/analyze.py         statistics, fits, figures/*.png|pdf, report/generated/*.html
scripts/collect_environment.ps1   hardware / software description -> results/environment.txt
tests/validate.py          correctness tests (known answers, identity, zero, NumPy reference, bit-exactness)
report/build_report.py     assembles the English report (HTML) and prints it to PDF with Chrome/Edge
results/                   raw_samples.csv, runs.csv, summary.csv, validation.csv, logs/, ... (see below)
figures/                   all figures (PNG + PDF)
config/benchmark_config.json   sizes, repetitions, warm-ups, budgets, sparse input list, experiments
build.ps1 / run_all.ps1    build and full reproduction (Windows PowerShell)
```

## Reused material (Assignment 1)

* `ijk` is the C triple loop of `Assignment01_AlejandroSantana_Materials/src/c/matmul.c`
  (same loop body, same row-pointer storage, same scalar accumulator). Only the program wrapper
  changed (Windows timer / memory API, auto-batching, additional methods).
* The dense input matrices are the Assignment 1 matrices: NumPy `default_rng` (PCG64), seeds
  20261012 (A) and 20261013 (B), uniform in [-1, 1). `generate_inputs.py --check-a1 <manifest.json>`
  compares the SHA-256 of every regenerated file with the Assignment 1 manifest (all 18 shared
  files matched).

## Requirements

* Windows 10/11, **gcc** (MinGW-w64; tested: WinLibs gcc 16.1.0 UCRT).
* **Python 3.12+** with `numpy`, `pandas`, `matplotlib` (tested: 3.14.5, numpy 2.5.4, pandas 3.0.6,
  matplotlib 3.11.2). With [uv](https://docs.astral.sh/uv/) no installation is needed:
  `uv run --with numpy --with pandas --with matplotlib python <script>`.
* Google Chrome or Microsoft Edge (only to print the report to PDF).
* About 1.8 GB of free disk space for the generated inputs and 4 GB of free RAM.

## Reproduce everything

```powershell
.\run_all.ps1                      # build -> environment -> inputs -> validation -> benchmarks -> analysis -> report
```

Step by step (inputs are stored outside the project folder, here `C:\bench_a2\data`, so that cloud
sync does not interfere with the timings):

```powershell
.\build.ps1                                                    # gcc -O2 -std=c11 -Wall -Wextra -lpsapi
.\scripts\collect_environment.ps1                              # results\environment.txt
uv run --with numpy python scripts/generate_inputs.py --outdir C:\bench_a2\data `
        --check-a1 ..\Assignment01_AlejandroSantana_Materials\data\inputs\manifest.json
uv run --with numpy python tests/validate.py --data-dir C:\bench_a2\data > results\validation.csv
uv run python scripts/run_benchmarks.py --data-dir C:\bench_a2\data D2_block          # tile-size sweep
uv run python scripts/run_benchmarks.py --data-dir C:\bench_a2\data D1_size S1_sparsity S2_size_z99 `
        S3_size_z99.9 S4_band_width S5_band_size
uv run --with numpy --with pandas --with matplotlib python scripts/analyze.py
uv run --with numpy --with pandas --with matplotlib python report/build_report.py
```

(`run_benchmarks.py` only needs the standard library; `--results DIR` redirects the output.)
The tile size of the `blocked` method is fixed from the sweep `D2_block` and stored in
`config/benchmark_config.json` (`block_size`; regenerate with `scripts/make_config.py --block B`).
The complete run takes roughly 1–1.5 hours.

Running one kernel by hand:

```powershell
build\matmul2.exe dense blocked A_n1024.bin B_n1024.bin 1024 1 5 0 64 [C_out.bin]
#   dense <ijk|ikj|unroll4|blocked> A.bin B.bin n warmup reps batch block [C_out.bin]
build\matmul2.exe csr A_n1024_uni_z99.csr B_n1024_uni_z99.csr 1 10 0 [C_out.csr]
build\matmul2.exe convert A_n1024_uni_z99.bin B_n1024_uni_z99.bin 1024 1 10
#   batch 0 = automatic (a sample lasts ~20 ms, otherwise one multiplication)
```

Every program prints `SAMPLE,<method>,<n>,<phase>,<index>,<batch>,<ns per multiplication>,<checksum>,<nnzA>,<nnzB>,<nnzC>`
for each warm-up and measured run, and `MEM,...` lines with the peak working set and peak commit charge.

## Inputs

| Family | Files | Generation |
|---|---|---|
| Dense (Part A) | `A_n<n>.bin`, `B_n<n>.bin` (raw little-endian float64, row-major) | identical to Assignment 1 (PCG64, seeds 20261012/13, U[-1,1)) |
| Uniform sparse | `A_n<n>_uni_z<Z>.{csr,bin}` | every row has exactly k = round(n·(1−Z/100)) non-zeros at columns drawn uniformly without replacement; values U[-1,1); position seeds 20261026 (A) / 20261027 (B), value seeds 20261028 / 20261029 (via `SeedSequence([seed, n, pattern id])`) |
| Banded sparse | `A_n<n>_band_w<w>.{csr,bin}` | entry (i,j) non-zero iff \|i−j\| ≤ w; values as above |

The `.csr` and `.bin` files of a case are two representations of the SAME logical matrix (checked in
`tests/validate.py`). Z = percentage of zero entries (sparsity); density = 100 − Z. The dense
(`.bin`) copies are only written for n ≤ 4096. `data/inputs/manifest.json` stores SHA-256 hashes and
non-zero counts. The input files are not committed (regenerate them with `generate_inputs.py`).

## Experimental protocol (summary — details in the report)

* Single thread, process pinned to one logical processor, high priority; AC power, "Balanced" plan.
* Timer: `QueryPerformanceCounter`. Timed region: dense — the kernel, including zeroing of C for
  ikj/unroll4/blocked, C allocated outside; CSR — the whole product including allocation of the
  output; conversion — both dense→CSR conversions including allocation. Inputs are loaded outside.
* Warm-up: 2 runs for n ≤ 512, 1 run above (kept in the raw data, excluded from statistics).
* Repetitions: 10 (n ≤ 512), 5 (n ≤ 2048), 3 (n > 2048) for dense; 10 for the sparse kernels.
  Very short kernels are batched automatically (batch printed in every sample).
* Budget: 120 s per single sample, 4 GB peak working set. A series that times out / fails / exceeds
  the memory budget is not attempted at larger sizes (`NOT_ATTEMPTED` rows in `results/runs.csv`).
* Memory metric: peak working set (`PeakWorkingSetSize`) and peak commit charge of the benchmark
  process, read by the program itself through `GetProcessMemoryInfo`.
* Statistics: median and interquartile range; every raw sample is kept.
* Correctness: `tests/validate.py` (known 3×3 product, identity, zero, empty rows/columns, sizes not
  divisible by the tile or the unroll factor, benchmark inputs) against NumPy with
  |ΔC| ≤ 1e-9 + 1e-9·|C|, and bit-exact equality with the baseline; the result checksum of all
  methods is compared again for every benchmark configuration (`results/checksum_check.csv`).

## AI-use disclosure

Generative AI (Claude, Anthropic) was used as described in the "AI-use disclosure" section of the
report. All measurements in `results/` were produced by running the code in this folder.

Repository: https://github.com/aimarsantana/BigData-Assignment02-MatMul
