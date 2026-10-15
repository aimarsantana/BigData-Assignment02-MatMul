#!/usr/bin/env python3
"""Assemble the Assignment 2 report (HTML, figures embedded) and print it to PDF with Chrome/Edge.

Tables come from report/generated/*.html and figures from figures/*.png (both written by
scripts/analyze.py). The narrative quotes numbers that were read from those tables and from
results/key_numbers.json; the few numbers computed here come from key_numbers.json.

Usage: python report/build_report.py [--no-pdf]
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEN = os.path.join(ROOT, "report", "generated")
FIG = os.path.join(ROOT, "figures")
OUT_HTML = os.path.join(ROOT, "report", "Assignment02_AlejandroSantana_Report.html")
OUT_PDF = os.path.join(ROOT, "report", "Assignment02_AlejandroSantana_Report.pdf")
REPO_URL = "https://github.com/aimarsantana/BigData-Assignment02-MatMul"

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt Georgia, serif; color: #555; } }
body { font: 10pt/1.38 Georgia, 'Times New Roman', serif; color: #111; background: #fff; max-width: 100%; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }
.sub { text-align: center; font-size: 11pt; margin: 2pt 0; }
.auth { text-align: center; font-size: 9.5pt; margin: 8pt 0 12pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 4pt; border-bottom: 0.6pt solid #888; padding-bottom: 1pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0; text-align: justify; hyphens: auto; }
.abstract { margin: 8pt 22pt; font-size: 9.5pt; }
table { border-collapse: collapse; margin: 8pt auto; font: 8.5pt/1.25 'Segoe UI', Arial, sans-serif; break-inside: avoid; }
table.small { font-size: 7.4pt; }
caption { caption-side: top; font: 8.5pt/1.3 Georgia, serif; text-align: left; margin-bottom: 3pt; font-weight: normal; }
th, td { border-top: 0.5pt solid #888; border-bottom: 0.5pt solid #888; padding: 2pt 5pt; text-align: right; }
th { background: #f0f0f0; text-align: center; }
td:first-child, th:first-child { text-align: center; }
.iqr { color: #666; font-size: 0.9em; }
figure { margin: 8pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.5pt; text-align: left; margin-top: 2pt; }
code { font: 8.8pt Consolas, 'Courier New', monospace; }
ul, ol { margin: 3pt 0 3pt 16pt; padding: 0; } li { margin: 1.5pt 0; text-align: justify; }
.env td { text-align: left; } .env td:first-child { font-weight: bold; }
.ref li { text-align: left; font-size: 8.8pt; }
"""


def fig(name, caption, width="100%"):
    path = os.path.join(FIG, name + ".png")
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img style="width:{width}" src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>'


def table(name):
    return open(os.path.join(GEN, name), encoding="utf8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    k = json.load(open(os.path.join(ROOT, "results", "key_numbers.json")))
    val_rows = [l for l in open(os.path.join(ROOT, "results", "validation.csv"), encoding="utf8").read().splitlines()[1:] if l]
    n_pass = sum(1 for l in val_rows if l.endswith("PASS") or "PASS" in l.split(",")[-1])
    n_fail = sum(1 for l in val_rows if "FAIL" in l)
    n_bit = sum(1 for l in val_rows if ",True," in l)
    fits = {(f["experiment"], f["method"]): f["slope"] for f in k["fits"]}

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Assignment 2 - Optimised Dense Multiplication and Sparse Matrices</title><style>{CSS}</style></head><body>

<h1>Which Optimisations Pay Off? Loop Reordering, Unrolling and Cache Blocking for Dense Matrix Multiplication, and When CSR Sparse Multiplication Wins</h1>
<div class="sub">Big Data &ndash; Individual Assignment 2: Optimised Matrix Multiplication Approaches and Sparse Matrices</div>
<div class="auth"><b>Aimar Daniel Alejandro Santana</b><br>Grado en Ciencia e Ingenier&iacute;a de Datos, Universidad de Las Palmas de Gran Canaria<br>
Repository: {REPO_URL}<br>October 2026</div>

<div class="abstract"><b>Abstract.</b> We study two questions on one machine (AMD Ryzen 7 7730U, Windows 11, single thread, C, binary64):
(A) how much three sequential optimisations&mdash;loop reordering (ikj), k-loop unrolling by 4, and cache blocking&mdash;speed up the
basic triple loop (ijk) of Assignment 1, and (B) for which tested sparsity levels and input structures a CSR (Gustavson) sparse product beats dense
multiplication on the same logical matrices. All methods read the same input files, give bit-identical results, and are timed on the kernel only
(median and IQR of 3&ndash;10 repetitions, warm-up excluded). Loop reordering alone makes the dense product {k['speedup_ikj_n1024']:.0f}&times; faster at n&nbsp;=&nbsp;1024; adding
a 4-fold unrolling reaches {k['speedup_unroll4_n1024']:.0f}&times; (n&nbsp;=&nbsp;1024) and {k['speedup_unroll4_n2048']:.0f}&times; (n&nbsp;=&nbsp;2048). Blocking helps ikj only once the matrix B no longer fits in the
16&nbsp;MB L3 cache (1.3&ndash;1.4&times; for n&nbsp;&ge;&nbsp;1536), but did not beat the unrolled kernel in our tests. At n&nbsp;=&nbsp;1024 the CSR product is slower than the best dense method on fully dense
inputs (0.29&times;), roughly on par at 50&nbsp;% zeros (1.07&times;), and 10&ndash;250&times; faster at 90&ndash;99.5&nbsp;% zeros. Dense&rarr;CSR conversion costs about 3&nbsp;ms at n&nbsp;=&nbsp;1024
whatever the sparsity, so it limits the advantage at very high sparsity (11,700&times; for the kernel alone, 85&times; with conversion at 99.9&nbsp;%). With the same density, banded inputs
(no fill-in) are 15&times; faster than uniformly random ones. The conclusions hold for the tested code, inputs and hardware only.</div>

<h2>1. Introduction and objective</h2>
<p>Assignment 1 measured a basic triple-loop multiplication in three languages and found that, even in C, the cost per inner iteration grew with n because
the column-wise access to B leaves the caches. This second study takes the C baseline and asks what the simplest source-level optimisations buy, and when
exploiting zeros pays off. Dense matrices are n&times;n; n is the side length.</p>
<p><b>Research questions.</b> <b>(A)</b> How do loop reordering, loop unrolling and cache blocking change the runtime and memory of a sequential dense
binary64 multiplication relative to the baseline, as n grows? <b>(B)</b> For which declared sparsity levels (percentage of zero entries) and sparsity
structures does a CSR sparse product beat dense multiplication on matched logical inputs, once conversion, indexing and the density of the output are taken
into account?</p>
<p><b>Predictions</b> (formulated from the background in Section 2 before the main runs; the short debugging runs at n&nbsp;&le;&nbsp;2048 that preceded them showed that blocking gave no gain over ikj at n&nbsp;=&nbsp;2048 with the first tile sizes tried):</p>
<ol>
<li><b>H1.</b> ikj is at least 5&times; faster than ijk for n&nbsp;&ge;&nbsp;1024, because the inner loop becomes unit-stride in B and C.</li>
<li><b>H2.</b> Unrolling the k loop by 4 adds a modest further gain (1.2&ndash;2&times;), because C[i][*] is loaded and stored once per four multiply-adds.</li>
<li><b>H3.</b> Blocking helps (relative to ikj) only when B no longer fits in the last-level cache (n&nbsp;&gtrsim;&nbsp;1500 here); the best tile side is a few tens of elements.</li>
<li><b>H4.</b> With uniformly random zeros, the CSR product overtakes the best dense method at roughly 90&nbsp;% zeros; counting conversion moves this point to higher sparsity.</li>
<li><b>H5.</b> Sparsity alone does not determine performance: the product of random sparse matrices fills in (so output cost stays high), whereas banded matrices of the same density do not.</li>
</ol>

<h2>2. Background</h2>
<p><b>Dense product and loop order.</b> C=AB needs n&sup3; multiplications and n&sup3; additions (2n&sup3; flops) [1]. All six loop orders perform the same arithmetic but
traverse memory differently. In the baseline ijk order the inner loop walks down a column of B with stride n, so for large n each access touches a new cache line
and a new page [5, 8]. In ikj order the inner loop reads a row of B and updates a row of C with unit stride, which suits hardware prefetching and cache lines.
<b>Loop unrolling</b> replicates the loop body to reduce loop overhead and, here, memory operations on C. <b>Cache blocking</b> (tiling) restructures the loops to work on sub-matrices
that fit in a cache level so that every loaded element is reused many times [2]; high-performance libraries combine it with register tiling and SIMD [6]. We did not
implement Strassen&rsquo;s sub-cubic algorithm [7]: its recursion overhead and numerical behaviour make it a separate study.</p>
<p><b>Sparse matrices.</b> A matrix with mostly zero entries can be stored in CSR (compressed sparse row) format: arrays <code>val</code> (non-zero values), <code>col</code> (column indices)
and <code>rowptr</code> (start of each row) [4, 11]. Gustavson&rsquo;s algorithm [3] multiplies row by row: for every non-zero A[i][k] the scaled row k of B is merged into a dense
accumulator indexed by column, and the touched columns are then gathered (and sorted) into row i of C. Its work is proportional to the number of non-zero products, not to n&sup3;,
but each product costs an indirect access. If A and B have density d with independent random positions, an entry of C is structurally non-zero with probability 1&minus;(1&minus;d&sup2;)<sup>n</sup>,
so the product of two sparse random matrices quickly becomes dense (&ldquo;fill-in&rdquo;); banded matrices of half-bandwidth w give a product of half-bandwidth 2w.</p>

<h2>3. Methodology</h2>
<h3>3.1 Implementations</h3>
<table class="env"><caption>Table 1. Methods (all in <code>src/c/matmul2.c</code>, C11, binary64, one thread; own code unless stated).</caption>
<thead><tr><th>Method</th><th>Change relative to the previous row / expected effect</th></tr></thead><tbody>
<tr><td>ijk</td><td>Baseline reused from Assignment 1 (<code>matmul.c</code>): triple loop, scalar accumulator, row-pointer storage <code>double**</code>. Column-wise (stride-n) access to B.</td></tr>
<tr><td>ikj</td><td>Loop reordering: C[i][*] += A[i][k]&middot;B[k][*]; unit stride in B and C; C is zeroed first (inside the timer).</td></tr>
<tr><td>unroll4</td><td>ikj with the k loop unrolled by 4 (four rows of B folded into C[i][*] per pass, summed in increasing k): fewer loads/stores of C per flop.</td></tr>
<tr><td>blocked</td><td>ikj kernel on tiles of side b (loops ii, kk, jj; k blocks in increasing order). b = {{}} fixed from the sweep of Section 4.2.</td></tr>
<tr><td>csr</td><td>Sparse&times;sparse product on CSR (int32 indices, binary64 values), Gustavson with dense accumulator, sorted output columns; own code, no library.</td></tr>
<tr><td>convert</td><td>Dense&rarr;CSR conversion of A and B (own code), timed separately from the product.</td></tr></tbody></table>
<p>Every dense kernel adds the products of C[i][j] in increasing k starting from +0.0, with no FMA contraction (generic x86-64 target, no <code>-ffast-math</code>); so the
results are expected to be <em>bit-identical</em>, which we verify. The compiler (gcc&nbsp;16.1.0, <code>-O2</code>) emitted only scalar SSE2 instructions (<code>mulsd/addsd</code>) in the four dense kernels
(checked with <code>objdump -d</code>): no SIMD vectorisation occurs, so differences between methods are due to memory access and instruction count, not to vector width.
Parallel and vectorised variants are out of scope (Assignment 3). Baseline and optimised methods use the same storage, numeric type, input matrices and timing boundary.</p>
<h3>3.2 Inputs</h3>
<p><b>Dense (Part A).</b> The matrices of Assignment 1: NumPy PCG64, seeds 20261012 (A) and 20261013 (B), U[&minus;1,1), binary64, little-endian row-major files, n&nbsp;&isin;&nbsp;{{64,&hellip;,4096}}; the SHA-256 of all 18 regenerated files
that also exist in Assignment 1 matched its manifest. <b>Sparse (Part B).</b> Sparsity is the percentage of zero entries (density = 100 &minus; sparsity). <i>Uniform</i>: each row of A and B has exactly
round(n(1&minus;sparsity)) non-zeros at uniformly random distinct columns (position seeds 20261026/27, value seeds 20261028/29, values U[&minus;1,1)). <i>Banded</i>: entry (i,j) is non-zero iff |i&minus;j|&nbsp;&le;&nbsp;w.
Each case is written in two representations of the same logical matrix (dense file for the dense methods, CSR file for the sparse method; checked equal). The sparsity-0&nbsp;% case uses the dense matrices of Part A.
Experiments: <b>D1</b> dense size sweep (n&nbsp;=&nbsp;64&ndash;4096); <b>D2</b> tile-size sweep; <b>S1</b> sparsity sweep (n&nbsp;=&nbsp;1024; 0, 50, 80, 90, 95, 98, 99, 99.5, 99.9&nbsp;% zeros); <b>S2</b>/<b>S3</b> size sweeps at 99&nbsp;% and 99.9&nbsp;% zeros;
<b>S4</b> banded widths w&nbsp;&isin;&nbsp;{{0,2,5,13,51}} (n&nbsp;=&nbsp;1024); <b>S5</b> banded size sweep (w&nbsp;=&nbsp;5). Dense methods were run on the matched dense file of each sparse case (all four methods in S1, S2, S4; blocked only in S3, S5; no dense input was generated above n&nbsp;=&nbsp;4096).</p>
<h3>3.3 Environment</h3>
<table class="env"><caption>Table 2. Measurement environment (<code>results/environment.txt</code>).</caption><tbody>
<tr><td>CPU</td><td>AMD Ryzen 7 7730U (laptop), 8 physical cores / 16 logical processors, L2 4&nbsp;MB total (8&times;512&nbsp;KB), L3 16&nbsp;MB, max clock reported by Windows 2.0&nbsp;GHz (turbo not controlled)</td></tr>
<tr><td>Memory / OS</td><td>13.84&nbsp;GB RAM (2&times;8&nbsp;GB DDR4-3200); Windows 11 Home build 26200; AC power, &ldquo;Balanced&rdquo; power plan</td></tr>
<tr><td>Toolchain</td><td>gcc 16.1.0 (WinLibs MinGW-w64 UCRT), <code>-O2 -std=c11 -Wall -Wextra -lpsapi</code>; Python 3.14.5, NumPy 2.5.4, pandas 3.0.6, matplotlib 3.11.2 (inputs, validation, analysis only)</td></tr>
<tr><td>Threads</td><td>1. Process pinned to one logical processor (#2), high priority; no numerical library is called in any timed region.</td></tr></tbody></table>
<h3>3.4 Correctness checks</h3>
<p><code>tests/validate.py</code> ran {len(val_rows)} checks (<b>{n_pass} pass, {n_fail} fail</b>) before benchmarking: a hand-computed 3&times;3 product, identity and zero matrices, sizes not divisible by the unroll factor or the tile
(n&nbsp;=&nbsp;1, 2, 5, 33, 100, 127, 130 with tiles 1, 7, 32, 64, 500), sparse matrices with forced empty rows/columns and extreme densities (0 and 100&nbsp;%), the benchmark inputs, and a check that the CSR structure of the output
(sorted unique columns, nnz equal to the structurally reachable positions) is valid. The reference is NumPy (OpenBLAS), used only for checking. The tolerance is |&Delta;C|&nbsp;&le;&nbsp;10<sup>&minus;9</sup>&nbsp;+&nbsp;10<sup>&minus;9</sup>|C|:
the rounding error of a length-n dot product with entries in [&minus;1,1] is bounded by n&middot;u&middot;&sum;|a<sub>ik</sub>b<sub>kj</sub>|&nbsp;&le;&nbsp;n&sup2;u&nbsp;=&nbsp;1.2&times;10<sup>&minus;10</sup> for n&nbsp;=&nbsp;1024 (u&nbsp;=&nbsp;2<sup>&minus;53</sup>), so 10<sup>&minus;9</sup> is above any legitimate
rounding error and far below logic errors (largest error on the benchmark inputs 1.4&times;10<sup>&minus;13</sup>). In addition, every optimised kernel must equal the baseline <em>bit for bit</em> ({n_bit} comparisons, all equal), and
the result checksum was compared across all methods for each of the {k['checksum_groups']} benchmark configurations (disagreements: {k['checksum_disagreements']}).</p>
<h3>3.5 Measurement protocol</h3>
<ul>
<li><b>Timer boundary (kernel only).</b> <i>Dense:</i> the multiplication, including zeroing of C for ikj/unroll4/blocked; C is allocated outside the timer. <i>CSR:</i> the whole product including allocation and growth of the output (its size is data dependent), which
makes the sparse boundary slightly wider than the dense one. <i>Conversion:</i> both dense&rarr;CSR conversions including allocation. Inputs are loaded and results are checksummed outside the timer. The two sparse timings are reported separately and as their sum.</li>
<li><b>Timer and repetitions.</b> <code>QueryPerformanceCounter</code> (monotonic). One fresh process per configuration. Warm-up: 2 runs (n&nbsp;&le;&nbsp;512) or 1 run, excluded from statistics but kept in the raw data. Measured repetitions: 10 (n&nbsp;&le;&nbsp;512), 5 (n&nbsp;&le;&nbsp;2048), 3 (n&nbsp;&gt;&nbsp;2048; fewer because a run takes up to 40&nbsp;s), and 10 for the sparse kernels.
Short kernels are batched automatically so that a sample lasts &asymp;20&nbsp;ms (the time per multiplication is reported; the batch is stored in every row). The result is consumed through a checksum.</li>
<li><b>Statistics.</b> Median and interquartile range (IQR) of the measured samples; all raw samples are in <code>results/raw_samples.csv</code>. No sample was removed.</li>
<li><b>Budgets.</b> 120&nbsp;s per sample and 4&nbsp;GB peak working set. A series that exceeds the budget is not attempted at larger sizes; timeouts are reported as results, with no invented time. The largest <em>tested feasible</em> size and the largest <em>attempted</em> size are reported separately.</li>
<li><b>Memory metric.</b> Peak working set (<code>PeakWorkingSetSize</code>, the Windows analogue of peak resident set) of the benchmark process, read by the process itself via <code>GetProcessMemoryInfo</code> [10]; the peak commit charge is also stored. It includes the runtime,
the loaded inputs and all temporaries, not only the matrices; it is compared with the analytic size of the arrays (8n&sup2; bytes per dense matrix; 12&middot;nnz + 4(n+1) bytes per CSR matrix).</li></ul>

<h2>4. Results</h2>
<h3>4.1 Part A: dense optimisation</h3>
{table('table_dense_time.html')}
{fig('fig_dense_runtime', 'Figure 1. Time per multiplication of the four dense methods (log&ndash;log, median and IQR; the dotted line is proportional to n<sup>3</sup>). The ijk series stops at n&nbsp;=&nbsp;2048: n&nbsp;=&nbsp;3072 exceeded the 120&nbsp;s budget (TIMEOUT), n&nbsp;=&nbsp;4096 was not attempted.', '78%')}
{table('table_dense_speedup.html')}
{fig('fig_dense_speedup', 'Figure 2. Speed-up of the optimised methods over the ijk baseline (ratio of medians).', '70%')}
<p>The median relative IQR over all dense configurations is {100 * k['dense_median_rel_iqr']:.1f}&nbsp;% (maximum {100 * k['dense_max_rel_iqr']:.0f}&nbsp;%, at the smallest sizes), so the differences discussed below (&ge;&nbsp;30&nbsp;%) exceed the run-to-run spread inside a session.
The fitted log&ndash;log slopes of time versus n (n&nbsp;&ge;&nbsp;256) are {fits[('D1_size', 'ijk')]:.2f} (ijk, up to 2048), {fits[('D1_size', 'ikj')]:.2f} (ikj), {fits[('D1_size', 'unroll4')]:.2f} (unroll4) and {fits[('D1_size', 'blocked')]:.2f} (blocked, up to 4096), against 3 for the pure operation count.
The baseline is not monotone in n: the sizes that are multiples of 512 (512, 1024, 1536, 2048) cost 5.9&ndash;6.8&nbsp;ns per n<sup>3</sup>, about twice the 1.9 and 3.1&nbsp;ns of n&nbsp;=&nbsp;384 and 768, consistent with a large power-of-two-related column stride in B (Assignment 1 reported a similar effect).</p>
{fig('fig_dense_gflops', 'Figure 3. Effective throughput 2n<sup>3</sup>/median time. The ijk baseline stays near 0.3&nbsp;GFLOP/s for n&nbsp;&ge;&nbsp;1024; the optimised methods reach 3&ndash;8&nbsp;GFLOP/s.', '70%')}
<h3>4.2 Tile-size sweep (blocked method)</h3>
{table('table_block.html')}
{fig('fig_block_sweep', 'Figure 4. Tile-size sweep of the blocked kernel (dashed lines: unblocked ikj in the main experiment D1 at the same n).', '70%')}
<p>Tiles of 32 (best at n&nbsp;=&nbsp;2048 and 4096) to 64 (best at 1024 within 0.5&nbsp;% of 512) elements are best, and tiles of 8&ndash;16 are 25&ndash;70&nbsp;% slower. The main experiments use b&nbsp;=&nbsp;32, chosen from this sweep, which was run on the same machine and sizes as the evaluation, so
the blocked results are mildly tuned in its favour. The sweep was run in an earlier session than D1: the same configuration (n&nbsp;=&nbsp;1024, b&nbsp;=&nbsp;32) took 354&nbsp;ms there and 456&nbsp;ms in D1, a 29&nbsp;% drift between sessions on this laptop (thermal state, turbo clock, background load),
which bounds how finely cross-session numbers can be compared. Comparisons inside D1 (same session) are more reliable.</p>
<h3>4.3 Part B: sparse matrices</h3>
{table('table_sparsity.html')}
{fig('fig_sparsity_sweep', 'Figure 5. n&nbsp;=&nbsp;1024: time per multiplication versus sparsity of A and B (uniform positions). The dense methods do not depend on the number of zeros; the CSR kernel does; the dashed line adds the dense&rarr;CSR conversion (A and B).', '72%')}
{fig('fig_fill_in', 'Figure 6. Density of the product versus density of the inputs (n&nbsp;=&nbsp;1024). Uniform random inputs follow 1&minus;(1&minus;d<sup>2</sup>)<sup>n</sup>; banded inputs stay sparse.', '66%')}
{table('table_size_z99.html')}
{table('table_size_z99.9.html')}
{table('table_band.html')}
{fig('fig_sparse_size', 'Figure 7. Size scaling at fixed sparsity (uniform positions). The dense methods exist up to n&nbsp;=&nbsp;4096 (ijk up to 2048); CSR continues to n&nbsp;=&nbsp;8192 (99&nbsp;%) and 32768 (99.9&nbsp;%).')}
{table('table_band_size.html')}
{fig('fig_banded', 'Figure 8. Banded inputs: left, effect of the bandwidth at n&nbsp;=&nbsp;1024; right, scaling with n at w&nbsp;=&nbsp;5 (the dotted line is linear).')}
<h3>4.4 Memory and limits</h3>
{table('table_memory.html')}
{fig('fig_memory', 'Figure 9. Peak working set of the benchmark processes versus n, dense and CSR runs, with the analytic size of three dense matrices.', '70%')}
<p><b>Limits.</b> Dense: ijk completed up to n&nbsp;=&nbsp;2048 (58&nbsp;s per product) and timed out at 3072; ikj, unroll4 and blocked completed every size generated, up to 4096 (24&ndash;41&nbsp;s), so 4096 is the largest <em>tested</em> size, not a limit of the methods
(dense inputs were not generated above n&nbsp;=&nbsp;4096; 8192 would need 1.6&nbsp;GB for the arrays and about 3&nbsp;min per product). CSR: largest feasible size 8192 at 99&nbsp;% zeros (the product is then 56&nbsp;% dense and the run needs 455&nbsp;MiB; n&nbsp;=&nbsp;16384 was not attempted
because the product would be almost dense, &asymp;3&nbsp;GB of CSR output plus the growth copy, above the 4&nbsp;GB budget), and 32768 at 99.9&nbsp;% and for the banded matrices (largest attempted; no failure occurred). Three dense matrices at n&nbsp;=&nbsp;32768 would need 25.8&nbsp;GB, more than the 13.8&nbsp;GB of RAM.
Non-OK rows (timeouts and not-attempted entries, with reasons) are listed in <code>results/non_ok_runs.csv</code>.</p>

<h2>5. Discussion</h2>
<p><b>Part A.</b> <i>H1 is supported:</i> loop reordering alone gives {k['speedup_ikj_n1024']:.1f}&times; (n&nbsp;=&nbsp;1024) and {k['speedup_ikj_n2048']:.1f}&times; (n&nbsp;=&nbsp;2048), and 1.9&ndash;16&times; for n&nbsp;&le;&nbsp;768, raising the throughput from 0.3 to 3.4&ndash;5.3&nbsp;GFLOP/s.
The change removes the stride-n access to B; this is the most plausible cause, in line with the literature [5, 8] and with the power-of-two irregularities of ijk, but we measured no cache counters, so it is an interpretation of the timings, not a direct observation.
<i>H2 is supported:</i> unrolling adds a further {k['speedup_unroll4_n1024'] / k['speedup_ikj_n1024']:.2f}&times; (n&nbsp;=&nbsp;1024) to {k['speedup_unroll4_n2048'] / k['speedup_ikj_n2048']:.2f}&times; (2048) over ikj, so it is the fastest method at every tested n (1.74&times; at 4096: 23.4&nbsp;s versus 40.7&nbsp;s).
Per four multiply-adds ikj performs 8 loads and 4 stores of data, the unrolled loop 5 loads and 1 store; the observed gain is consistent with fewer memory operations in a scalar kernel, although we did not isolate this.
<i>H3 is partly supported:</i> blocking was slower than ikj for n&nbsp;&le;&nbsp;1024 (456 versus 404&nbsp;ms at 1024; the extra loop overhead is not repaid while B, 8&nbsp;MB at n&nbsp;=&nbsp;1024, still fits in the 16&nbsp;MB L3) and faster once B exceeds it:
1.31&times; (n&nbsp;=&nbsp;1536, B&nbsp;=&nbsp;18&nbsp;MB), 1.39&times; (2048) and 1.43&times; (4096). This coincidence between the cache size and the onset of the benefit supports the capacity explanation. However, blocked did <em>not</em> beat unroll4 at any size (3.54 versus 2.72&nbsp;s at 2048),
because our tile kernel is not unrolled. Combining tiling with unrolling or register blocking is the obvious next step and was not tested, so we cannot say which of the two is better in a combined kernel. We did not test Strassen. The gains concern this machine&rsquo;s cache sizes and a scalar-code compiler setting; with vectorised code the balance between
loads and arithmetic, and thus the benefit of each technique, would change.</p>
<p><b>Part B.</b> <i>H4 is refuted in its numbers, supported in its direction.</i> The CSR product is faster than the best dense method already at 50&nbsp;% zeros (1.07&times;; 1.02&times; including conversion), not only at 90&nbsp;%: the crossover lies between 0&nbsp;% (CSR 3.5&times; <em>slower</em> than unroll4, 0.94&nbsp;s versus 0.27&nbsp;s) and 50&nbsp;% zeros; we did not test values in between.
Against the ijk baseline CSR is faster at every tested sparsity, even at 0&nbsp;% zeros (6.7&times;), but that is the access-pattern effect of Part A, not a sparsity effect, which is why the comparison with the best dense method is the meaningful one. Above 90&nbsp;% zeros the advantage grows quickly
(10&times; at 90&nbsp;%, 46&times; at 99&nbsp;%, 248&times; at 99.5&nbsp;%). The product of random matrices is fully dense up to 90&nbsp;% zeros (Figure 6, observed densities match 1&minus;(1&minus;d<sup>2</sup>)<sup>n</sup> closely), and still 33&nbsp;% dense at 98&nbsp;%, so the cost
of the sparse product is dominated by writing and sorting output entries long before the inputs are sparse enough to make the result sparse.</p>
<p><i>Conversion.</i> Dense&rarr;CSR conversion scans n&sup2; entries and so costs about 2.9&ndash;3&nbsp;ms at n&nbsp;=&nbsp;1024 (44&nbsp;ms at 4096) regardless of sparsity. It is negligible while the kernel takes tens of milliseconds, but at 99.9&nbsp;% zeros the kernel needs 23&nbsp;&micro;s and the conversion is 99&nbsp;% of the total: the speed-up over the best dense
method falls from 11,730&times; (kernel only) to 85&times; (with conversion). Sparse multiplication therefore pays at very high sparsity only if the data are kept in sparse form (as in the CSR-only scaling runs). <i>Structure.</i> At the same density the input pattern matters (H5 supported): 1&nbsp;% uniform (10 non-zeros per row) takes 5.5&nbsp;ms, banded w&nbsp;=&nbsp;5 (11 per row, density 1.07&nbsp;%) 0.37&nbsp;ms, 15&times; less,
because the uniform product fills in to 9.3&nbsp;% (98,000 entries) against 2.0&nbsp;% (21,000) and its accesses are scattered, while banded rows are contiguous. The same counts explain scaling: banded CSR time grows linearly with n (slope {k['csr_slope_band_w5']:.2f}; 12&nbsp;ms at n&nbsp;=&nbsp;32768), uniform 99&nbsp;% with slope {k['csr_slope_z99']:.2f} (256&ndash;8192)
and uniform 99.9&nbsp;% with slope {k['csr_slope_z99.9']:.2f} (1024&ndash;32768; the number of non-zeros per row grows proportionally to n, so the work is again of order n&sup3;, with a much smaller constant than the dense one). At 99.9&nbsp;% and n&nbsp;=&nbsp;32768 the sparse product takes 2.9&nbsp;s and
694&nbsp;MiB; the dense matrices would not fit in memory.</p>
<p><i>Memory.</i> The dense processes use the analytic 3&times;8n&sup2; bytes plus a fixed &asymp;8&nbsp;MiB (runtime and I/O buffers; the n&nbsp;=&nbsp;64 process peaks at {k['process_baseline_kib_dense'] / 1024:.1f}&nbsp;MiB) and do not depend on the method or the sparsity. CSR processes need memory in proportion to the non-zeros of A, B and the product: 8&ndash;11&nbsp;MiB for n&nbsp;=&nbsp;1024 at 99&ndash;99.9&nbsp;% zeros against 32&nbsp;MiB dense, and the output dominates when the product fills in
(455&nbsp;MiB at n&nbsp;=&nbsp;8192, 99&nbsp;%, 37.7&nbsp;M output entries &times; 12&nbsp;B). Peak working set is a process-level metric and is not the same as the live matrix size.</p>
<p><b>Limitations.</b> (i) One laptop, one compiler setting (<code>-O2</code>, scalar code), one run order; between-session drift of about 30&nbsp;% was observed, so the tile sweep and D1 are not directly comparable. (ii) Tiles were tuned on the evaluation machine. (iii) Only 3 repetitions at n&nbsp;&gt;&nbsp;2048 because of the run time. (iv) The sparse and dense timing boundaries differ slightly (output allocation is inside the sparse timer). (v) Dense inputs above n&nbsp;=&nbsp;4096 were not generated, so
the sparse/dense comparison is limited to n&nbsp;&le;&nbsp;4096, and dense runs for the 99.9&nbsp;% and banded size sweeps use the blocked kernel instead of the fastest dense kernel (unroll4); this understates the dense speed by about 20&nbsp;% there. (vi) Cache and TLB explanations are hypotheses consistent with the data, not counter measurements. (vii) The crossover between 0 and 50&nbsp;% zeros was not located more finely; the product&rsquo;s sorting step and the int32 indices are design choices of our CSR kernel,
and a library implementation (e.g.&nbsp;SciPy or MKL) could behave differently. (viii) Results for uniform and banded patterns do not cover other structures (e.g.&nbsp;power-law row lengths, block sparsity).</p>

<h2>6. Conclusions</h2>
<ol>
<li>Within the tested scope (Ryzen 7 7730U, gcc 16.1 <code>-O2</code>, scalar single-thread code, binary64, kernel-only timing), three optimised dense kernels are correct and bit-identical to the baseline.</li>
<li>Loop reordering gives 8.5&ndash;16&times; for n&nbsp;&ge;&nbsp;768; unrolling by 4 adds 1.5&ndash;1.8&times; (best overall: up to 24&times; over ijk); blocking helps ikj by 1.3&ndash;1.4&times; only when B exceeds the L3 cache, and was not better than the unrolled kernel.</li>
<li>CSR multiplication beat the best dense method from 50&nbsp;% zeros at n&nbsp;=&nbsp;1024 (not tested between 0 and 50&nbsp;%), by 10&times; at 90&nbsp;% and 46&times; at 99&nbsp;%; conversion and output fill-in reduce the advantage, and the input structure (random versus banded) changes the cost by more than an order of magnitude at equal density.</li>
<li>Sparse storage extends the feasible sizes: 32768 at 99.9&nbsp;% zeros in 694&nbsp;MiB where dense is infeasible in RAM. The largest tested feasible dense size was 4096 (2048 for ijk).</li></ol>
<p>Unresolved: the best combination of tiling and unrolling, the exact sparsity crossover, and hardware-level confirmation of the cache explanations.</p>

<h2>AI-use disclosure</h2>
<p>In line with the course policy, the AI assistant Claude (Anthropic) was used to plan the experimental design, draft and debug the C kernels, the benchmark and analysis scripts and the tests, and to draft and revise the English of this report. All numbers in the report were produced by running the code in the repository
(tables and figures are generated by <code>scripts/analyze.py</code>); no result was generated by the AI tool. The author is responsible for understanding the code, checking the measurements and references, and defending the conclusions.</p>

<h2>Reproducibility materials</h2>
<p>Folder <code>Assignment02_AlejandroSantana_Materials</code> (and the repository {REPO_URL}): <code>src/c/matmul2.c</code>, <code>README.md</code> with exact build and run commands, <code>config/benchmark_config.json</code>, input generators and manifest (<code>data/inputs/manifest.json</code>), raw samples and run logs (<code>results/</code>),
validation output, environment description, and the scripts that produce all tables and figures. <code>run_all.ps1</code> rebuilds, validates, measures, analyses and regenerates this report. Reused from Assignment 1: the ijk kernel and the dense input generator (same seeds and hashes).</p>

<h2>References</h2>
<ol class="ref">
<li>G. H. Golub and C. F. Van Loan, <i>Matrix Computations</i>, 4th ed. Baltimore, MD, USA: Johns Hopkins Univ. Press, 2013.</li>
<li>M. S. Lam, E. E. Rothberg, and M. E. Wolf, &ldquo;The cache performance and optimizations of blocked algorithms,&rdquo; in <i>Proc. 4th Int. Conf. Architectural Support for Programming Languages and Operating Systems (ASPLOS-IV)</i>, 1991, pp. 63&ndash;74.</li>
<li>F. G. Gustavson, &ldquo;Two fast algorithms for sparse matrices: Multiplication and permuted transposition,&rdquo; <i>ACM Trans. Math. Softw.</i>, vol. 4, no. 3, pp. 250&ndash;269, 1978.</li>
<li>Y. Saad, <i>Iterative Methods for Sparse Linear Systems</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2003.</li>
<li>U. Drepper, &ldquo;What every programmer should know about memory,&rdquo; Red Hat, Inc., Tech. Rep., Nov. 2007.</li>
<li>K. Goto and R. A. van de Geijn, &ldquo;Anatomy of high-performance matrix multiplication,&rdquo; <i>ACM Trans. Math. Softw.</i>, vol. 34, no. 3, Art. 12, 2008.</li>
<li>V. Strassen, &ldquo;Gaussian elimination is not optimal,&rdquo; <i>Numer. Math.</i>, vol. 13, no. 4, pp. 354&ndash;356, 1969.</li>
<li>J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.</li>
<li>N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.</li>
<li>Microsoft, &ldquo;PROCESS_MEMORY_COUNTERS structure (psapi.h)&rdquo; and &ldquo;GetProcessMemoryInfo function,&rdquo; Microsoft Learn, Windows API documentation.</li>
<li>T. A. Davis, <i>Direct Methods for Sparse Linear Systems</i>. Philadelphia, PA, USA: SIAM, 2006.</li></ol>
</body></html>"""
    # tile size placeholder in Table 1
    block = json.load(open(os.path.join(ROOT, "config", "benchmark_config.json")))["block_size"]
    html = html.replace("b = {}", f"b = {block}")
    open(OUT_HTML, "w", encoding="utf8").write(html)
    print("wrote", OUT_HTML)
    if args.no_pdf:
        return
    browsers = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
    exe = next((b for b in browsers if os.path.exists(b)), shutil.which("chrome") or shutil.which("msedge"))
    if not exe:
        sys.exit("No Chrome/Edge found: open the HTML file and print it to PDF manually.")
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                    "file:///" + OUT_HTML.replace("\\", "/")], check=True, timeout=180)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
