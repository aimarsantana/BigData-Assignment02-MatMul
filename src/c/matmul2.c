/*
 * Assignment 2 - optimised dense multiplication and sparse (CSR) multiplication, C, float64.
 *
 * Dense methods (all sequential, same storage: contiguous row-major block + row pointers):
 *   ijk      baseline, REUSED UNCHANGED from Assignment 1 (triple loop, scalar accumulator)
 *   ikj      loop reordering: the inner loop walks rows of B and C with unit stride
 *   unroll4  ikj with the k loop unrolled by 4: four rows of B are folded into C[i][*] per pass,
 *            so C[i][*] is loaded/stored once per four multiply-adds instead of once per one
 *   blocked  cache blocking (tiling) of the ikj kernel with square tiles of side `block`
 * Sparse method:
 *   csr      Gustavson's row-wise sparse x sparse product on CSR operands, dense accumulator
 *            (SPA), own code; no library is used.
 *   convert  dense -> CSR conversion of both operands (timed separately from the product)
 *
 * Summation order: for every C[i][j] all three dense kernels and the CSR kernel add the products
 * A[i][k]*B[k][j] in increasing k, starting from +0.0, with no FMA contraction (gcc -O2, generic
 * x86-64 target, no -ffast-math). The results are therefore expected to be bit-identical.
 *
 * Usage:
 *   matmul2 dense   <ijk|ikj|unroll4|blocked> A.bin B.bin n warmup reps batch block [C_out.bin]
 *   matmul2 csr     A.csr B.csr warmup reps batch [C_out.csr]
 *   (batch = multiplications per timed sample; batch 0 = automatic, see auto_batch)
 *   matmul2 convert A.bin B.bin n warmup reps
 *
 * Timer: QueryPerformanceCounter (monotonic). Memory: peak working set of the process
 * (PeakWorkingSetSize) and peak commit charge (PeakPagefileUsage), printed as MEM lines.
 * Timed region:
 *   dense   : the kernel, including zeroing of C for ikj/blocked; C is allocated outside the timer
 *   csr     : the kernel including allocation/growth of the output (its size is data dependent)
 *   convert : conversion of A and B, including allocation of the CSR arrays
 * Files: dense = raw little-endian float64 row-major; CSR = "CSR1", int32 n, int64 nnz,
 *        int32 rowptr[n+1], int32 col[nnz], float64 val[nnz].
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- utilities */

static long long now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    long long sec = c.QuadPart / freq.QuadPart, rem = c.QuadPart % freq.QuadPart;
    return sec * 1000000000LL + rem * 1000000000LL / freq.QuadPart;
}

static void die(const char *msg, int code) {
    fprintf(stderr, "%s\n", msg);
    exit(code);
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) die("allocation failed", 3);
    return p;
}

static void *xrealloc(void *old, size_t bytes) {
    void *p = realloc(old, bytes ? bytes : 1);
    if (!p) die("reallocation failed", 3);
    return p;
}

static void print_memory(const char *method, long long n) {
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof pmc);
    pmc.cb = sizeof pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        printf("MEM,%s,%lld,peak_working_set_kib,%llu\n", method, n,
               (unsigned long long)(pmc.PeakWorkingSetSize / 1024));
        printf("MEM,%s,%lld,peak_commit_kib,%llu\n", method, n,
               (unsigned long long)(pmc.PeakPagefileUsage / 1024));
    } else {
        printf("MEM,%s,%lld,peak_working_set_kib,-1\n", method, n);
    }
}

/* Pin the process to one logical processor (the 3rd, i.e. the first SMT thread of core 1) to avoid
 * migrations between cores; every kernel here is single threaded. */
static void pin_to_one_cpu(void) {
    DWORD_PTR mask = (DWORD_PTR)1 << 2, proc, sys;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) && (proc & mask)) {
        SetProcessAffinityMask(GetCurrentProcess(), mask);
        printf("INFO,all,affinity,logical processor 2\n");
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        printf("INFO,all,priority,high\n");
    } else {
        printf("INFO,all,affinity,not pinned\n");
    }
}

static void print_build_info(void) {
#ifdef __VERSION__
    printf("INFO,all,compiler,gcc %s\n", __VERSION__);
#endif
#ifdef CFLAGS_STR
    printf("INFO,all,cflags,%s\n", CFLAGS_STR);
#endif
}

/* batch == 0 asks for an automatic batch: the first warm-up run is executed with batch 1 and its
 * duration is used to choose the number of multiplications per sample so that one sample lasts
 * about 20 ms (at most 10000, at least 1). The batch is printed in every SAMPLE line. */
static int auto_batch(long long ns_single) {
    long long b = ns_single > 0 ? 20000000LL / ns_single : 10000;
    return b < 1 ? 1 : (b > 10000 ? 10000 : (int)b);
}

/* ------------------------------------------------------------ dense matrices */

static double **alloc_matrix(int n) {
    double *block = calloc((size_t)n * n, sizeof(double));
    double **rows = malloc((size_t)n * sizeof(double *));
    if (!block || !rows) die("allocation failed (dense matrix)", 3);
    for (int i = 0; i < n; i++) rows[i] = block + (size_t)i * n;
    return rows;
}

static void free_matrix(double **m) {
    free(m[0]);
    free(m);
}

static double **read_matrix(const char *path, int n) {
    double **m = alloc_matrix(n);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(4); }
    size_t got = fread(m[0], sizeof(double), (size_t)n * n, f);
    fclose(f);
    if (got != (size_t)n * n) die("short read on dense input", 5);
    return m;
}

static void write_matrix(const char *path, double **m, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(6); }
    fwrite(m[0], sizeof(double), (size_t)n * n, f);
    fclose(f);
}

/* Baseline (Assignment 1): straightforward i-j-k triple loop, scalar accumulator. */
static void matmul_ijk(double **A, double **B, double **C, int n, int block) {
    (void)block;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
}

/* Optimisation 1 - loop reordering (i-k-j). The inner loop reads B[k][*] and updates C[i][*]
 * with unit stride, instead of striding down a column of B. */
static void matmul_ikj(double **A, double **B, double **C, int n, int block) {
    (void)block;
    memset(C[0], 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *Ci = C[i];
        for (int k = 0; k < n; k++) {
            double a = A[i][k];
            const double *Bk = B[k];
            for (int j = 0; j < n; j++) {
                Ci[j] += a * Bk[j];
            }
        }
    }
}

/* Optimisation 2 - loop unrolling (k by 4) on top of the ikj order. The four products are added to
 * the running value of C[i][j] one after the other, in increasing k, so every C[i][j] keeps exactly
 * the same summation order as in the baseline. */
static void matmul_unroll4(double **A, double **B, double **C, int n, int block) {
    (void)block;
    memset(C[0], 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *Ci = C[i];
        const double *Ai = A[i];
        int k = 0;
        for (; k + 3 < n; k += 4) {
            double a0 = Ai[k], a1 = Ai[k + 1], a2 = Ai[k + 2], a3 = Ai[k + 3];
            const double *B0 = B[k], *B1 = B[k + 1], *B2 = B[k + 2], *B3 = B[k + 3];
            for (int j = 0; j < n; j++) {
                double c = Ci[j];
                c += a0 * B0[j];
                c += a1 * B1[j];
                c += a2 * B2[j];
                c += a3 * B3[j];
                Ci[j] = c;
            }
        }
        for (; k < n; k++) {
            double a = Ai[k];
            const double *Bk = B[k];
            for (int j = 0; j < n; j++) Ci[j] += a * Bk[j];
        }
    }
}

/* Optimisation 3 - cache blocking (tiling) of the ikj kernel. Tiles of side `bs` are processed
 * in the order ii, kk, jj so that, for a fixed (ii, kk) pair, a bs x bs tile of B is re-used from
 * cache for the bs rows of A/C in the block. kk blocks are visited in increasing order, so the
 * summation order for each C[i][j] is still increasing k. */
static void matmul_blocked(double **A, double **B, double **C, int n, int bs) {
    memset(C[0], 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs) {
        int imax = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int kmax = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++) {
                    double *Ci = C[i];
                    for (int k = kk; k < kmax; k++) {
                        double a = A[i][k];
                        const double *Bk = B[k];
                        for (int j = jj; j < jmax; j++) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}

/* Row-by-row sum of all entries, same summation order for dense and CSR results. */
static double checksum_dense(double **C, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double rs = 0.0;
        for (int j = 0; j < n; j++) rs += C[i][j];
        total += rs;
    }
    return total;
}

/* ------------------------------------------------------------------ CSR */

typedef struct {
    int n;
    int64_t nnz;
    int32_t *rp;  /* n+1 row pointers */
    int32_t *ci;  /* nnz column indices, increasing inside each row */
    double *v;    /* nnz values */
} csr_t;

static void csr_free(csr_t *m) {
    free(m->rp);
    free(m->ci);
    free(m->v);
    m->rp = NULL; m->ci = NULL; m->v = NULL;
}

static csr_t csr_read(const char *path) {
    csr_t m;
    char magic[4];
    int32_t n;
    int64_t nnz;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(4); }
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CSR1", 4) != 0) die("bad CSR magic", 5);
    if (fread(&n, 4, 1, f) != 1 || fread(&nnz, 8, 1, f) != 1) die("bad CSR header", 5);
    if (nnz > INT32_MAX) die("nnz exceeds int32 range", 5);
    m.n = n; m.nnz = nnz;
    m.rp = xmalloc((size_t)(n + 1) * 4);
    m.ci = xmalloc((size_t)nnz * 4);
    m.v = xmalloc((size_t)nnz * 8);
    if (fread(m.rp, 4, (size_t)n + 1, f) != (size_t)n + 1 ||
        fread(m.ci, 4, (size_t)nnz, f) != (size_t)nnz ||
        fread(m.v, 8, (size_t)nnz, f) != (size_t)nnz) die("short read on CSR input", 5);
    fclose(f);
    return m;
}

static void csr_write(const char *path, const csr_t *m) {
    FILE *f = fopen(path, "wb");
    int32_t n = m->n;
    int64_t nnz = m->nnz;
    if (!f) { perror(path); exit(6); }
    fwrite("CSR1", 1, 4, f);
    fwrite(&n, 4, 1, f);
    fwrite(&nnz, 8, 1, f);
    fwrite(m->rp, 4, (size_t)n + 1, f);
    fwrite(m->ci, 4, (size_t)nnz, f);
    fwrite(m->v, 8, (size_t)nnz, f);
    fclose(f);
}

/* Dense -> CSR: structural non-zeros are the entries that differ from 0.0. */
static csr_t csr_from_dense(double **D, int n) {
    csr_t m;
    int64_t nnz = 0;
    m.n = n;
    m.rp = xmalloc((size_t)(n + 1) * 4);
    for (int i = 0; i < n; i++) {
        int64_t c = 0;
        const double *row = D[i];
        for (int j = 0; j < n; j++) c += (row[j] != 0.0);
        m.rp[i] = (int32_t)nnz;
        nnz += c;
    }
    if (nnz > INT32_MAX) die("nnz exceeds int32 range", 5);
    m.rp[n] = (int32_t)nnz;
    m.nnz = nnz;
    m.ci = xmalloc((size_t)nnz * 4);
    m.v = xmalloc((size_t)nnz * 8);
    int64_t p = 0;
    for (int i = 0; i < n; i++) {
        const double *row = D[i];
        for (int j = 0; j < n; j++) {
            if (row[j] != 0.0) { m.ci[p] = j; m.v[p] = row[j]; p++; }
        }
    }
    return m;
}

static int cmp_i32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* Sparse x sparse product (Gustavson 1978): for each row i of A and each non-zero A[i][k],
 * the scaled row k of B is merged into a dense accumulator (SPA) indexed by column. The touched
 * columns are then sorted so that the output is a valid CSR matrix. The output arrays grow by
 * doubling; their allocation is therefore part of the timed region. */
static csr_t spgemm(const csr_t *A, const csr_t *B) {
    int n = A->n;
    csr_t C;
    double *acc = xmalloc((size_t)n * sizeof(double));
    int32_t *mark = xmalloc((size_t)n * sizeof(int32_t));
    int32_t *cols = xmalloc((size_t)n * sizeof(int32_t));
    int64_t cap = (A->nnz + B->nnz > 1024) ? A->nnz + B->nnz : 1024;
    int64_t nnz = 0;
    for (int j = 0; j < n; j++) mark[j] = -1;
    C.n = n;
    C.rp = xmalloc((size_t)(n + 1) * 4);
    C.ci = xmalloc((size_t)cap * 4);
    C.v = xmalloc((size_t)cap * 8);
    C.rp[0] = 0;
    for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int32_t p = A->rp[i]; p < A->rp[i + 1]; p++) {
            int32_t k = A->ci[p];
            double a = A->v[p];
            for (int32_t q = B->rp[k]; q < B->rp[k + 1]; q++) {
                int32_t j = B->ci[q];
                if (mark[j] != i) {
                    mark[j] = i;
                    acc[j] = a * B->v[q];
                    cols[cnt++] = j;
                } else {
                    acc[j] += a * B->v[q];
                }
            }
        }
        if (nnz + cnt > cap) {
            while (nnz + cnt > cap) cap *= 2;
            if (cap > INT32_MAX) die("output nnz exceeds int32 range", 7);
            C.ci = xrealloc(C.ci, (size_t)cap * 4);
            C.v = xrealloc(C.v, (size_t)cap * 8);
        }
        if (cnt > n / 8) {                    /* dense-ish row: scan the marks, already sorted */
            for (int j = 0; j < n; j++) {
                if (mark[j] == i) { C.ci[nnz] = j; C.v[nnz] = acc[j]; nnz++; }
            }
        } else {                               /* sparse row: sort the touched columns */
            qsort(cols, (size_t)cnt, sizeof(int32_t), cmp_i32);
            for (int t = 0; t < cnt; t++) { C.ci[nnz] = cols[t]; C.v[nnz] = acc[cols[t]]; nnz++; }
        }
        C.rp[i + 1] = (int32_t)nnz;
    }
    C.nnz = nnz;
    free(acc); free(mark); free(cols);
    return C;
}

static double checksum_csr(const csr_t *C) {
    double total = 0.0;
    for (int i = 0; i < C->n; i++) {
        double rs = 0.0;
        for (int32_t p = C->rp[i]; p < C->rp[i + 1]; p++) rs += C->v[p];
        total += rs;
    }
    return total;
}

/* --------------------------------------------------------------- drivers */

typedef void (*dense_kernel)(double **, double **, double **, int, int);

static int run_dense(int argc, char **argv) {
    if (argc < 10) die("dense: need method A.bin B.bin n warmup reps batch block [C_out.bin]", 2);
    const char *method = argv[2], *a_path = argv[3], *b_path = argv[4];
    int n = atoi(argv[5]), warmup = atoi(argv[6]), reps = atoi(argv[7]);
    int batch = atoi(argv[8]), block = atoi(argv[9]);
    const char *out_path = argc > 10 ? argv[10] : NULL;
    dense_kernel kernel;
    if (strcmp(method, "ijk") == 0) kernel = matmul_ijk;
    else if (strcmp(method, "ikj") == 0) kernel = matmul_ikj;
    else if (strcmp(method, "unroll4") == 0) kernel = matmul_unroll4;
    else if (strcmp(method, "blocked") == 0) kernel = matmul_blocked;
    else die("unknown dense method", 2);
    if (block < 1) die("block must be >= 1", 2);

    double **A = read_matrix(a_path, n);
    double **B = read_matrix(b_path, n);
    double **C = alloc_matrix(n); /* allocated outside the timer */
    printf("INFO,%s,block,%d\n", method, strcmp(method, "blocked") == 0 ? block : 0);
    int auto_mode = (batch == 0);
    if (auto_mode && warmup < 1) warmup = 1;
    int cur = auto_mode ? 1 : batch;
    for (int r = 0; r < warmup + reps; r++) {
        const char *phase = r < warmup ? "warmup" : "measured";
        int idx = r < warmup ? r : r - warmup;
        long long t0 = now_ns();
        for (int b = 0; b < cur; b++) kernel(A, B, C, n, block);
        long long t1 = now_ns();
        double cs = checksum_dense(C, n); /* consumes the result, outside the timer */
        printf("SAMPLE,%s,%d,%s,%d,%d,%.1f,%.17g,-1,-1,-1\n", method, n, phase, idx, cur,
               (double)(t1 - t0) / cur, cs);
        fflush(stdout);
        if (auto_mode && r == 0) cur = auto_batch(t1 - t0);
    }
    print_memory(method, n);
    if (out_path) write_matrix(out_path, C, n);
    free_matrix(A); free_matrix(B); free_matrix(C);
    return 0;
}

static int run_csr(int argc, char **argv) {
    if (argc < 7) die("csr: need A.csr B.csr warmup reps batch [C_out.csr]", 2);
    int warmup = atoi(argv[4]), reps = atoi(argv[5]), batch = atoi(argv[6]);
    const char *out_path = argc > 7 ? argv[7] : NULL;
    csr_t A = csr_read(argv[2]), B = csr_read(argv[3]);
    csr_t C = {0};
    if (A.n != B.n) die("operand sizes differ", 2);
    int n = A.n;
    int auto_mode = (batch == 0);
    if (auto_mode && warmup < 1) warmup = 1;
    int cur = auto_mode ? 1 : batch;
    for (int r = 0; r < warmup + reps; r++) {
        const char *phase = r < warmup ? "warmup" : "measured";
        int idx = r < warmup ? r : r - warmup;
        long long t0 = now_ns();
        for (int b = 0; b < cur; b++) {
            if (C.rp) csr_free(&C);  /* the previous product is released inside the batch */
            C = spgemm(&A, &B);
        }
        long long t1 = now_ns();
        double cs = checksum_csr(&C);
        printf("SAMPLE,csr,%d,%s,%d,%d,%.1f,%.17g,%lld,%lld,%lld\n", n, phase, idx, cur,
               (double)(t1 - t0) / cur, cs, (long long)A.nnz, (long long)B.nnz, (long long)C.nnz);
        fflush(stdout);
        if (auto_mode && r == 0) cur = auto_batch(t1 - t0);
    }
    print_memory("csr", n);
    if (out_path) csr_write(out_path, &C);
    csr_free(&A); csr_free(&B); csr_free(&C);
    return 0;
}

static int run_convert(int argc, char **argv) {
    if (argc < 7) die("convert: need A.bin B.bin n warmup reps", 2);
    int n = atoi(argv[4]), warmup = atoi(argv[5]), reps = atoi(argv[6]);
    double **A = read_matrix(argv[2], n);
    double **B = read_matrix(argv[3], n);
    for (int r = 0; r < warmup + reps; r++) {
        const char *phase = r < warmup ? "warmup" : "measured";
        int idx = r < warmup ? r : r - warmup;
        long long t0 = now_ns();
        csr_t sa = csr_from_dense(A, n);
        csr_t sb = csr_from_dense(B, n);
        long long t1 = now_ns();
        double cs = checksum_csr(&sa) + checksum_csr(&sb);
        printf("SAMPLE,convert,%d,%s,%d,1,%.1f,%.17g,%lld,%lld,-1\n", n, phase, idx,
               (double)(t1 - t0), cs, (long long)sa.nnz, (long long)sb.nnz);
        fflush(stdout);
        csr_free(&sa); csr_free(&sb);
    }
    print_memory("convert", n);
    free_matrix(A); free_matrix(B);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s dense|csr|convert ...\n", argv[0]);
        return 2;
    }
    print_build_info();
    pin_to_one_cpu();
    if (strcmp(argv[1], "dense") == 0) return run_dense(argc, argv);
    if (strcmp(argv[1], "csr") == 0) return run_csr(argc, argv);
    if (strcmp(argv[1], "convert") == 0) return run_convert(argc, argv);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
