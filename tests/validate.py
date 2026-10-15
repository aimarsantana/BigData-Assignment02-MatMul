#!/usr/bin/env python3
"""Correctness tests for the dense and sparse kernels of Assignment 2 (run BEFORE benchmarking).

Reference: NumPy matmul (OpenBLAS), used only for checking, never as a student implementation.
Tolerance (float64): |C - C_ref| <= atol + rtol*|C_ref| with atol = rtol = 1e-9.
  Justification: for entries in [-1, 1] the floating-point error of a length-n dot product is bounded
  by n*u*sum|a_ik b_kj| <= n^2*u (u = 2^-53 = 1.1e-16) -> 1.2e-10 for n = 1024. 1e-9 is above this
  bound yet far below any plausible logic error (products are O(0.1 - 1)).
Additionally every optimised kernel is compared BIT-FOR-BIT with the baseline ijk, because all
kernels add the products of C[i][j] in increasing k without FMA (see matmul2.c): the expected
difference is exactly 0.

Output: CSV on stdout (test, method, n, block, max_abs_err_vs_numpy, bitwise_equal_to_ijk, result).
Exit code 1 if any test fails.

Usage: python tests/validate.py [--data-dir DIR]   (the data dir enables tests on benchmark inputs)
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE = os.path.join(ROOT, "build", "matmul2.exe")
ATOL = RTOL = 1e-9
FAILED = []


def run(args):
    r = subprocess.run([EXE] + [str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args}: {r.stderr}")
    return r.stdout


def write_csr(path, d):
    n = d.shape[0]
    rows, cols = np.nonzero(d)
    rowptr = np.zeros(n + 1, dtype="<i4")
    np.add.at(rowptr, rows + 1, 1)
    rowptr = np.cumsum(rowptr).astype("<i4")
    with open(path, "wb") as f:
        f.write(b"CSR1" + np.int32(n).tobytes() + np.int64(len(cols)).tobytes())
        f.write(rowptr.tobytes() + cols.astype("<i4").tobytes() + d[rows, cols].astype("<f8").tobytes())


def read_csr(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"CSR1"
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        nnz = int(np.frombuffer(f.read(8), "<i8")[0])
        rp = np.frombuffer(f.read(4 * (n + 1)), "<i4")
        ci = np.frombuffer(f.read(4 * nnz), "<i4")
        v = np.frombuffer(f.read(8 * nnz), "<f8")
    d = np.zeros((n, n))
    d[np.repeat(np.arange(n), np.diff(rp)), ci] = v
    assert all(np.all(np.diff(ci[rp[i]:rp[i + 1]]) > 0) for i in range(n)), "column indices not sorted/unique"
    return d, nnz


def report(test, method, n, block, A, B, C, C_ijk):
    ref = A @ B
    err = float(np.max(np.abs(C - ref))) if C.size else 0.0
    ok = bool(np.allclose(C, ref, rtol=RTOL, atol=ATOL))
    bit = "" if C_ijk is None else str(bool(np.array_equal(C, C_ijk)))
    ok = ok and (C_ijk is None or bit == "True")
    if not ok:
        FAILED.append((test, method, n))
    print(f"{test},{method},{n},{block},{err:.3e},{bit},{'PASS' if ok else 'FAIL'}")


def dense_case(tmp, test, A, B, blocks=(64,)):
    n = A.shape[0]
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    A.astype("<f8").tofile(pa)
    B.astype("<f8").tofile(pb)
    outs = {}
    for method in ("ijk", "ikj", "unroll4", "blocked"):
        for b in (blocks if method == "blocked" else (64,)):
            po = os.path.join(tmp, f"C_{method}_{b}.bin")
            run(["dense", method, pa, pb, n, 0, 1, 1, b, po])
            outs[(method, b)] = np.fromfile(po, "<f8").reshape(n, n)
    base = outs[("ijk", 64)]
    for (method, b), C in outs.items():
        report(test, method, n, b if method == "blocked" else 0, A, B, C, None if method == "ijk" else base)


def sparse_case(tmp, test, A, B):
    n = A.shape[0]
    pa, pb, po = (os.path.join(tmp, x) for x in ("A.csr", "B.csr", "C.csr"))
    write_csr(pa, A)
    write_csr(pb, B)
    run(["csr", pa, pb, 0, 1, 1, po])
    C, nnz = read_csr(po)
    # dense baseline on the same logical matrices
    pda, pdb, pdo = (os.path.join(tmp, x) for x in ("A.bin", "B.bin", "C.bin"))
    A.astype("<f8").tofile(pda)
    B.astype("<f8").tofile(pdb)
    run(["dense", "ijk", pda, pdb, n, 0, 1, 1, 64, pdo])
    C_ijk = np.fromfile(pdo, "<f8").reshape(n, n)
    report(test, "csr", n, 0, A, B, C, C_ijk)
    # the structural nnz of C must equal the number of structurally reachable positions
    reach = ((A != 0).astype(np.int64) @ (B != 0).astype(np.int64)) > 0
    if nnz != int(reach.sum()):
        FAILED.append((test + "-nnz", "csr", n))
        print(f"{test}-nnz,csr,{n},0,,,FAIL (nnz {nnz} != {int(reach.sum())})")
    # conversion round trip: dense -> CSR -> same entries
    out = run(["convert", pda, pdb, n, 0, 1])
    nA = int([l for l in out.splitlines() if l.startswith("SAMPLE")][0].split(",")[8])
    if nA != int(np.count_nonzero(A)):
        FAILED.append((test + "-convert", "convert", n))
        print(f"{test}-convert,convert,{n},0,,,FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(424242)
    print("test,method,n,block,max_abs_err_vs_numpy,bitwise_equal_to_ijk,result")
    with tempfile.TemporaryDirectory() as tmp:
        # --- hand-computed 3x3 example (dense and sparse)
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], float)
        assert np.array_equal(A @ B, [[30, 24, 18], [84, 69, 54], [138, 114, 90]])
        dense_case(tmp, "known_3x3", A, B)
        sparse_case(tmp, "known_3x3", A, B)
        # --- identity, zero
        X = rng.uniform(-1, 1, (37, 37))
        dense_case(tmp, "identity_right", X, np.eye(37))
        dense_case(tmp, "zero", X, np.zeros((37, 37)))
        sparse_case(tmp, "identity_right", X, np.eye(37))
        sparse_case(tmp, "zero_matrix", np.zeros((37, 37)), X)
        sparse_case(tmp, "both_zero", np.zeros((5, 5)), np.zeros((5, 5)))
        # --- sizes that are not multiples of 4 or of the tile
        for n in (1, 2, 5, 33, 100, 127, 130):
            A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
            dense_case(tmp, f"random_n{n}", A, B, blocks=(1, 7, 32, 64, 500))
        # --- sparse patterns with empty rows/columns, cancellation-free random positions
        for n, dens in ((50, 0.05), (200, 0.02), (300, 0.3), (64, 0.0), (64, 1.0)):
            A = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < dens)
            B = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < dens)
            A[3, :] = 0.0                                  # force an empty row
            B[:, 5] = 0.0                                  # force an empty column
            sparse_case(tmp, f"sparse_random_n{n}_d{dens}", A, B)
        # --- benchmark inputs (if available)
        if args.data_dir:
            for n in (512, 1024):
                A = np.fromfile(os.path.join(args.data_dir, f"A_n{n}.bin"), "<f8").reshape(n, n)
                B = np.fromfile(os.path.join(args.data_dir, f"B_n{n}.bin"), "<f8").reshape(n, n)
                dense_case(tmp, f"benchmark_dense_n{n}", A, B, blocks=(64,))
            for tag in ("n1024_uni_z99", "n1024_uni_z90", "n1024_uni_z99.9", "n1024_band_w5", "n1024_band_w0"):
                A = read_csr(os.path.join(args.data_dir, f"A_{tag}.csr"))[0]
                B = read_csr(os.path.join(args.data_dir, f"B_{tag}.csr"))[0]
                same = (np.array_equal(A, np.fromfile(os.path.join(args.data_dir, f"A_{tag}.bin"), "<f8").reshape(A.shape)) and
                        np.array_equal(B, np.fromfile(os.path.join(args.data_dir, f"B_{tag}.bin"), "<f8").reshape(B.shape)))
                print(f"representations_match_{tag},csr-vs-dense-file,{A.shape[0]},0,,,{'PASS' if same else 'FAIL'}")
                if not same:
                    FAILED.append((tag, "repr", A.shape[0]))
                sparse_case(tmp, f"benchmark_{tag}", A, B)
    print(f"# {len(FAILED)} failure(s)", file=sys.stderr)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
