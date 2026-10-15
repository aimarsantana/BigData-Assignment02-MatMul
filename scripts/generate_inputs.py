#!/usr/bin/env python3
"""Generate the shared input matrices of Assignment 2.

Dense inputs (Part A) are the SAME matrices as Assignment 1: NumPy PCG64, seeds 20261012 (A) and
20261013 (B), uniform in [-1, 1), float64. Their SHA-256 hashes can be compared with the manifest
of Assignment 1 (`--check-a1 <path to Assignment 1 manifest.json>`).

Sparse inputs (Part B) are generated with a documented process, written in two equivalent
representations of the SAME logical matrix:
  * dense   raw little-endian float64, row-major (A_<tag>.bin)       -> read by the dense methods
  * CSR     "CSR1", int32 n, int64 nnz, int32 rowptr[n+1], int32 col[nnz], float64 val[nnz]
                                                                      -> read by the sparse method
Patterns
  uni_z<Z>   "uniform": every row has exactly k = round(n * (1 - Z/100)) non-zeros at columns drawn
             uniformly without replacement (fixed nnz per row, random positions).
  band_w<W>  "banded": entry (i, j) is non-zero iff |i - j| <= W (2W+1 non-zeros per row, away from
             the borders).
Non-zero values: uniform in [-1, 1); an (essentially impossible) exact 0.0 draw is replaced by 0.5 so
that the stored non-zero count is exactly the intended one.
Seeds: positions of A / B -> 20261026 / 20261027, values of A / B -> 20261028 / 20261029
(per-pattern streams are derived with numpy SeedSequence([seed, n, pattern id])).
Sparsity Z = percentage of zero entries, density = 100 - Z.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED_A, SEED_B = 20261012, 20261013          # dense inputs (identical to Assignment 1)
POS_SEED = {"A": 20261026, "B": 20261027}
VAL_SEED = {"A": 20261028, "B": 20261029}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dense_matrix(n, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, n)).astype("<f8")


def write_csr(path, n, rowptr, col, val):
    with open(path, "wb") as f:
        f.write(b"CSR1")
        f.write(np.int32(n).tobytes())
        f.write(np.int64(len(col)).tobytes())
        f.write(rowptr.astype("<i4").tobytes())
        f.write(col.astype("<i4").tobytes())
        f.write(val.astype("<f8").tobytes())


def csr_uniform(n, zero_pct, name, pattern_id):
    k = int(round(n * (1.0 - zero_pct / 100.0)))
    k = max(1, min(n, k))
    pos = np.random.default_rng(np.random.SeedSequence([POS_SEED[name], n, pattern_id]))
    val = np.random.default_rng(np.random.SeedSequence([VAL_SEED[name], n, pattern_id]))
    col = np.empty(n * k, dtype=np.int64)
    for i in range(n):
        col[i * k:(i + 1) * k] = np.sort(pos.choice(n, size=k, replace=False))
    rowptr = np.arange(0, n * k + 1, k, dtype=np.int64)
    values = val.uniform(-1.0, 1.0, size=n * k)
    values[values == 0.0] = 0.5
    return rowptr, col, values


def csr_band(n, w, name, pattern_id):
    val = np.random.default_rng(np.random.SeedSequence([VAL_SEED[name], n, pattern_id]))
    cols, rowptr = [], [0]
    for i in range(n):
        lo, hi = max(0, i - w), min(n - 1, i + w)
        cols.append(np.arange(lo, hi + 1))
        rowptr.append(rowptr[-1] + hi - lo + 1)
    col = np.concatenate(cols)
    values = val.uniform(-1.0, 1.0, size=len(col))
    values[values == 0.0] = 0.5
    return np.array(rowptr, dtype=np.int64), col, values


def csr_to_dense(n, rowptr, col, val):
    d = np.zeros((n, n), dtype="<f8")
    rows = np.repeat(np.arange(n), np.diff(rowptr))
    d[rows, col] = val
    return d


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "benchmark_config.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "inputs"))
    ap.add_argument("--check-a1", help="Assignment 1 manifest.json: compare dense hashes")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.outdir, exist_ok=True)
    a1 = json.load(open(args.check_a1))["files"] if args.check_a1 else {}
    manifest = {"numpy_version": np.__version__, "dense": {}, "sparse": {}}

    # ---- dense inputs (identical to Assignment 1)
    for n in cfg["inputs"]["dense_sizes"]:
        for name, seed in (("A", SEED_A), ("B", SEED_B)):
            m = dense_matrix(n, seed)
            path = os.path.join(args.outdir, f"{name}_n{n}.bin")
            m.tofile(path)
            digest = hashlib.sha256(m.tobytes()).hexdigest()
            key = os.path.basename(path)
            ok = "" if key not in a1 else (" (matches Assignment 1)" if a1[key] == digest else " (DIFFERS from Assignment 1!)")
            manifest["dense"][key] = digest
            print(f"dense  {key}{ok}", file=sys.stderr)

    # ---- sparse inputs
    for spec in cfg["inputs"]["sparse"]:
        n, kind, param = spec["n"], spec["pattern"], spec["param"]
        tag = f"n{n}_uni_z{param:g}" if kind == "uni" else f"n{n}_band_w{param:g}"
        pid = (1 if kind == "uni" else 2) * 100000 + int(round(param * 10))
        entry = {"n": n, "pattern": kind, "param": param}
        for name in ("A", "B"):
            if kind == "uni" and param == 0:   # 0 % zeros: the dense Assignment-1 matrices
                dense = dense_matrix(n, SEED_A if name == "A" else SEED_B)
                rowptr = np.arange(0, n * n + 1, n, dtype=np.int64)
                col = np.tile(np.arange(n), n)
                val = dense.reshape(-1)
            elif kind == "uni":
                rowptr, col, val = csr_uniform(n, param, name, pid)
            else:
                rowptr, col, val = csr_band(n, int(param), name, pid)
            cpath = os.path.join(args.outdir, f"{name}_{tag}.csr")
            write_csr(cpath, n, rowptr, col, val)
            entry[f"{name}_nnz"] = int(len(col))
            entry[f"{name}_csr_sha256"] = sha(cpath)
            if n <= cfg["inputs"]["dense_file_max_n"]:
                bpath = os.path.join(args.outdir, f"{name}_{tag}.bin")
                csr_to_dense(n, rowptr, col, val).tofile(bpath)
                entry[f"{name}_bin_sha256"] = sha(bpath)
        manifest["sparse"][tag] = entry
        print(f"sparse {tag}: nnz(A)={entry['A_nnz']} nnz(B)={entry['B_nnz']}", file=sys.stderr)

    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
