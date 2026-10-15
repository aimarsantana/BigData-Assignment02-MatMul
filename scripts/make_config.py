#!/usr/bin/env python3
"""Write config/benchmark_config.json (the single source of truth for inputs and experiments).

The file is generated here only to avoid typing the long input list by hand; the JSON is committed
and is what generate_inputs.py and run_benchmarks.py read. Re-run with --block B to change the
cache-blocking tile used by the main experiments (it is fixed after the block-size sweep D2).
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DENSE_METHODS = ["ijk", "ikj", "unroll4", "blocked"]
SIZES_D1 = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
ZS_S1 = [0, 50, 80, 90, 95, 98, 99, 99.5, 99.9]          # percentage of zeros, n = 1024
SIZES_S2 = [256, 512, 1024, 2048, 4096, 8192]            # z = 99 %
SIZES_S3 = [1024, 2048, 4096, 8192, 16384, 32768]        # z = 99.9 %
WIDTHS_S4 = [0, 2, 5, 13, 51]                            # banded, n = 1024
SIZES_S5 = [1024, 2048, 4096, 8192, 16384, 32768]        # banded, w = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=64)
    args = ap.parse_args()

    sparse_inputs = []
    def add(spec):
        if spec not in sparse_inputs:
            sparse_inputs.append(spec)

    s1 = [{"n": 1024, "pattern": "uni", "param": z} for z in ZS_S1]
    s2 = [{"n": n, "pattern": "uni", "param": 99} for n in SIZES_S2]
    s3 = [{"n": n, "pattern": "uni", "param": 99.9} for n in SIZES_S3]
    s4 = [{"n": 1024, "pattern": "band", "param": w} for w in WIDTHS_S4]
    s5 = [{"n": n, "pattern": "band", "param": 5} for n in SIZES_S5]
    for c in s1 + s2 + s3 + s4 + s5:
        add(c)

    cfg = {
        "description": "Assignment 2 benchmark configuration (read by generate_inputs.py and run_benchmarks.py).",
        "numeric_type": "float64 (IEEE-754 binary64)",
        "threads": 1,
        "block_size": args.block,
        "budget": {
            "max_seconds_per_sample": 120,
            "memory_budget_peak_working_set_bytes": 4000000000,
            "stop_rule": "if a series (experiment, method, pattern, parameter) exceeds the time or memory budget at size n, larger sizes of that series are NOT_ATTEMPTED",
        },
        "repetitions": {"n<=512": 10, "n<=2048": 5, "n>2048": 3},
        "warmup_runs": {"n<=512": 2, "n>512": 1},
        "batch": 0,
        "inputs": {
            "dense_sizes": sorted(set(SIZES_D1)),
            "dense_file_max_n": 4096,
            "sparse": sparse_inputs,
        },
        "experiments": [
            {"id": "D1_size", "type": "dense", "sizes": SIZES_D1, "methods": DENSE_METHODS},
            {"id": "D2_block", "type": "block_sweep", "sizes": [1024, 2048, 4096],
             "blocks": [8, 16, 32, 64, 128, 256, 512]},
            {"id": "S1_sparsity", "type": "sparse", "cases": s1,
             "dense_methods": DENSE_METHODS, "csr": True, "convert": True},
            {"id": "S2_size_z99", "type": "sparse", "cases": s2,
             "dense_methods": DENSE_METHODS, "csr": True, "convert": True},
            {"id": "S3_size_z99.9", "type": "sparse", "cases": s3,
             "dense_methods": ["blocked"], "csr": True, "convert": True},
            {"id": "S4_band_width", "type": "sparse", "cases": s4,
             "dense_methods": DENSE_METHODS, "csr": True, "convert": True},
            {"id": "S5_band_size", "type": "sparse", "cases": s5,
             "dense_methods": ["blocked"], "csr": True, "convert": True},
        ],
    }
    path = os.path.join(ROOT, "config", "benchmark_config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
