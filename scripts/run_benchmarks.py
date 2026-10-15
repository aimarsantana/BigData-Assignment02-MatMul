#!/usr/bin/env python3
"""Benchmark harness for Assignment 2 (Windows, single thread).

Every configuration (experiment, method, input) runs in ONE fresh process of build/matmul2.exe that
  - loads the shared input files (outside the timer),
  - performs the declared warm-up runs (kept in the raw data, excluded from statistics),
  - performs the measured repetitions (a sample = `batch` multiplications, reported per multiplication;
    the batch is chosen automatically so that a sample lasts ~20 ms, and is 1 for long kernels),
  - reports its peak working set / peak commit charge (GetProcessMemoryInfo) just before exiting.

Budgets (config/benchmark_config.json): if no new SAMPLE line appears within the per-sample budget
the process is killed and the configuration is recorded as TIMEOUT. A series (experiment, method,
pattern, parameter) that times out, fails to allocate or exceeds the memory budget is not attempted
at larger sizes (NOT_ATTEMPTED). No execution time is invented for those rows.

Outputs (results/, appended so that experiments can be run one at a time):
  raw_samples.csv  every warm-up and measured sample
  runs.csv         one row per process: status, peak memory, wall time
  info.csv         build / affinity information printed by the program
  logs/*.log       raw stdout/stderr of every process
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE = os.path.join(ROOT, "build", "matmul2.exe")

SAMPLE_COLS = ["experiment", "method", "representation", "n", "pattern", "param", "numeric_type", "threads",
               "block", "phase", "index", "batch", "ns_per_multiplication", "checksum", "nnz_A", "nnz_B", "nnz_C"]
RUN_COLS = ["experiment", "method", "representation", "n", "pattern", "param", "numeric_type", "threads", "block",
            "status", "warmup_runs", "measured_reps", "peak_working_set_kib", "peak_commit_kib", "process_wall_s",
            "exit_code", "note"]


def reps_for(cfg, n, sparse_kernel):
    if sparse_kernel:                      # cheap kernels: a fixed, larger number of repetitions
        return 10
    r = cfg["repetitions"]
    return r["n<=512"] if n <= 512 else (r["n<=2048"] if n <= 2048 else r["n>2048"])


def warmup_for(cfg, n):
    w = cfg["warmup_runs"]
    return w["n<=512"] if n <= 512 else w["n>512"]


def tag_of(case):
    n, kind, p = case["n"], case["pattern"], case["param"]
    return f"n{n}_uni_z{p:g}" if kind == "uni" else f"n{n}_band_w{p:g}"


def run_one(cmd, budget_s, log_path):
    """Run a process and stream its stdout. Returns (status, lines, wall_s, exit_code)."""
    t_start = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT)
    q = queue.Queue()

    def reader():
        for line in p.stdout:
            q.put(line.rstrip("\n"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    lines, status, last = [], "OK", time.monotonic()
    while True:
        remaining = budget_s - (time.monotonic() - last)
        if remaining <= 0:
            status = "TIMEOUT"
            p.kill()
            break
        try:
            item = q.get(timeout=min(remaining, 2.0))
        except queue.Empty:
            continue
        if item is None:
            break
        lines.append(item)
        if item.startswith("SAMPLE"):
            last = time.monotonic()
    p.wait()
    wall = time.monotonic() - t_start
    if status == "OK" and p.returncode != 0:
        status = "MEMORY_FAILURE" if p.returncode in (3, 7) else f"ERROR({p.returncode})"
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n" + "\n".join(lines) + "\n")
    return status, lines, wall, p.returncode


class Recorder:
    def __init__(self, res):
        os.makedirs(os.path.join(res, "logs"), exist_ok=True)
        self.res = res
        self.files = {}
        for name, cols in (("raw_samples.csv", SAMPLE_COLS), ("runs.csv", RUN_COLS), ("info.csv", ["key", "value"])):
            path = os.path.join(res, name)
            new = not os.path.exists(path)
            f = open(path, "a", newline="")
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            self.files[name] = (f, w)
        self.info_seen = set()

    def row(self, name, values):
        f, w = self.files[name]
        w.writerow(values)
        f.flush()

    def close(self):
        for f, _ in self.files.values():
            f.close()


def parse(lines):
    samples, mem, info = [], {}, []
    for ln in lines:
        parts = ln.split(",")
        if parts[0] == "SAMPLE":
            samples.append(parts[1:])
        elif parts[0] == "MEM":
            mem[parts[3]] = parts[4]
        elif parts[0] == "INFO":
            info.append((f"{parts[1]}.{parts[2]}", ",".join(parts[3:])))
    return samples, mem, info


def execute(rec, cfg, stopped, job):
    """job: dict(experiment, method, representation, n, pattern, param, block, cmd, warm, reps, series)."""
    budget = cfg["budget"]["max_seconds_per_sample"]
    mem_budget = cfg["budget"]["memory_budget_peak_working_set_bytes"]
    base = [job["experiment"], job["method"], job["representation"], job["n"], job["pattern"], job["param"],
            "float64", 1, job["block"]]
    if job["series"] in stopped:
        rec.row("runs.csv", base + ["NOT_ATTEMPTED", job["warm"], job["reps"], "", "", "", "",
                                    "series stopped by an earlier failure/over-budget run"])
        print(f"[{job['experiment']}] {job['method']} n={job['n']} {job['pattern']}{job['param']}: not attempted", flush=True)
        return
    if job.get("skip_note"):
        rec.row("runs.csv", base + ["NOT_ATTEMPTED", job["warm"], job["reps"], "", "", "", "", job["skip_note"]])
        print(f"[{job['experiment']}] {job['method']} n={job['n']}: not attempted ({job['skip_note']})", flush=True)
        return
    name = f"{job['experiment']}_{job['method']}_n{job['n']}_{job['pattern']}{job['param']}_b{job['block']}".replace(".", "p")
    status, lines, wall, ec = run_one(job["cmd"], budget, os.path.join(rec.res, "logs", name + ".log"))
    samples, mem, info = parse(lines)
    for key, val in info:
        if (key, val) not in rec.info_seen:
            rec.info_seen.add((key, val))
            rec.row("info.csv", [key, val])
    for s in samples:
        method, n, phase, idx, batch, ns, chk, na, nb, nc = s
        rec.row("raw_samples.csv", [job["experiment"], method, job["representation"], n, job["pattern"], job["param"],
                                    "float64", 1, job["block"], phase, idx, batch, ns, chk, na, nb, nc])
    ws = mem.get("peak_working_set_kib", "")
    note = ""
    if status == "OK" and ws not in ("", "-1") and int(ws) * 1024 > mem_budget:
        status, note = "OVER_MEMORY_BUDGET", f"peak working set above {mem_budget} bytes"
    rec.row("runs.csv", base + [status, job["warm"], job["reps"], ws, mem.get("peak_commit_kib", ""),
                                f"{wall:.3f}", ec, note])
    if status != "OK":
        stopped.add(job["series"])
    print(f"[{job['experiment']}] {job['method']} n={job['n']} {job['pattern']}{job['param']} b={job['block']}: "
          f"{status}, peak WS {ws} KiB, wall {wall:.1f}s", flush=True)


def dense_job(cfg, exp, method, n, A, B, block, pattern="dense", param="", series_extra=""):
    warm, reps = warmup_for(cfg, n), reps_for(cfg, n, False)
    return dict(experiment=exp, method=method, representation="dense", n=n, pattern=pattern, param=param, block=block,
                warm=warm, reps=reps, series=(exp, method, pattern, param, series_extra),
                cmd=[EXE, "dense", method, A, B, str(n), str(warm), str(reps), str(cfg["batch"]), str(block)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "benchmark_config.json"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "inputs"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--block", type=int, help="override the tile size of the 'blocked' method")
    ap.add_argument("experiments", nargs="*", help="experiment ids to run (default: all)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    block = args.block or cfg["block_size"]
    data = args.data_dir
    rec = Recorder(args.results)
    stopped = set()
    dense_max = cfg["inputs"]["dense_file_max_n"]

    for exp in cfg["experiments"]:
        if args.experiments and exp["id"] not in args.experiments:
            continue
        jobs = []
        if exp["type"] == "dense":
            for n in exp["sizes"]:
                for m in exp["methods"]:
                    A, B = (os.path.join(data, f"{x}_n{n}.bin") for x in "AB")
                    jobs.append(dense_job(cfg, exp["id"], m, n, A, B, block))
        elif exp["type"] == "block_sweep":
            for n in exp["sizes"]:
                for b in exp["blocks"]:
                    A, B = (os.path.join(data, f"{x}_n{n}.bin") for x in "AB")
                    jobs.append(dense_job(cfg, exp["id"], "blocked", n, A, B, b, series_extra=str(n)))
        else:  # sparse experiment: matched dense / CSR / conversion runs for every case
            for case in exp["cases"]:
                n, tag = case["n"], tag_of(case)
                A_csr, B_csr = (os.path.join(data, f"{x}_{tag}.csr") for x in "AB")
                A_bin, B_bin = (os.path.join(data, f"{x}_{tag}.bin") for x in "AB")
                pat, par = case["pattern"], case["param"]
                have_dense = n <= dense_max
                for m in exp.get("dense_methods", []):
                    j = dense_job(cfg, exp["id"], m, n, A_bin, B_bin, block, pattern=pat, param=par)
                    if not have_dense:
                        j["skip_note"] = f"dense input not generated for n > {dense_max} (8n^2 bytes per matrix)"
                    jobs.append(j)
                if exp.get("csr"):
                    warm, reps = warmup_for(cfg, n), reps_for(cfg, n, True)
                    jobs.append(dict(experiment=exp["id"], method="csr", representation="CSR", n=n, pattern=pat,
                                     param=par, block=0, warm=warm, reps=reps, series=(exp["id"], "csr", pat, par, ""),
                                     cmd=[EXE, "csr", A_csr, B_csr, str(warm), str(reps), str(cfg["batch"])]))
                if exp.get("convert"):
                    warm, reps = warmup_for(cfg, n), reps_for(cfg, n, True)
                    j = dict(experiment=exp["id"], method="convert", representation="dense->CSR", n=n, pattern=pat,
                             param=par, block=0, warm=warm, reps=reps, series=(exp["id"], "convert", pat, par, ""),
                             cmd=[EXE, "convert", A_bin, B_bin, str(n), str(warm), str(reps)])
                    if not have_dense:
                        j["skip_note"] = f"dense input not generated for n > {dense_max}"
                    jobs.append(j)
        # run: for the sparse experiments the per-series stop rule must follow increasing n
        for job in jobs:
            execute(rec, cfg, stopped, job)
    rec.close()


if __name__ == "__main__":
    sys.exit(main())
