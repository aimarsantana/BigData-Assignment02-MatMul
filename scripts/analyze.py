#!/usr/bin/env python3
"""Analysis of the Assignment 2 measurements: summary statistics, fits, figures and report tables.

Input : results/raw_samples.csv, results/runs.csv (written by run_benchmarks.py)
Output: results/summary.csv            median / quartiles / min / max / std of every configuration
        results/checksum_check.csv     bitwise agreement of the result checksums of all methods
        results/scaling_fit.csv        log-log slopes of time vs n
        results/key_numbers.json       numbers quoted in the report
        figures/*.png, *.pdf           all figures
        report/generated/*.html        tables included in the report

Statistics: central value = median of the measured samples (warm-up excluded); variability = interquartile
range (IQR, linear interpolation quartiles). Time per multiplication in seconds.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Okabe-Ito colour-blind safe palette
COL = {"ijk": "#D55E00", "ikj": "#0072B2", "unroll4": "#009E73", "blocked": "#CC79A7",
       "csr": "#000000", "csr+convert": "#E69F00", "convert": "#56B4E9"}
LABEL = {"ijk": "ijk (baseline)", "ikj": "ikj (loop reorder)", "unroll4": "ikj + unroll x4",
         "blocked": "blocked (tiled ikj)", "csr": "CSR product (kernel)",
         "csr+convert": "CSR product + dense->CSR conversion"}
MARK = {"ijk": "o", "ikj": "s", "unroll4": "^", "blocked": "D", "csr": "X", "csr+convert": "P"}
DENSE = ["ijk", "ikj", "unroll4", "blocked"]


# ----------------------------------------------------------------------------- loading
def load(results):
    s = pd.read_csv(os.path.join(results, "raw_samples.csv"), keep_default_na=False)
    r = pd.read_csv(os.path.join(results, "runs.csv"), keep_default_na=False)
    for df in (s, r):
        df["param"] = df["param"].astype(str)
        df["pattern"] = df["pattern"].astype(str)
    s["seconds"] = pd.to_numeric(s["ns_per_multiplication"]) / 1e9
    return s, r


KEY = ["experiment", "method", "representation", "n", "pattern", "param", "block"]


def summarise(s, r):
    m = s[s["phase"] == "measured"]
    g = m.groupby(KEY)["seconds"]
    out = g.agg(reps="count", median_s="median", min_s="min", max_s="max", mean_s="mean", std_s="std").reset_index()
    out["q1_s"] = g.quantile(0.25).values
    out["q3_s"] = g.quantile(0.75).values
    out["iqr_s"] = out["q3_s"] - out["q1_s"]
    out["rel_iqr"] = out["iqr_s"] / out["median_s"]
    out["batch"] = m.groupby(KEY)["batch"].first().values
    for c in ("nnz_A", "nnz_B", "nnz_C"):
        out[c] = pd.to_numeric(m.groupby(KEY)[c].first().values)
    runs = r.copy()
    runs["peak_working_set_kib"] = pd.to_numeric(runs["peak_working_set_kib"], errors="coerce")
    runs["peak_commit_kib"] = pd.to_numeric(runs["peak_commit_kib"], errors="coerce")
    full = runs[KEY + ["status", "warmup_runs", "measured_reps", "peak_working_set_kib", "peak_commit_kib",
                       "process_wall_s", "note"]].merge(out.drop(columns=["reps"]), on=KEY, how="left")
    full["reps"] = full["measured_reps"]
    full["param_f"] = pd.to_numeric(full["param"], errors="coerce")
    return full


# ----------------------------------------------------------------------------- formatting helpers
def fmt_t(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "&ndash;"
    if x >= 100:
        return f"{x:.0f} s"
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f} ms"
    return f"{x * 1e6:.1f} &micro;s"


def fmt_mem(kib):
    if kib is None or (isinstance(kib, float) and math.isnan(kib)):
        return "&ndash;"
    mb = kib / 1024
    return f"{mb / 1024:.2f} GiB" if mb >= 1024 else f"{mb:.1f} MiB"


def fmt_int(x):
    return "&ndash;" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{int(x):,}"


def html_table(header, rows, caption, cls=""):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><caption>{caption}</caption><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def save_fig(fig, name, figdir):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def style(ax, xlabel, ylabel, title=None, logx=False, logy=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if logx:
        ax.set_xscale("log", base=2) if logx == 2 else ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def cell(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def ok(df):
    return df[(df["status"] == "OK") & df["median_s"].notna()]


# ----------------------------------------------------------------------------- analysis
def slope(n, t):
    if len(n) < 3:
        return float("nan")
    return float(np.polyfit(np.log(n), np.log(t), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--gendir", default=os.path.join(ROOT, "report", "generated"))
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.gendir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 110})

    s, r = load(args.results)
    S = summarise(s, r)
    S.to_csv(os.path.join(args.results, "summary.csv"), index=False)
    key = {}

    # ---------------------------------------------------------------- checksum agreement
    m = s[(s["phase"] == "measured") & (s["method"] != "convert")]
    rows = []
    for (exp, n, pat, par), g in m.groupby(["experiment", "n", "pattern", "param"]):
        sums = g.groupby("method")["checksum"].nunique()
        rows.append(dict(experiment=exp, n=n, pattern=pat, param=par, methods=",".join(sorted(g["method"].unique())),
                         distinct_checksums=g["checksum"].nunique(), samples_stable=bool((sums == 1).all())))
    chk = pd.DataFrame(rows)
    chk.to_csv(os.path.join(args.results, "checksum_check.csv"), index=False)
    key["checksum_groups"] = int(len(chk))
    key["checksum_disagreements"] = int((chk["distinct_checksums"] != 1).sum())

    # ---------------------------------------------------------------- D1: dense optimisation
    d1 = ok(cell(S, experiment="D1_size"))
    fits = []
    if len(d1):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for meth in DENSE:
            g = d1[d1["method"] == meth].sort_values("n")
            if g.empty:
                continue
            ax.errorbar(g["n"], g["median_s"], yerr=[g["median_s"] - g["q1_s"], g["q3_s"] - g["median_s"]],
                        label=LABEL[meth], color=COL[meth], marker=MARK[meth], ms=4, lw=1.3, capsize=2)
            gg = g[g["n"] >= 256]
            fits.append(dict(experiment="D1_size", method=meth, n_min=int(gg["n"].min()), n_max=int(gg["n"].max()),
                             slope=slope(gg["n"], gg["median_s"])))
        ref = d1[d1["method"] == "ikj"].sort_values("n")
        if len(ref):
            n0, t0 = ref["n"].iloc[len(ref) // 2], ref["median_s"].iloc[len(ref) // 2]
            xs = np.array([ref["n"].min(), ref["n"].max()])
            ax.plot(xs, t0 * (xs / n0) ** 3, ls=":", color="grey", label=r"$\propto n^3$ (reference)")
        style(ax, "matrix size n", "time per multiplication (s)", "Dense methods: runtime vs n (median, bars = IQR)",
              logx=2, logy=True)
        ax.legend()
        save_fig(fig, "fig_dense_runtime", args.figdir)

        # speed-up over the baseline and over ikj
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        base = d1[d1["method"] == "ijk"].set_index("n")["median_s"]
        for meth in ("ikj", "unroll4", "blocked"):
            g = d1[d1["method"] == meth].sort_values("n")
            g = g[g["n"].isin(base.index)]
            if g.empty:
                continue
            ax.plot(g["n"], base.loc[g["n"]].values / g["median_s"].values, label=LABEL[meth], color=COL[meth],
                    marker=MARK[meth], ms=4)
        ax.axhline(1, color="grey", lw=0.8)
        style(ax, "matrix size n", "speed-up over ijk baseline (x)", "Speed-up of the optimised dense methods", logx=2)
        ax.legend()
        save_fig(fig, "fig_dense_speedup", args.figdir)

        # throughput
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for meth in DENSE:
            g = d1[d1["method"] == meth].sort_values("n")
            if g.empty:
                continue
            ax.plot(g["n"], 2.0 * g["n"] ** 3 / g["median_s"] / 1e9, label=LABEL[meth], color=COL[meth],
                    marker=MARK[meth], ms=4)
        style(ax, "matrix size n", "throughput (GFLOP/s, 2n$^3$ / median time)", "Dense methods: effective throughput", logx=2)
        ax.legend()
        save_fig(fig, "fig_dense_gflops", args.figdir)

        # tables
        ns = sorted(d1["n"].unique())
        allrun = cell(S, experiment="D1_size")
        rows = []
        for n in ns:
            row = [str(n)]
            for meth in DENSE:
                g = cell(d1, method=meth, n=n)
                if g.empty:
                    st = cell(allrun, method=meth, n=n)["status"]
                    row.append(st.iloc[0].replace("_", " ").lower() if len(st) else "&ndash;")
                else:
                    g = g.iloc[0]
                    row.append(f"{fmt_t(g['median_s'])} <span class='iqr'>({fmt_t(g['iqr_s'])})</span>")
            rows.append(row)
        key["dense_table_rows"] = len(rows)
        open(os.path.join(args.gendir, "table_dense_time.html"), "w", encoding="utf8").write(html_table(
            ["n"] + [LABEL[m_] for m_ in DENSE], rows,
            "Table 3. Dense methods: median time per multiplication (IQR in parentheses; 10 repetitions for n&nbsp;&le;&nbsp;512, "
            "5 for n&nbsp;&le;&nbsp;2048, 3 above). Entries without a time are runs that hit the declared budget.", "num"))
        rows = []
        for n in ns:
            b = cell(d1, method="ijk", n=n)
            if b.empty:
                continue
            bt = b["median_s"].iloc[0]
            row = [str(n), fmt_t(bt)]
            for meth in ("ikj", "unroll4", "blocked"):
                g = cell(d1, method=meth, n=n)
                row.append("&ndash;" if g.empty else f"{bt / g['median_s'].iloc[0]:.2f}&times;")
            rows.append(row)
        open(os.path.join(args.gendir, "table_dense_speedup.html"), "w", encoding="utf8").write(html_table(
            ["n", "ijk time", "ikj", "ikj + unroll x4", "blocked"], rows,
            "Table 4. Speed-up over the ijk baseline (ratio of medians) wherever the baseline completed within the budget.", "num"))
        # runtime spread
        key["dense_max_rel_iqr"] = float(d1["rel_iqr"].max())
        key["dense_median_rel_iqr"] = float(d1["rel_iqr"].median())
        # largest feasible / attempted
        lf = {}
        for meth in DENSE:
            a = allrun[allrun["method"] == meth]
            o = a[a["status"] == "OK"]
            lf[meth] = dict(largest_ok=int(o["n"].max()) if len(o) else None,
                            largest_attempted=int(a[a["status"] != "NOT_ATTEMPTED"]["n"].max()) if len(a) else None)
        key["dense_limits"] = lf
        # headline numbers
        for n in (1024, 2048, 4096):
            for meth in ("ikj", "unroll4", "blocked"):
                b, g = cell(d1, method="ijk", n=n), cell(d1, method=meth, n=n)
                if len(b) and len(g):
                    key[f"speedup_{meth}_n{n}"] = float(b["median_s"].iloc[0] / g["median_s"].iloc[0])
            for meth in DENSE:
                g = cell(d1, method=meth, n=n)
                if len(g):
                    key[f"gflops_{meth}_n{n}"] = float(2.0 * n ** 3 / g["median_s"].iloc[0] / 1e9)
                    key[f"time_{meth}_n{n}"] = float(g["median_s"].iloc[0])

    # ---------------------------------------------------------------- D2: block sweep
    d2 = ok(cell(S, experiment="D2_block"))
    if len(d2):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        rows = []
        for i, n in enumerate(sorted(d2["n"].unique())):
            g = d2[d2["n"] == n].sort_values("block")
            ax.errorbar(g["block"], g["median_s"], yerr=[g["median_s"] - g["q1_s"], g["q3_s"] - g["median_s"]],
                        label=f"blocked, n = {n}", color=palette[i % 4], marker="o", ms=4, capsize=2)
            ikj = cell(d1, method="ikj", n=n) if len(d1) else pd.DataFrame()
            if len(ikj):
                ax.axhline(ikj["median_s"].iloc[0], color=palette[i % 4], ls="--", lw=0.9, alpha=0.7)
            best = g.loc[g["median_s"].idxmin()]
            rows.append([str(n)] + [fmt_t(g[g["block"] == b]["median_s"].iloc[0]) if (g["block"] == b).any() else "&ndash;"
                                    for b in sorted(d2["block"].unique())] + [f"{int(best['block'])}"])
            key[f"best_block_n{n}"] = int(best["block"])
            key[f"best_block_time_n{n}"] = float(best["median_s"])
            key[f"worst_block_time_n{n}"] = float(g["median_s"].max())
        style(ax, "tile side (elements)", "time per multiplication (s)",
              "Block-size sweep (dashed = unblocked ikj at the same n)", logx=2, logy=True)
        ax.legend()
        save_fig(fig, "fig_block_sweep", args.figdir)
        open(os.path.join(args.gendir, "table_block.html"), "w", encoding="utf8").write(html_table(
            ["n"] + [f"b={b}" for b in sorted(d2["block"].unique())] + ["best b"], rows,
            "Table 5. Block-size sweep of the blocked kernel (median time per multiplication, 5 repetitions; 3 for n = 4096).", "num"))

    # ---------------------------------------------------------------- S*: sparse experiments
    sp = S[S["experiment"].str.startswith("S")].copy()
    def sparse_view(exp):
        return cell(sp, experiment=exp)

    def combined(exp):
        """per (n, pattern, param): kernel/convert/dense medians in one row."""
        d = ok(sparse_view(exp))
        rows = []
        for (n, pat, par), g in d.groupby(["n", "pattern", "param_f"]):
            row = dict(n=int(n), pattern=pat, param=par)
            c = g[g["method"] == "csr"]
            cv = g[g["method"] == "convert"]
            if len(c):
                c = c.iloc[0]
                row.update(t_csr=c["median_s"], iqr_csr=c["iqr_s"], nnz_A=c["nnz_A"], nnz_B=c["nnz_B"], nnz_C=c["nnz_C"],
                           ws_csr=c["peak_working_set_kib"], reps_csr=c["reps"])
            if len(cv):
                row.update(t_conv=cv["median_s"].iloc[0])
            for meth in DENSE:
                dm = g[g["method"] == meth]
                if len(dm):
                    row[f"t_{meth}"] = dm["median_s"].iloc[0]
                    row[f"ws_{meth}"] = dm["peak_working_set_kib"].iloc[0]
            dense_times = [row[f"t_{m_}"] for m_ in DENSE if f"t_{m_}" in row]
            if dense_times:
                row["t_dense_best"] = min(dense_times)
                row["best_dense"] = min((m_ for m_ in DENSE if f"t_{m_}" in row), key=lambda m_: row[f"t_{m_}"])
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["pattern", "param", "n"]) if rows else pd.DataFrame()

    # ----- S1: sparsity sweep at n = 1024
    c1 = combined("S1_sparsity")
    if len(c1):
        c1 = c1.sort_values("param")
        c1["density_pct"] = 100.0 - c1["param"]
        c1["out_density_pct"] = 100.0 * c1["nnz_C"] / c1["n"] ** 2
        c1.to_csv(os.path.join(args.results, "sparsity_sweep_table.csv"), index=False)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        z = c1["param"].values
        x = np.arange(len(z))
        for meth in ("ijk", "blocked"):
            if f"t_{meth}" in c1:
                ax.plot(x, c1[f"t_{meth}"], color=COL[meth], marker=MARK[meth], ms=4, label=f"dense {LABEL[meth]}")
        if "t_unroll4" in c1:
            ax.plot(x, c1["t_unroll4"], color=COL["unroll4"], marker=MARK["unroll4"], ms=4, label=f"dense {LABEL['unroll4']}")
        if "t_ikj" in c1:
            ax.plot(x, c1["t_ikj"], color=COL["ikj"], marker=MARK["ikj"], ms=4, label=f"dense {LABEL['ikj']}")
        ax.plot(x, c1["t_csr"], color=COL["csr"], marker=MARK["csr"], ms=5, lw=1.6, label=LABEL["csr"])
        if "t_conv" in c1:
            ax.plot(x, c1["t_csr"] + c1["t_conv"], color=COL["csr+convert"], marker=MARK["csr+convert"], ms=5, ls="--",
                    label=LABEL["csr+convert"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in z])
        style(ax, "sparsity of A and B (% zero entries; uniform random positions)", "time per multiplication (s)",
              "Sparsity sweep, n = 1024", logy=True)
        ax.legend(ncol=2)
        save_fig(fig, "fig_sparsity_sweep", args.figdir)

        # table
        rows = []
        for _, q in c1.iterrows():
            best = q.get("t_dense_best")
            rows.append([f"{q['param']:g}", fmt_int(q["nnz_A"]), fmt_int(q["nnz_C"]), f"{q['out_density_pct']:.1f}%",
                         fmt_t(q["t_csr"]), fmt_t(q.get("t_conv")),
                         fmt_t(q.get("t_ijk")), fmt_t(best) + (f" ({q['best_dense']})" if isinstance(q.get("best_dense"), str) else ""),
                         f"{q['t_ijk'] / q['t_csr']:.2f}&times;" if "t_ijk" in q and q["t_csr"] > 0 else "&ndash;",
                         f"{best / q['t_csr']:.2f}&times;" if best is not None and q["t_csr"] > 0 else "&ndash;",
                         f"{best / (q['t_csr'] + q.get('t_conv', 0)):.2f}&times;" if best is not None else "&ndash;"])
        open(os.path.join(args.gendir, "table_sparsity.html"), "w", encoding="utf8").write(html_table(
            ["zeros (%)", "nnz(A)", "nnz(C)", "density of C", "CSR product", "dense&rarr;CSR (A+B)", "dense ijk",
             "best dense", "ijk / CSR", "best dense / CSR", "best dense / (CSR + conv.)"], rows,
            "Table 6. Sparsity sweep at n = 1024 (uniform random positions, 10 repetitions for the sparse kernels). "
            "A ratio above 1 means the sparse kernel is faster. nnz(C) is the structural non-zero count of the product.", "num small"))

        # crossover points
        def first_zero_where(col):
            v = c1[c1[col]]
            return float(v["param"].min()) if len(v) else None
        c1["csr_faster_than_ijk"] = c1["t_csr"] < c1.get("t_ijk", np.inf)
        c1["csr_faster_than_best"] = c1["t_csr"] < c1.get("t_dense_best", np.inf)
        c1["csrconv_faster_than_best"] = (c1["t_csr"] + c1.get("t_conv", 0)) < c1.get("t_dense_best", np.inf)
        key["crossover_zero_pct_vs_ijk"] = first_zero_where("csr_faster_than_ijk")
        key["crossover_zero_pct_vs_best"] = first_zero_where("csr_faster_than_best")
        key["crossover_zero_pct_vs_best_with_conversion"] = first_zero_where("csrconv_faster_than_best")
        key["sparsity_rows"] = [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in rec.items()}
                                for rec in c1.drop(columns=[c for c in c1.columns if c.startswith("iqr")]).to_dict("records")]

    # ----- output fill-in (analytic vs observed)
    c4 = combined("S4_band_width")
    if len(c1) or len(c4):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        if len(c1):
            d_in = (100.0 - c1["param"]).values / 100.0
            ax.plot(d_in * 100, c1["out_density_pct"], "o-", color="#0072B2", label="uniform random: observed density of C")
            xs = np.logspace(-1.2, 2, 100)
            ax.plot(xs, 100 * (1 - (1 - (xs / 100) ** 2) ** 1024), ":", color="#0072B2",
                    label=r"model $1-(1-d^2)^n$ (independent positions)")
        if len(c4):
            ax.plot(100.0 * c4["nnz_A"] / c4["n"] ** 2, 100.0 * c4["nnz_C"] / c4["n"] ** 2, "s-", color="#D55E00",
                    label="banded: observed density of C")
        ax.plot([0.05, 100], [0.05, 100], color="grey", lw=0.8, ls="--", label="density of C = density of A")
        style(ax, "density of the inputs (% non-zero)", "density of the product C (% non-zero)",
              "Fill-in of the product, n = 1024", logx=True, logy=True)
        ax.legend()
        save_fig(fig, "fig_fill_in", args.figdir)

    # ----- S2 / S3: size scaling of CSR vs dense on uniform matrices
    c2, c3 = combined("S2_size_z99"), combined("S3_size_z99.9")
    if len(c2) or len(c3):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
        for ax, c, title in ((axes[0], c2, "99 % zeros (1 % dense)"), (axes[1], c3, "99.9 % zeros (0.1 % dense)")):
            if not len(c):
                continue
            ax.errorbar(c["n"], c["t_csr"], yerr=c["iqr_csr"] / 2, color=COL["csr"], marker=MARK["csr"], ms=5, label=LABEL["csr"])
            if "t_conv" in c:
                cc = c.dropna(subset=["t_conv"])
                ax.plot(cc["n"], cc["t_csr"] + cc["t_conv"], color=COL["csr+convert"], marker=MARK["csr+convert"], ms=5, ls="--",
                        label=LABEL["csr+convert"])
            for meth in DENSE:
                if f"t_{meth}" in c:
                    cm = c.dropna(subset=[f"t_{meth}"])
                    ax.plot(cm["n"], cm[f"t_{meth}"], color=COL[meth], marker=MARK[meth], ms=4, label=f"dense {LABEL[meth]}")
            style(ax, "matrix size n", "time per multiplication (s)", title, logx=2, logy=True)
        axes[0].legend(fontsize=7)
        save_fig(fig, "fig_sparse_size", args.figdir)
        for nm, c in (("z99", c2), ("z99.9", c3)):
            if len(c):
                cc = c[c["n"] >= c["n"].min()]
                key[f"csr_slope_{nm}"] = slope(cc["n"], cc["t_csr"])
                fits.append(dict(experiment=nm, method="csr", n_min=int(cc["n"].min()), n_max=int(cc["n"].max()),
                                 slope=slope(cc["n"], cc["t_csr"])))
        def size_table(c, cap, name):
            rows = []
            for _, q in c.iterrows():
                dense = q.get("t_dense_best")
                rows.append([str(int(q["n"])), fmt_int(q["nnz_A"]), fmt_int(q["nnz_C"]),
                             f"{100.0 * q['nnz_C'] / q['n'] ** 2:.2f}%", fmt_t(q["t_csr"]), fmt_t(q.get("t_conv")),
                             fmt_t(q.get("t_ijk")), fmt_t(dense) + (f" ({q['best_dense']})" if isinstance(q.get("best_dense"), str) else ""),
                             f"{dense / q['t_csr']:.1f}&times;" if dense is not None and not (isinstance(dense, float) and math.isnan(dense)) else "&ndash;",
                             fmt_mem(q.get("ws_csr")), fmt_mem(q.get("ws_blocked"))])
            open(os.path.join(args.gendir, name), "w", encoding="utf8").write(html_table(
                ["n", "nnz(A)", "nnz(C)", "density of C", "CSR product", "dense&rarr;CSR (A+B)", "dense ijk", "best dense",
                 "best dense / CSR", "peak RSS CSR run", "peak RSS dense run"], rows, cap, "num small"))
        if len(c2):
            size_table(c2, "Table 7. Size scaling at 99 % zeros (uniform positions). Dense columns are only filled where a dense run "
                       "was generated and completed (n &le; 4096; ijk stops at the time budget).", "table_size_z99.html")
        if len(c3):
            size_table(c3, "Table 8. Size scaling at 99.9 % zeros (uniform positions). Only the blocked dense method was run, up to n = 4096.",
                       "table_size_z99.9.html")

    # ----- S4 / S5: banded structure
    c5 = combined("S5_band_size")
    if len(c4) or len(c5):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
        if len(c4):
            c4s = c4.sort_values("param")
            ax = axes[0]
            ax.plot(c4s["param"] * 2 + 1, c4s["t_csr"], color=COL["csr"], marker=MARK["csr"], ms=5, label=LABEL["csr"])
            for meth in ("ijk", "blocked"):
                if f"t_{meth}" in c4s:
                    ax.plot(c4s["param"] * 2 + 1, c4s[f"t_{meth}"], color=COL[meth], marker=MARK[meth], ms=4, label=f"dense {LABEL[meth]}")
            style(ax, "non-zeros per row of A and B (2w+1)", "time per multiplication (s)", "Banded matrices, n = 1024", logx=True, logy=True)
            ax.legend(fontsize=7)
        if len(c5):
            ax = axes[1]
            ax.plot(c5["n"], c5["t_csr"], color=COL["csr"], marker=MARK["csr"], ms=5, label=LABEL["csr"])
            if "t_blocked" in c5:
                cb = c5.dropna(subset=["t_blocked"])
                ax.plot(cb["n"], cb["t_blocked"], color=COL["blocked"], marker=MARK["blocked"], ms=4, label=f"dense {LABEL['blocked']}")
            xs = np.array([c5["n"].min(), c5["n"].max()])
            ax.plot(xs, c5["t_csr"].iloc[0] * (xs / xs[0]), ":", color="grey", label=r"$\propto n$")
            style(ax, "matrix size n", "time per multiplication (s)", "Banded matrices (w = 5): scaling with n", logx=2, logy=True)
            ax.legend(fontsize=7)
            key["csr_slope_band_w5"] = slope(c5["n"], c5["t_csr"])
            fits.append(dict(experiment="band_w5", method="csr", n_min=int(c5["n"].min()), n_max=int(c5["n"].max()),
                             slope=slope(c5["n"], c5["t_csr"])))
        save_fig(fig, "fig_banded", args.figdir)
        rows = []
        for _, q in c4.sort_values("param").iterrows():
            best = q.get("t_dense_best")
            rows.append([f"{int(q['param'])}", f"{2 * int(q['param']) + 1}", fmt_int(q["nnz_A"]), fmt_int(q["nnz_C"]),
                         f"{100.0 * q['nnz_C'] / q['n'] ** 2:.2f}%", fmt_t(q["t_csr"]), fmt_t(q.get("t_conv")), fmt_t(q.get("t_ijk")),
                         fmt_t(best) + (f" ({q['best_dense']})" if isinstance(q.get("best_dense"), str) else ""),
                         f"{best / q['t_csr']:.1f}&times;" if best is not None else "&ndash;"])
        if rows:
            open(os.path.join(args.gendir, "table_band.html"), "w", encoding="utf8").write(html_table(
                ["w", "nnz/row", "nnz(A)", "nnz(C)", "density of C", "CSR product", "dense&rarr;CSR", "dense ijk", "best dense",
                 "best dense / CSR"], rows,
                "Table 9. Banded inputs at n = 1024 (|i-j| &le; w). The product has bandwidth 2w, so it stays sparse.", "num small"))
        rows = []
        for _, q in c5.iterrows():
            rows.append([str(int(q["n"])), fmt_int(q["nnz_A"]), fmt_int(q["nnz_C"]), fmt_t(q["t_csr"]), fmt_t(q.get("t_conv")),
                         fmt_t(q.get("t_blocked")), fmt_mem(q.get("ws_csr")), fmt_mem(q.get("ws_blocked"))])
        if rows:
            open(os.path.join(args.gendir, "table_band_size.html"), "w", encoding="utf8").write(html_table(
                ["n", "nnz(A)", "nnz(C)", "CSR product", "dense&rarr;CSR", "dense blocked", "peak RSS CSR run", "peak RSS dense run"],
                rows, "Table 10. Banded inputs with w = 5, size scaling.", "num small"))

    # ---------------------------------------------------------------- memory
    runs_ok = S[(S["status"].isin(["OK", "OVER_MEMORY_BUDGET"])) & S["peak_working_set_kib"].notna()]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    dm = runs_ok[(runs_ok["experiment"] == "D1_size") & (runs_ok["method"].isin(DENSE))]
    if len(dm):
        for meth in ("ijk", "blocked"):
            g = dm[dm["method"] == meth].sort_values("n")
            ax.plot(g["n"], g["peak_working_set_kib"] * 1024 / 2 ** 20, color=COL[meth], marker=MARK[meth], ms=4, label=f"dense {meth} (measured)")
        nn = np.array(sorted(dm["n"].unique()))
        ax.plot(nn, 3 * 8 * nn.astype(float) ** 2 / 2 ** 20, ":", color="grey", label="3 x 8n$^2$ bytes (A, B, C)")
    for exp, col, lab in (("S3_size_z99.9", "#000000", "CSR, 99.9 % zeros"), ("S2_size_z99", "#E69F00", "CSR, 99 % zeros"),
                          ("S5_band_size", "#56B4E9", "CSR, banded w = 5")):
        g = runs_ok[(runs_ok["experiment"] == exp) & (runs_ok["method"] == "csr")].sort_values("n")
        if len(g):
            ax.plot(g["n"], g["peak_working_set_kib"] * 1024 / 2 ** 20, marker="X", ms=5, color=col, label=lab + " (measured)")
    style(ax, "matrix size n", "peak working set of the process (MiB)", "Peak memory", logx=2, logy=True)
    ax.legend(fontsize=7)
    save_fig(fig, "fig_memory", args.figdir)
    mem_rows = []
    if len(dm):
        base = dm[dm["n"] == dm["n"].min()]["peak_working_set_kib"].min()
        key["process_baseline_kib_dense"] = float(base)
        for n in sorted(dm["n"].unique()):
            if n < 512:
                continue
            row = [str(n), fmt_mem(3 * 8 * n * n / 1024)]
            for meth in DENSE:
                g = cell(dm, method=meth, n=n)
                row.append(fmt_mem(g["peak_working_set_kib"].iloc[0]) if len(g) else "&ndash;")
            mem_rows.append(row)
        open(os.path.join(args.gendir, "table_memory.html"), "w", encoding="utf8").write(html_table(
            ["n", "3 x 8n&sup2; B (arrays)"] + [LABEL[m_] for m_ in DENSE], mem_rows,
            "Table 11. Peak working set of the dense benchmark processes (GetProcessMemoryInfo, PeakWorkingSetSize) vs the analytic array size.", "num small"))

    # ---------------------------------------------------------------- limits / failures
    st = S.groupby(["experiment", "status"]).size().unstack(fill_value=0)
    st.to_csv(os.path.join(args.results, "status_counts.csv"))
    bad = S[S["status"] != "OK"][["experiment", "method", "n", "pattern", "param", "status", "note"]]
    bad.to_csv(os.path.join(args.results, "non_ok_runs.csv"), index=False)
    key["non_ok_runs"] = bad.to_dict("records")
    key["runs_total"] = int(len(S))
    key["runs_ok"] = int((S["status"] == "OK").sum())
    key["samples_measured"] = int((s["phase"] == "measured").sum())
    key["samples_warmup"] = int((s["phase"] == "warmup").sum())
    # largest sizes
    for exp in ("S2_size_z99", "S3_size_z99.9", "S5_band_size"):
        g = S[(S["experiment"] == exp) & (S["method"] == "csr")]
        o = g[g["status"] == "OK"]
        key[f"csr_largest_ok_{exp}"] = int(o["n"].max()) if len(o) else None
        key[f"csr_largest_attempted_{exp}"] = int(g[g["status"] != "NOT_ATTEMPTED"]["n"].max()) if len(g) else None
    pd.DataFrame(fits).to_csv(os.path.join(args.results, "scaling_fit.csv"), index=False)
    key["fits"] = fits
    with open(os.path.join(args.results, "key_numbers.json"), "w") as f:
        json.dump(key, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("analysis done:", len(S), "configurations")


if __name__ == "__main__":
    main()
