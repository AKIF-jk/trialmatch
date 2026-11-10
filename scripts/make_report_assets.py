#!/usr/bin/env python3
"""Regenerate every table and figure of the report from results/ with one command. CPU only.

    python3 scripts/make_report_assets.py [--out report] [--png] [--bench-source final|earlier]

Tables go to <out>/tables/*.tex (booktabs rows, no caption; the paper adds captions), figures to <out>/figures/*.pdf
(--png also writes previews). Each generator reads only files under results/ (plus splits/manifest.json). A generator whose inputs are missing is skipped with a warning, so a
partial checkout still builds what it can. Needs matplotlib.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LABELS = ("included", "not included", "excluded", "not excluded", "not enough information", "not applicable")
SHORT = {"included": "incl.", "not included": "not incl.", "excluded": "excl.", "not excluded": "not excl.", "not enough information": "NEI", "not applicable": "N/A"}
SEEDS = ("seed42", "seed43", "seed44")


BENCH_SOURCE = "final"  # T4 benchmarks: results/final_bench (rerun from the final-freeze tag) or "earlier" (results/, each at its own commit)


def load(path):
    return json.loads((ROOT / path).read_text())


def bench(rel):
    """Path of a T4 benchmark file: the rerun from the freeze tag when present and selected, else the earlier report."""
    if BENCH_SOURCE == "final" and (RESULTS / "final_bench" / rel).exists():
        return f"results/final_bench/{rel}"
    return f"results/{rel}"


def f3(x):
    return f"{x:.3f}"


def pm(values):
    return f"{statistics.fmean(values):.3f} $\\pm$ {statistics.stdev(values):.3f}" if len(values) > 1 else f3(values[0])


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_table(out, name, header, rows, align=None, note=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.insert(0, f"% {note}")
    path = out / "tables" / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---- tables -------------------------------------------------------------------------------------------------

def split_table(out):
    m = load("splits/manifest.json")
    rows = []
    for split in ("train", "val", "test"):
        counts = {label: 0 for label in LABELS}
        for kind in m["class_counts"][split].values():
            for label, n in kind.items():
                counts[label] += n
        rows.append([split, len(m["patients"][split]), sum(counts.values()), *[counts[l] for l in LABELS]])
    return write_table(out, "split", ["Split", "Patients", "Rows", *[SHORT[l] for l in LABELS]], rows,
                       note="Counts from splits/manifest.json; train includes annotation 883 (null criterion), which training and metrics drop.")


def cells(report):
    o = report["labels"]["overall"]
    return [f3(o["accuracy"]), f3(o["macro_f1"]), f3(report["evidence"]["overall"]["macro"]["f1"]), report["malformed"]]


def baselines_table(out):
    rows = []
    try:
        h = load("results/final/val/summary.json")["majority"]["strict"]  # per-type majority on validation, from the dry run
        rows.append(["Majority class (per criterion type)", "n/a", f3(h["accuracy"]), f3(h["macro_f1"]), f3(h["evidence_macro_f1"]), h["malformed"]])
    except FileNotFoundError:
        print("  note: validation majority row skipped (results/final/val/summary.json missing)")
    for key, name in (("qwen", "Qwen2.5-1.5B-Instruct"), ("phi3", "Phi-3-mini-4k-instruct")):
        r = load(f"results/baseline/{key}/report.json")
        rows.append([name, "strict", *cells(r["strict"])])
        rows.append([name, "repaired", *cells(r["repair"])])
    return write_table(out, "baselines_val", ["Model", "Parsing", "Accuracy", "Macro-F1", "Evidence F1", "Malformed"], rows, "llrrrr")


def sweep_table(out):
    runs = (("base_r16", "r=16 (base)"), ("r8", "r=8"), ("r32", "r=32"), ("label_first", "label first"), ("over4", "rare labels x4"), ("last2", "last 2 layers"))
    rows = []
    for key, name in runs:
        r = load(f"results/train/{key}/report.json")
        b = r["best"]["strict"]
        rows.append([name, r["best"]["epoch"] + 1, *cells(b), f"{r['peak_gpu_gb']:.1f}"])
    return write_table(out, "sweep_val", ["Run", "Epoch", "Accuracy", "Macro-F1", "Evidence F1", "Malformed", "Peak GB"], rows, "lrrrrrr")


def seeds_table(out):
    s = load("results/train/seeds_summary.json")
    rows = []
    for key, name in (("over4", "rare labels x4 (chosen)"), ("r32", "r=32"), ("label_first", "label first")):
        runs = s[key]
        rows.append([name, " / ".join(f3(r["mf1"]) for r in runs), pm([r["mf1"] for r in runs]), pm([r["acc"] for r in runs]), pm([r["ev"] for r in runs])])
    return write_table(out, "seeds_val", ["Config", "Macro-F1 per seed", "Macro-F1", "Accuracy", "Evidence F1"], rows, "lrrrr")


def test_tables(out):
    s = load("results/final/test/summary.json")
    rows = []

    def row(name, h, extra=""):
        return [name, f3(h["accuracy"]), f3(h["macro_f1"]), f3(h["macro_f1_supported"]), f3(h["evidence_micro_f1"]), f3(h["evidence_macro_f1"]), h["malformed"]]

    rows.append(row("Majority class", s["majority"]["strict"]))
    rows.append(row("Base, strict", s["models"]["base"]["strict"]))
    rows.append(row("Base, repaired", s["models"]["base"]["repair"]))
    for seed in SEEDS:
        rows.append(row(f"Fine-tuned, {seed.replace('seed', 'seed ')}", s["models"][seed]["strict"]))
    f = s["fine_tuned_seeds"]
    rows.append(["Fine-tuned, mean $\\pm$ SD", *[pm(list(f[k]["per_seed"].values())) for k in ("accuracy", "macro_f1", "macro_f1_supported", "evidence_micro_f1", "evidence_macro_f1")], 0])
    t1 = write_table(out, "test_main", ["Model", "Accuracy", "Macro-F1", "Macro-F1 (5)", "Evid. micro", "Evid. macro", "Malformed"], rows, "lrrrrrr",
                     note=f"Test split, {s['n_rows']} rows, {s['n_patients']} patients; `excluded` has no test row, so Macro-F1 (5) averages the five supported labels.")
    b = []
    for seed in SEEDS:
        bs = s["models"][seed]["bootstrap_strict"]
        b.append([seed.replace("seed", "seed "), *[f"{bs[k]['point']:.3f} [{bs[k]['lo']:.3f}, {bs[k]['hi']:.3f}]" for k in ("accuracy", "macro_f1", "evidence_macro_f1")]])
    bm = s["majority"]["bootstrap_strict"]
    b.append(["majority", *[f"{bm[k]['point']:.3f} [{bm[k]['lo']:.3f}, {bm[k]['hi']:.3f}]" for k in ("accuracy", "macro_f1", "evidence_macro_f1")]])
    t2 = write_table(out, "test_bootstrap", ["Model", "Accuracy", "Macro-F1", "Evid. macro-F1"], b, "lrrr",
                     note="95% percentile intervals from resampling the test patients with replacement (1000 draws).")
    per = {seed: load(f"results/final/test/{seed}/report.json")["strict"]["labels"]["overall"]["per_class"] for seed in SEEDS}
    pc = []
    for label in LABELS:
        support = per["seed44"][label]["support"]
        pc.append([label, support, *[f3(per[s_][label]["f1"]) for s_ in SEEDS], pm([per[s_][label]["f1"] for s_ in SEEDS]),
                   f3(per["seed44"][label]["precision"]), f3(per["seed44"][label]["recall"])])
    t3 = write_table(out, "test_per_class", ["Label", "Support", "F1 s42", "F1 s43", "F1 s44", "F1 mean $\\pm$ SD", "P (s44)", "R (s44)"], pc, "lrrrrrrr",
                     note="Test split. `excluded` has support 0, so every value for it is 0 by the zero-division rule.")
    return [t1, t2, t3]


def unseen_table(out):
    cases = [json.loads(l) for l in open(RESULTS / "final/unseen/cases.jsonl")]
    report = load("results/final/unseen/report.json")
    rows = []
    for seed in SEEDS:
        preds = {o["id"]: o for o in map(json.loads, open(RESULTS / f"final/unseen/{seed}/outputs.jsonl"))}
        hit = lambda kind: [preds[c["id"]]["label"] == c["expected_label"] for c in cases if kind is None or c["type"] == kind]
        inc, exc, allc = hit("inclusion"), hit("exclusion"), hit(None)
        rows.append([seed.replace("seed", "seed "), f"{sum(inc)}/{len(inc)}", f"{sum(exc)}/{len(exc)}", f"{sum(allc)}/{len(allc)}", f3(sum(allc) / len(allc)), f3(report["models"][seed]["evidence_macro_f1"])])
    return write_table(out, "unseen_cases", ["Model", "Inclusion", "Exclusion", "All", "Label accuracy", "Evid. macro-F1"], rows, "lrrrrr",
                       note="26 hand-written criteria on 3 unseen patients; expected labels written by the assistant, not clinician-reviewed.")


def decode_table(out):
    d = load(bench("decoding/benchmark_cuda_float16/report.json"))
    rows = []
    for c in d["cells"]:
        u, k = c["paths"]["uncached"], c["paths"]["cached"]
        rows.append([f"{c['prompt']} x {c['output']}", c["batch"], f"{u['total_seconds']['median']:.2f}", f"{k['total_seconds']['median']:.2f}",
                     f"{u['total_seconds']['median'] / k['total_seconds']['median']:.1f}", f"{k['total_tokens_per_second']['median']:.1f}",
                     f"{k['peak_memory_gb']['median']:.2f}", "yes" if c["tokens_equal"] else "no"])
    return write_table(out, "decode_t4", ["Prompt x output", "Batch", "No cache s", "KV cache s", "Speedup", "Cached tok/s", "Peak GB", "Tokens equal"], rows, "lrrrrrrl",
                       note=f"Tesla T4, fp16, sdpa, median of 5 repeats ({bench('decoding/benchmark_cuda_float16/report.json')}).")


def attention_tables(out):
    a = load(bench("attention/bench_cuda_float16/report.json"))
    rows = []
    for c in a["cells"]:
        b = c["backends"]
        rows.append([c["case"], c["length"], *[f"{b[k]['median_ms']:.2f}" for k in ("reference", "tiled_64", "sdpa")], *[f"{b[k]['peak_extra_mb']:.0f}" for k in ("reference", "tiled_64", "sdpa")]])
    t1 = write_table(out, "attention_bench", ["Case", "T", "ms ref", "ms tiled 64", "ms sdpa", "MB ref", "MB tiled 64", "MB sdpa"], rows, "lrrrrrrr",
                     note="Tesla T4, fp16, 12 query heads, 2 KV heads, head dim 128, median ms and peak extra MB.")
    c = load("results/attention/check/report.json")
    t2 = write_table(out, "attention_errors", ["Dtype", "Cases", "Worst max error", "Tolerance"],
                     [[dt, sum(r["dtype"] == dt for r in c["rows"]), f"{c['worst_max_error'][dt]:.2e}", tol] for dt, tol in (("float32", "$10^{-5}$"), ("float64", "$10^{-12}$"))], "lrrr")
    return [t1, t2]


def spec_table(out):
    rows = []
    for tag, path in (("model draft", bench("speculative/bench_cuda_float16/report.json")), ("prompt lookup", bench("speculative/bench_cuda_float16_lookup/report.json"))):
        w = load(path)["workloads"]
        for kind in ("json", "text"):
            for c in w[kind]["cells"]:
                rows.append([kind, c["draft"], c["k"], f3(c["acceptance_rate"]), f"{c['tokens_per_round']:.2f}", c["target_forwards"], f"{c['speedup_total']:.2f}", f"{c['prompts_equal']}/{c['prompts']}"])
    return write_table(out, "speculative_t4", ["Workload", "Draft", "K", "Acceptance", "Tokens/round", "Target passes", "Speedup", "Prompts equal"], rows, "llrrrrrr",
                       note="Tesla T4, fp16; speedup = greedy cached total time / speculative total time.")


# ---- figures ------------------------------------------------------------------------------------------------

def figures(out, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{name}.pdf", metadata={"CreationDate": None, "Creator": "make_report_assets.py", "Producer": None})  # no timestamp: same bytes every run
        if png:
            fig.savefig(out / "figures" / f"{name}.png", dpi=120)
        plt.close(fig)

    written = []
    try:
        m = load("results/final/test/seed44/report.json")["strict"]["labels"]["overall"]["confusion"]
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        mat = m["matrix"]
        norm = [[v / sum(row) if sum(row) else 0 for v in row] for row in mat]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        names = [SHORT.get(l, l) for l in m["labels"]]
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(mat)), names[:-1])
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                if v:
                    ax.text(j, i, v, ha="center", va="center", color="white" if norm[i][j] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("expert label")
        save(fig, "confusion_test_seed44")
        written.append("confusion_test_seed44")
    except FileNotFoundError as exc:
        print(f"  skipped confusion matrix: {exc}")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key, label in (("over4", "seed 42"), ("over4_s43", "seed 43"), ("over4_s44", "seed 44")):
        ep = load(f"results/train/{key}/report.json")["epochs"]
        ax.plot([e["epoch"] + 1 for e in ep], [e["strict_macro_f1"] for e in ep], marker="o", label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.set_xticks([1, 2, 3])
    ax.legend()
    save(fig, "val_macro_f1_by_epoch")
    written.append("val_macro_f1_by_epoch")

    d = load(bench("decoding/benchmark_cuda_float16/report.json"))["cells"]
    single = [c for c in d if c["batch"] == 1 and c["output"] == 256]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    xs = [c["prompt"] for c in single]
    for path, label in (("uncached", "no cache"), ("cached", "KV cache")):
        ax.plot(xs, [c["paths"][path]["total_tokens_per_second"]["median"] for c in single], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt tokens (256 new tokens)")
    ax.set_ylabel("tokens / s (T4, fp16)")
    ax.legend()
    save(fig, "decode_tokens_per_second")
    written.append("decode_tokens_per_second")

    a = load(bench("attention/bench_cuda_float16/report.json"))["cells"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    pre = [c for c in a if c["case"] == "prefill"]
    for k, label in (("reference", "full score matrix"), ("tiled_64", "tiled, tile 64"), ("tiled_256", "tiled, tile 256"), ("sdpa", "sdpa")):
        ax.plot([c["length"] for c in pre], [c["backends"][k]["peak_extra_mb"] for c in pre], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length T (prefill)")
    ax.set_ylabel("peak extra memory (MB)")
    ax.legend(fontsize=8)
    save(fig, "attention_peak_memory")
    written.append("attention_peak_memory")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    series = {}
    for path in (bench("speculative/bench_cuda_float16/report.json"), bench("speculative/bench_cuda_float16_lookup/report.json")):
        for c in load(path)["workloads"]["json"]["cells"]:
            series.setdefault(c["draft"], []).append((c["k"], c["speedup_total"]))
    for draft, pts in series.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label={"base": "0.5B draft", "lora": "0.5B draft + LoRA", "lookup": "prompt lookup"}.get(draft, draft))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("draft length K")
    ax.set_ylabel("speedup over cached greedy")
    ax.legend(fontsize=8)
    save(fig, "speculative_speedup")
    written.append("speculative_speedup")
    return written


def test_by_type_table(out):
    """Test accuracy, macro-F1 and evidence F1 for inclusion and exclusion rows separately (strict parsing)."""
    rows = []
    for seed in SEEDS:
        r = load(f"results/final/test/{seed}/report.json")["strict"]
        for kind in ("inclusion", "exclusion"):
            lab, ev = r["labels"][kind], r["evidence"][kind]
            rows.append([seed.replace("seed", "seed "), kind, lab["n"], f3(lab["accuracy"]), f3(lab["macro_f1"]), f3(lab["macro_f1_supported"]), f3(ev["micro"]["f1"]), f3(ev["macro"]["f1"])])
    return write_table(out, "test_by_type", ["Model", "Type", "Rows", "Accuracy", "Macro-F1", "Macro-F1 (supp.)", "Evid. micro F1", "Evid. macro F1"], rows, "llrrrrrr",
                       note="Test split, strict parsing, inclusion and exclusion rows scored separately (macro-F1 over the six-label list; supp. = over labels with support).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "report"))
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--bench-source", choices=("final", "earlier"), default="final", help="T4 benchmark reports: the rerun from the final-freeze tag (default) or the earlier ones")
    args = parser.parse_args()
    global BENCH_SOURCE
    BENCH_SOURCE = args.bench_source
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    produced = []
    for gen in (split_table, baselines_table, sweep_table, seeds_table, test_tables, test_by_type_table, unseen_table, decode_table, attention_tables, spec_table):
        try:
            made = gen(out)
            produced += made if isinstance(made, list) else [made]
            print(f"ok   {gen.__name__}")
        except FileNotFoundError as exc:
            print(f"SKIP {gen.__name__}: missing {exc.filename}")
    try:
        produced += figures(out, args.png)
        print("ok   figures")
    except ImportError:
        print("SKIP figures: matplotlib is not installed")
    print(f"{len(produced)} assets under {out}")


if __name__ == "__main__":
    main()
