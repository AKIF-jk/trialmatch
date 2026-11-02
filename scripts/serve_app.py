#!/usr/bin/env python3
"""Run the app on a Kaggle GPU kernel. Kaggle cannot read the private adapter repo, so adapters arrive through --fetch.

--selftest launches the app on localhost, drives it through gradio_client (segmentation, screening with rationale, an invalid
input and recovery, report files, the Inference Lab in every mode with sdpa and tiled attention) and saves what it sees to
results/app/selftest. --serve-seconds N also opens a public share link for N seconds (the link is printed in the kernel log,
which Kaggle shows only to the owner).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from trialmatch.runtime import check_versions, environment, results_dir, write_json


def ensure_gradio():
    try:
        import gradio

        if gradio.__version__ == "5.50.0":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio==5.50.0"], check=True)


def selftest(url, out):
    from gradio_client import Client

    from trialmatch.app.ui import EXAMPLES

    client, log = Client(url, verbose=False), {}
    example = EXAMPLES["Type 2 diabetes, hand-written"]
    numbered, status = client.predict(example["note"], api_name="/segment")
    log["segment"] = {"numbered": numbered, "status": status}

    def screen(label, note, inclusion, exclusion, rationale):
        t0 = time.perf_counter()
        status, outcome, table, detail, md, js = client.predict(note, inclusion, exclusion, rationale, api_name="/screen")
        log[label] = {"seconds": time.perf_counter() - t0, "status": status, "outcome_html": outcome, "table": table["data"], "detail_chars": len(detail)}
        if md:
            (out / f"{label}.md").write_text(Path(md).read_text())
            (out / f"{label}.json").write_text(Path(js).read_text())
        write_json(out / "selftest.json", log)

    screen("screen_example", numbered, example["inclusion"], example["exclusion"], True)
    screen("screen_empty_note", "", example["inclusion"], "", False)
    screen("screen_broken_numbering", "0. a\n5. b", example["inclusion"], "", False)
    screen("screen_no_criteria", numbered, "", "", False)
    screen("screen_recovery", numbered, example["inclusion"][:200].split("\n")[0], "", False)
    lab_note = "0. 67-year-old man with heart failure and an ejection fraction of 30%.\n1. He takes carvedilol and lisinopril.\n2. He has had atrial fibrillation for 3 years.\n3. He has no history of stroke.\n4. Creatinine is 1.1 mg/dL."
    criterion = "Diagnosis of heart failure with reduced ejection fraction"
    modes = ["no cache", "KV cache", "speculative", "speculative (prompt lookup)"]
    for label, args in {
        "lab_sdpa": (modes, "sdpa", 64, 2, "LoRA", 32, False, 1),
        "lab_sdpa_forced": (modes, "sdpa", 64, 4, "LoRA", 48, True, 2),
        "lab_tiled64_forced": (modes, "tiled online-softmax", 64, 2, "LoRA", 32, True, 1),
    }.items():
        t0 = time.perf_counter()
        status, table, commentary, generated, js = client.predict(lab_note, "inclusion", criterion, *args, api_name="/lab")
        log[label] = {"seconds": time.perf_counter() - t0, "status": status, "table": table["data"], "commentary": commentary, "generated": generated}
        (out / f"{label}.json").write_text(Path(js).read_text())
        write_json(out / "selftest.json", log)
    status = client.predict(lab_note, "inclusion", "Diagnosis of heart failure", [], "sdpa", 64, 2, "LoRA", 32, False, 1, api_name="/lab")[0]
    log["lab_no_modes"] = {"status": status}
    write_json(out / "selftest.json", log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="append", default=[], help="<folder>/<file>=<url>, repeatable")
    parser.add_argument("--adapter-root", default="adapters")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--serve-seconds", type=int, default=0)
    args = parser.parse_args()

    check_versions()
    ensure_gradio()
    from trialmatch.app.engine import Engine
    from trialmatch.app.ui import build_ui
    from trialmatch.decoding.experiment import fetch_files

    out = results_dir("app/selftest")
    write_json(out / "environment.json", environment())
    fetch_files(args.fetch, args.adapter_root)
    t0 = time.perf_counter()
    engine = Engine.load(adapter_root=args.adapter_root)
    print("engine loaded in", round(time.perf_counter() - t0, 1), "s on", engine.device, flush=True)

    def screen_fn(sentences, criteria, rationale):
        results, timing = engine.screen(sentences, criteria, rationale)
        return results, timing, engine.provenance()

    def lab_fn(sentences, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats):
        rows, text = engine.lab(sentences, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats)
        return rows, text, engine.provenance()

    demo = build_ui(screen_fn, lab_fn)
    demo.queue(max_size=8)
    _, local_url, share_url = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", share=bool(args.serve_seconds))
    print("local", local_url, "share", share_url, flush=True)
    if args.selftest:
        selftest(local_url, out)
        print("selftest done", flush=True)
    if args.serve_seconds:
        print("SHARE URL", share_url, flush=True)
        time.sleep(args.serve_seconds)
    demo.close()


if __name__ == "__main__":
    main()
