import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TABLES = ("split", "baselines_val", "sweep_val", "seeds_val", "test_main", "test_bootstrap", "test_per_class", "decode_t4", "attention_bench", "attention_errors", "speculative_t4")
FIGURES = ("confusion_test_seed44", "val_macro_f1_by_epoch", "decode_tokens_per_second", "attention_peak_memory", "speculative_speedup")


def test_one_command_builds_every_table_and_figure(tmp_path):
    pytest.importorskip("matplotlib")
    done = subprocess.run([sys.executable, str(REPO / "scripts/make_report_assets.py"), "--out", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr
    assert "SKIP" not in done.stdout, done.stdout
    for name in TABLES:
        text = (tmp_path / "tables" / f"{name}.tex").read_text()
        assert "\\begin{tabular}" in text and "\\end{tabular}" in text
    for name in FIGURES:
        assert (tmp_path / "figures" / f"{name}.pdf").stat().st_size > 1000


def test_test_table_numbers_come_from_the_saved_summary(tmp_path):
    pytest.importorskip("matplotlib")
    import json

    subprocess.run([sys.executable, str(REPO / "scripts/make_report_assets.py"), "--out", str(tmp_path)], check=True, capture_output=True, cwd=REPO)
    summary = json.loads((REPO / "results/final/test/summary.json").read_text())
    text = (tmp_path / "tables/test_main.tex").read_text()
    for seed in ("seed42", "seed43", "seed44"):
        assert f"{summary['models'][seed]['strict']['macro_f1']:.3f}" in text
    assert f"{summary['majority']['strict']['accuracy']:.3f}" in text
