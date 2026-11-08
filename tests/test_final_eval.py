import hashlib
import json
import statistics
from pathlib import Path

import pytest

from trialmatch.data.load import Row
from trialmatch.data.note import format_note
from trialmatch.eval.freeze import (
    FREEZE_FILE, RUN_LOG, FreezeError, build_freeze, check_rerun, file_hashes, open_test_gate, previous_runs, verify_freeze,
)
from trialmatch.eval.metrics import evaluate
from trialmatch.eval.summary import headline, mean_sd, patient_bootstrap, seed_summary

REPO = Path(__file__).resolve().parent.parent
NOTE = format_note(["a", "b", "c", "d"])
SHA = "a" * 40


def make_tree(root):
    files = {
        "trialmatch/data/a.py": "x = 1",
        "trialmatch/eval/b.py": "y = 2",
        "trialmatch/runtime.py": "z = 3",
        "scripts/final_eval.py": "main",
        "configs/final.json": json.dumps({"model": "m", "model_revision": "r", "draft": {"model": "d"}, "adapters": {"hf_revision": "h"}}),
        "splits/manifest.json": json.dumps({"sha256": "abc"}),
        "requirements-kaggle.txt": "torch==1",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def freeze(root):
    record = build_freeze(root, SHA, "final-freeze")
    (root / FREEZE_FILE).write_text(json.dumps(record))
    return record


def test_freeze_passes_on_an_unchanged_tree(tmp_path):
    root = make_tree(tmp_path)
    record = freeze(root)
    assert verify_freeze(root) == record
    assert record["code_commit"] == SHA and record["manifest_sha256"] == "abc"
    assert "trialmatch/data/a.py" in record["files"] and FREEZE_FILE not in record["files"]


def test_freeze_names_the_changed_file(tmp_path):
    root = make_tree(tmp_path)
    freeze(root)
    (root / "trialmatch/eval/b.py").write_text("y = 3")
    with pytest.raises(FreezeError, match="trialmatch/eval/b.py"):
        verify_freeze(root)


def test_freeze_catches_added_and_deleted_frozen_files(tmp_path):
    root = make_tree(tmp_path)
    freeze(root)
    (root / "trialmatch/data/new.py").write_text("q = 1")
    with pytest.raises(FreezeError, match="trialmatch/data/new.py"):
        verify_freeze(root)
    (root / "trialmatch/data/new.py").unlink()
    (root / "scripts/final_eval.py").unlink()
    with pytest.raises(FreezeError, match="scripts/final_eval.py"):
        verify_freeze(root)


def test_missing_freeze_record_blocks_the_test_split(tmp_path):
    with pytest.raises(FreezeError, match="missing"):
        verify_freeze(make_tree(tmp_path))


def test_unfrozen_files_do_not_matter(tmp_path):
    root = make_tree(tmp_path)
    freeze(root)
    (root / "docs").mkdir()
    (root / "docs/STATUS.md").write_text("changed")
    (root / "trialmatch/app").mkdir()
    (root / "trialmatch/app/ui.py").write_text("changed")
    verify_freeze(root)


def test_rerun_needs_a_reason(tmp_path):
    root = make_tree(tmp_path)
    assert check_rerun(root, None) == []
    log = root / RUN_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"runs": [{"started": "t", "commit": SHA}]}))
    assert len(previous_runs(root)) == 1
    for reason in (None, "", "   "):
        with pytest.raises(FreezeError, match="rerun"):
            check_rerun(root, reason)
    assert len(check_rerun(root, "scoring bug in the parser")) == 1


def test_gate_checks_freeze_commit_and_reruns_in_order(tmp_path):
    root = make_tree(tmp_path)
    with pytest.raises(FreezeError, match="missing"):
        open_test_gate(root, SHA, SHA)
    record = freeze(root)
    for expect in (None, "", "abc"):
        with pytest.raises(FreezeError, match="expect-commit"):
            open_test_gate(root, expect, SHA)
    with pytest.raises(FreezeError, match="not the expected"):
        open_test_gate(root, "b" * 40, SHA)
    with pytest.raises(FreezeError, match="not the expected"):
        open_test_gate(root, SHA, "")
    assert open_test_gate(root, SHA[:10], SHA) == (record, [])
    (root / "trialmatch/eval/b.py").write_text("changed after the freeze")
    with pytest.raises(FreezeError, match="frozen files differ"):
        open_test_gate(root, SHA, SHA)


def test_repo_freeze_covers_the_files_the_evaluation_depends_on():
    names = set(file_hashes(REPO))
    for needed in ("trialmatch/data/prompt.py", "trialmatch/data/split.py", "trialmatch/eval/metrics.py", "trialmatch/eval/parse.py",
                   "trialmatch/eval/generate.py", "trialmatch/eval/summary.py", "trialmatch/eval/freeze.py", "scripts/final_eval.py",
                   "configs/final.json", "splits/manifest.json"):
        assert needed in names


def test_only_final_eval_opens_the_test_split():
    sources = [*REPO.glob("trialmatch/**/*.py"), *REPO.glob("scripts/*.py"), *REPO.glob("tools/*.py"), REPO / "app.py"]
    offenders = [str(p.relative_to(REPO)) for p in sources if p.name not in ("final_eval.py", "split.py") and "allow_test=True" in p.read_text()]
    assert offenders == []
    source = (REPO / "scripts/final_eval.py").read_text()
    assert source.count('"test", allow_test=True)') == 1
    assert source.index("open_test_gate(ROOT") < source.index('"test", allow_test=True)')


def row(i, patient, kind, label, evidence=()):
    return Row(i, patient, "t", NOTE, kind, "criterion", tuple(evidence), label)


def out(label, ids):
    return json.dumps({"evidence_sentence_ids": ids, "label": label})


ROWS = [
    row(0, "p0", "inclusion", "included", [0]),
    row(1, "p0", "exclusion", "not excluded"),
    row(2, "p1", "inclusion", "not enough information"),
    row(3, "p1", "exclusion", "excluded", [1, 2]),
    row(4, "p2", "inclusion", "included", [3]),
    row(5, "p2", "exclusion", "not excluded"),
    row(6, "p3", "inclusion", "not applicable"),
    row(7, "p3", "exclusion", "not excluded"),
]
PERFECT = [out(r.label, list(r.evidence_ids)) for r in ROWS]
MIXED = [PERFECT[0], out("excluded", []), out("included", [0]), "not json", PERFECT[4], out("not excluded", [9]), PERFECT[6], out("not excluded", [])]


def test_bootstrap_point_estimates_equal_the_evaluator():
    for texts in (PERFECT, MIXED):
        report = evaluate(ROWS, texts)
        result = patient_bootstrap(ROWS, texts, n=50)
        assert result["accuracy"]["point"] == report["labels"]["overall"]["accuracy"]
        assert result["macro_f1"]["point"] == report["labels"]["overall"]["macro_f1"]
        assert result["evidence_macro_f1"]["point"] == report["evidence"]["overall"]["macro"]["f1"]


def test_bootstrap_resamples_patients_and_is_reproducible():
    first, second = patient_bootstrap(ROWS, MIXED, n=200, seed=3), patient_bootstrap(ROWS, MIXED, n=200, seed=3)
    assert first == second
    assert first["n_patients"] == 4 and first["n_resamples"] == 200
    for name in ("accuracy", "macro_f1", "evidence_macro_f1"):
        assert 0.0 <= first[name]["lo"] <= first[name]["hi"] <= 1.0
    assert patient_bootstrap(ROWS, MIXED, n=200, seed=4) != first


def test_bootstrap_collapses_when_nothing_varies():
    perfect = patient_bootstrap(ROWS, PERFECT, n=100)
    assert perfect["accuracy"] == {"point": 1.0, "lo": 1.0, "hi": 1.0}
    assert perfect["evidence_macro_f1"] == {"point": 1.0, "lo": 1.0, "hi": 1.0}
    one_patient = [row(i, "solo", "inclusion", "included", [0]) for i in range(4)]
    result = patient_bootstrap(one_patient, [out("included", [0]), out("included", [0]), "bad", "bad"], n=100)
    assert result["accuracy"]["lo"] == result["accuracy"]["hi"] == result["accuracy"]["point"] == 0.5


def test_mean_sd_uses_the_sample_convention():
    stats = mean_sd([0.426, 0.440, 0.473])
    assert stats["mean"] == pytest.approx(0.446333, abs=1e-6)
    assert stats["sd"] == pytest.approx(statistics.stdev([0.426, 0.440, 0.473]))
    assert stats["sd"] == pytest.approx(0.0241, abs=1e-4)
    assert (stats["min"], stats["max"]) == (0.426, 0.473)
    assert mean_sd([0.5])["sd"] == 0.0


def test_seed_summary_aggregates_each_headline_number():
    reports = {"seed42": evaluate(ROWS, PERFECT), "seed43": evaluate(ROWS, MIXED), "seed44": evaluate(ROWS, PERFECT)}
    summary = seed_summary(reports)
    accuracies = [reports[s]["labels"]["overall"]["accuracy"] for s in reports]
    assert summary["accuracy"]["per_seed"] == dict(zip(reports, accuracies))
    assert summary["accuracy"]["mean"] == pytest.approx(statistics.fmean(accuracies))
    assert summary["accuracy"]["sd"] == pytest.approx(statistics.stdev(accuracies))
    assert summary["malformed"]["per_seed"] == {"seed42": 0, "seed43": 1, "seed44": 0}
    assert "f1[excluded]" in summary and set(headline(reports["seed42"])) == set(summary)


def test_frozen_files_and_run_log_are_in_the_kaggle_tarball():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_kernel", REPO / "tools/build_kernel.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    for name in [*file_hashes(REPO), FREEZE_FILE, RUN_LOG]:
        assert name.split("/")[0] in build.SOURCE_PATHS, f"{name} is not shipped to Kaggle jobs"


def test_restrict_keeps_named_files_and_directories_only():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_kernel", REPO / "tools/build_kernel.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    files = {"trialmatch/eval/a.py": "1", "trialmatch/evalx/b.py": "2", "scripts/final_eval.py": "3", "scripts/train.py": "4", "configs/final.json": "5"}
    assert set(build.restrict(files, ["trialmatch/eval/", "scripts/final_eval.py"])) == {"trialmatch/eval/a.py", "scripts/final_eval.py"}
    assert build.restrict(files, []) == files


def test_hashes_use_the_form_the_kaggle_tarball_stores(tmp_path):
    import hashlib

    root = make_tree(tmp_path)
    (root / "trialmatch/data/__init__.py").write_text("")
    (root / "trialmatch/eval/b.py").write_text("y = 2\n\n\n")
    (root / "trialmatch/runtime.py").write_text("z = 3")
    found = file_hashes(root)
    assert found["trialmatch/data/__init__.py"] == hashlib.sha256(b"\n").hexdigest()
    assert found["trialmatch/eval/b.py"] == hashlib.sha256(b"y = 2\n").hexdigest()
    assert found["trialmatch/runtime.py"] == hashlib.sha256(b"z = 3\n").hexdigest()


def test_repo_hashes_equal_the_committed_tarball_form():
    import subprocess

    names = file_hashes(REPO)
    for name, digest in names.items():
        shown = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=REPO, capture_output=True, text=True)
        if shown.returncode:
            pytest.skip("not a git checkout")
        if (REPO / name).read_text() != shown.stdout:
            continue  # edited since the last commit: nothing to compare yet
        assert hashlib.sha256((shown.stdout.strip() + "\n").encode()).hexdigest() == digest, name
