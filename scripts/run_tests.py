#!/usr/bin/env python3
"""Run the pytest suite inside a kernel and keep the report in the job results."""
import sys

import pytest

from trialmatch.runtime import environment, results_dir, write_json

if __name__ == "__main__":
    out = results_dir("tests")
    code = pytest.main(["-q", "-p", "no:cacheprovider", f"--junitxml={out / 'junit.xml'}", *(sys.argv[1:] or ["tests"])])
    write_json(out / "report.json", {"exit_code": int(code), "environment": environment()})
    sys.exit(int(code))
