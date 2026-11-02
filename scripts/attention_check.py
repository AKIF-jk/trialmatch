#!/usr/bin/env python3
"""Error table for tiled attention against the float64 reference over the full test grid.

Grid: lengths 1, 7, 64, 256, 1024 x tiles 1, 16, 64 and larger than T x six mask kinds x float32 and float64.
The reference runs in float64 on the same (already rounded) inputs. Results go to results/attention/check.
"""
import torch

from trialmatch.attention.cases import MASK_KINDS, make_inputs, make_mask, query_length
from trialmatch.attention.reference import reference_attention
from trialmatch.attention.tiled import tiled_attention
from trialmatch.runtime import environment, results_dir, write_json

LENGTHS, TILES = [1, 7, 64, 256, 1024], [1, 16, 64, None]


def main():
    out = results_dir("attention/check")
    rows = []
    for kind in MASK_KINDS:
        for length in LENGTHS:
            q_len = query_length(kind, length)
            mask, causal = make_mask(kind, 2, q_len, length)
            for dtype in (torch.float32, torch.float64):
                q, k, v = make_inputs(q_len, length, dtype)
                want = reference_attention(q.double(), k.double(), v.double(), mask=mask, causal=causal)
                for tile in TILES:
                    got = tiled_attention(q, k, v, mask=mask, causal=causal, tile=tile or 2 * length + 1)
                    error = (got.double() - want).abs()
                    rows.append({
                        "mask": kind, "length": length, "queries": q_len, "tile": tile or "larger", "dtype": str(dtype).split(".")[1],
                        "max_error": float(error.max()), "mean_error": float(error.mean()), "has_nan": bool(got.isnan().any()),
                    })
    worst = {d: max(r["max_error"] for r in rows if r["dtype"] == d) for d in ("float32", "float64")}
    report = {"cases": len(rows), "worst_max_error": worst, "any_nan": any(r["has_nan"] for r in rows), "environment": environment(), "rows": rows}
    write_json(out / "report.json", report)
    print({k: v for k, v in report.items() if k not in ("rows", "environment")}, flush=True)


if __name__ == "__main__":
    main()
