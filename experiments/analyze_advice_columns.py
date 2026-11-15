#!/usr/bin/env python3
"""Summarize advice-column settings and row-by-column proxies for the revision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "revision_columns"
OUT = ROOT / "artifacts" / "revision_columns_summary.json"

records = [json.loads(p.read_text()) for p in sorted(DATA.glob("d*_*.json"))]
records = [r for r in records if r.get("status") == "ok"]
if len(records) != 108:
    raise SystemExit(f"Expected 108 successful grid records; found {len(records)}")

rows = np.array([r["num_rows"] for r in records], dtype=float)
cols = np.array([r["num_inner_cols"] for r in records], dtype=float)
work = np.array([r["analytical_constraints"] for r in records], dtype=float)
shape = np.array([r["d_model"] * r["n_layers"] * r["seq_len"] for r in records], dtype=float)
row_cells = rows * cols

def rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)

summary = {
    "n": len(records),
    "num_inner_cols": {
        "unique": sorted(set(int(x) for x in cols)),
        "min": int(cols.min()), "median": float(np.median(cols)), "max": int(cols.max()),
    },
    "dynamic_column_size": {
        "min": min(r["total_dynamic_col_size"] for r in records),
        "median": float(np.median([r["total_dynamic_col_size"] for r in records])),
        "max": max(r["total_dynamic_col_size"] for r in records),
    },
    "rows": {"min": int(rows.min()), "median": float(np.median(rows)), "max": int(rows.max())},
    "rows_times_inner_cols": {
        "min": int(row_cells.min()), "median": float(np.median(row_cells)), "max": int(row_cells.max()),
    },
    "spearman": {
        "rows_vs_operation_count": rho(work, rows),
        "rows_times_inner_cols_vs_operation_count": rho(work, row_cells),
        "rows_vs_dimension_product": rho(shape, rows),
        "rows_times_inner_cols_vs_dimension_product": rho(shape, row_cells),
        "rows_vs_rows_times_inner_cols": rho(rows, row_cells),
    },
    "by_width": {},
    "by_activation": {},
}
for width in sorted({r["d_model"] for r in records}):
    rs = [r for r in records if r["d_model"] == width]
    cell_counts = [r["num_rows"] * r["num_inner_cols"] for r in rs]
    summary["by_width"][str(width)] = {
        "n": len(rs),
        "num_inner_cols": sorted({r["num_inner_cols"] for r in rs}),
        "dynamic_column_size": sorted({r["total_dynamic_col_size"] for r in rs}),
        "rows_times_inner_cols": {
            "min": min(cell_counts),
            "median": float(np.median(cell_counts)),
            "max": max(cell_counts),
        },
    }
for act in sorted({r["activation"] for r in records}):
    rs = [r for r in records if r["activation"] == act]
    cs = [r["num_inner_cols"] for r in rs]
    summary["by_activation"][act] = {
        "n": len(rs), "unique_num_inner_cols": sorted(set(cs)),
        "rows_min": min(r["num_rows"] for r in rs),
        "rows_median": float(np.median([r["num_rows"] for r in rs])),
        "rows_max": max(r["num_rows"] for r in rs),
    }
OUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
