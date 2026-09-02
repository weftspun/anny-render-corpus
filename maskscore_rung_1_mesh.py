"""Rung 1 MaskScore — Mesh stub, ETNF-normal three-parquet emit.

Walking-skeleton Rung 1 for the Mesh stub: rest ANNY pose is the input, rank1 is
identity (== input), rank5 is the same identity with a small random pose perturbation.
Both are rendered over 64 sphere_hammersley views with depth + normal AOVs and scored
against the input by `score_render_pair.py`. Identity control (rank1 ≡ input) and
negative control (rank5 > rank1) are asserted before the emit and would fail the write
if the metric had lost the ability to detect a change.

ETNF-NORMAL EMIT, IN THREE PARQUETS RATHER THAN ONE FLAT ROW.

The earlier single-file version carried four means (rank1/5 x depth/normal L1) that any
consumer can compute from the per-view scores; the per-view scores themselves in a flat
list with an implicit "first 64 = rank1, next 64 = rank5" ordering; and n_views, which
is a length. All three are derivable and, per CLAUDE.md, out. `perturbation_sigma_rad`
and `seed` were duplicated in the row and in poses.json, and are dropped from the row
because the row already points at poses.json.

  maskscore_rung_1_mesh.parquet             the root row: key, task_type, dimension,
                                            input_mesh, poses.
  maskscore_rung_1_mesh_candidates.parquet  satellite: (row_key, candidate, rank,
                                            mesh_path). One row per candidate.
  maskscore_rung_1_mesh_scores.parquet      satellite: (row_key, candidate, view_index,
                                            depth_l1, normal_l1, normal_dot). One row
                                            per (candidate, view). 128 rows for
                                            2 candidates x 64 views.

Every parquet ships with ZSTD compression.

Usage:
    pixi run --environment anny-mac python maskscore_rung_1_mesh.py \
        --pose-dir build/bootstrap \
        --scores build/scores \
        --out-dir .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
KEY = "rung1/bootstrap/mesh"

# The two candidates this row carries. Kept in a table rather than a list so the emit
# generalises to arbitrary rank counts without a schema change.
CANDIDATES = [
    ("rank1", 1, "rank1.npz"),
    ("rank5", 5, "rank5.npz"),
]


def assert_controls(scores: dict[str, dict]) -> None:
    """Refuse to write if rule 2's controls were not satisfied by the metric."""
    r1 = scores["rank1"]
    r5 = scores["rank5"]
    max_r1 = max(v["depth_l1"] for v in r1["views"])
    if max_r1 > 1e-6:
        raise SystemExit(f"identity control failed: rank1 max depth_l1 {max_r1:.6e} > 1e-6")
    if r5["mean_depth_l1"] <= r1["mean_depth_l1"]:
        raise SystemExit(f"negative control failed: rank5 mean depth_l1 "
                         f"({r5['mean_depth_l1']:.6f}) not strictly worse than rank1 "
                         f"({r1['mean_depth_l1']:.6f})")


def build_tables(pose_dir: Path, scores_dir: Path) -> tuple[pa.Table, pa.Table, pa.Table]:
    scores = {}
    for name, _rank, _mesh in CANDIDATES:
        scores[name] = json.loads((scores_dir / f"{name}.json").read_text())
    assert_controls(scores)

    def real(p: Path) -> str:
        if not p.is_file():
            raise SystemExit(f"missing: {p}")
        return str(p.relative_to(HERE))

    root = pa.table({
        "key": [KEY],
        "task_type": ["pose_change"],
        "dimension": ["instruction_following"],
        "input_mesh": [real(pose_dir / "rest.npz")],
        "poses": [real(pose_dir / "poses.json")],
    })

    cands = pa.table({
        "row_key": [KEY] * len(CANDIDATES),
        "candidate": [name for name, _, _ in CANDIDATES],
        "rank": [rank for _, rank, _ in CANDIDATES],
        "mesh_path": [real(pose_dir / mesh) for _, _, mesh in CANDIDATES],
    })

    n_views = len(scores["rank1"]["views"])
    if n_views != 64:
        raise SystemExit(f"expected 64 views, got {n_views}")
    # Enforce the same view count on every candidate; otherwise the satellite table
    # silently gets ragged.
    for name in scores:
        if len(scores[name]["views"]) != n_views:
            raise SystemExit(f"candidate {name} has {len(scores[name]['views'])} views, "
                             f"expected {n_views}")

    rows_key, rows_cand, rows_view = [], [], []
    rows_dl1, rows_nl1, rows_ndot = [], [], []
    for name, _, _ in CANDIDATES:
        for i, v in enumerate(scores[name]["views"]):
            rows_key.append(KEY)
            rows_cand.append(name)
            rows_view.append(i)
            rows_dl1.append(v["depth_l1"])
            rows_nl1.append(v["normal_l1"])
            rows_ndot.append(v["normal_dot"])
    scores_tbl = pa.table({
        "row_key": rows_key,
        "candidate": rows_cand,
        "view_index": rows_view,
        "depth_l1": rows_dl1,
        "normal_l1": rows_nl1,
        "normal_dot": rows_ndot,
    })

    return root, cands, scores_tbl


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", type=Path, default=HERE / "build" / "bootstrap")
    ap.add_argument("--scores", type=Path, default=HERE / "build" / "scores")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv[1:])

    root, cands, scores_tbl = build_tables(a.pose_dir, a.scores)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    root_out = a.out_dir / "maskscore_rung_1_mesh.parquet"
    cand_out = a.out_dir / "maskscore_rung_1_mesh_candidates.parquet"
    scr_out = a.out_dir / "maskscore_rung_1_mesh_scores.parquet"
    pq.write_table(root, root_out, compression="zstd")
    pq.write_table(cands, cand_out, compression="zstd")
    pq.write_table(scores_tbl, scr_out, compression="zstd")

    print(f"ok rung 1 mesh (ETNF, 3 parquets):")
    print(f"  root       -> {root_out.name}  ({root.num_rows} row, {root.num_columns} cols)")
    print(f"  candidates -> {cand_out.name}  ({cands.num_rows} rows, {cands.num_columns} cols)")
    print(f"  scores     -> {scr_out.name}  ({scores_tbl.num_rows} rows, {scores_tbl.num_columns} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
