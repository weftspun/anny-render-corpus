"""Rung 1 MaskScore — Mesh stub only, one row, real content.

Walking-skeleton Rung 1: rest ANNY pose is the input, rank1 is identity (== input),
rank5 is the same identity with a small random pose perturbation. Rank1 and rank5 are
both rendered over 64 sphere_hammersley views with depth + normal AOVs and scored
against the input by `score_render_pair.py`. The per-view scores make up the row's
`scores` list; rank1 must score at machine precision (identity control) and rank5
strictly worse (negative control, rule 2). This is the walking skeleton for MaskScore
Rung 1 — a single row with real numbers, provable end-to-end. The remaining seven
stubs come once persona is back to align the ANNY-fit and SpeakingFaces halves.

Output goes to `maskscore_rung_1_mesh.parquet` in ZStandard compression. ETNF: no
nulls, no derivable columns. The `key` names this trial + stub uniquely so a future
per-stub parquet family can join back cleanly.

Usage:
    pixi run --environment anny-mac python maskscore_rung_1_mesh.py \
        --pose-dir build/bootstrap \
        --scores build/scores \
        --out maskscore_rung_1_mesh.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent


def build_row(pose_dir: Path, scores_dir: Path) -> dict:
    poses_meta = json.loads((pose_dir / "poses.json").read_text())
    rank1 = json.loads((scores_dir / "rank1.json").read_text())
    rank5 = json.loads((scores_dir / "rank5.json").read_text())

    # ETNF: every path we cite must exist. Assert here rather than let the parquet
    # writer emit a row that points at nothing.
    def real(p: Path) -> str:
        if not p.is_file():
            raise SystemExit(f"missing: {p}")
        return str(p.relative_to(HERE))
    input_mesh = real(pose_dir / "rest.npz")
    rank1_mesh = real(pose_dir / "rank1.npz")
    rank5_mesh = real(pose_dir / "rank5.npz")
    poses_json = real(pose_dir / "poses.json")

    # Scores list: rank1 depth_l1 across all 64 views, then rank5 depth_l1 across all 64
    # views. The negative control (rank5 > rank1 on this metric) is asserted before we
    # write, so a corpus row cannot ship past a broken metric.
    rank1_view_scores = [v["depth_l1"] for v in rank1["views"]]
    rank5_view_scores = [v["depth_l1"] for v in rank5["views"]]
    if len(rank1_view_scores) != 64 or len(rank5_view_scores) != 64:
        raise SystemExit(f"expected 64 views each, got rank1={len(rank1_view_scores)}, "
                         f"rank5={len(rank5_view_scores)}")
    if max(rank1_view_scores) > 1e-6:
        raise SystemExit(f"identity control failed: rank1 max depth_l1 "
                         f"{max(rank1_view_scores):.6e} > 1e-6")
    if rank5["mean_depth_l1"] <= rank1["mean_depth_l1"]:
        raise SystemExit(f"negative control failed: rank5 mean depth_l1 "
                         f"({rank5['mean_depth_l1']:.6f}) not strictly worse than rank1 "
                         f"({rank1['mean_depth_l1']:.6f})")

    return {
        "key": "rung1/bootstrap/mesh",
        "instruction": "identity edit (walking-skeleton bootstrap, rest anny pose)",
        "task_type": "pose_change",
        "dimension": "instruction_following",
        "scores": rank1_view_scores + rank5_view_scores,
        "input_mesh": input_mesh,
        "conditioning_image": "build/bootstrap/renders/input/view_000.png",
        "output_meshes": [rank1_mesh, rank5_mesh],
        "poses": poses_json,
        "n_views": 64,
        "rank1_mean_depth_l1": rank1["mean_depth_l1"],
        "rank5_mean_depth_l1": rank5["mean_depth_l1"],
        "rank1_mean_normal_l1": rank1["mean_normal_l1"],
        "rank5_mean_normal_l1": rank5["mean_normal_l1"],
        "perturbation_sigma_rad": poses_meta["perturbation_sigma_rad"],
        "seed": poses_meta["seed"],
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", type=Path, default=HERE / "build" / "bootstrap")
    ap.add_argument("--scores", type=Path, default=HERE / "build" / "scores")
    ap.add_argument("--out", type=Path, default=HERE / "maskscore_rung_1_mesh.parquet")
    a = ap.parse_args(argv[1:])

    row = build_row(a.pose_dir, a.scores)
    # One column per key, one row.
    columns = {k: [v] for k, v in row.items()}
    table = pa.table(columns)
    pq.write_table(table, a.out, compression="zstd")

    print(f"ok rung 1 mesh: 1 row -> {a.out}")
    print(f"  rank1 mean depth L1: {row['rank1_mean_depth_l1']:.6e}  (identity control)")
    print(f"  rank5 mean depth L1: {row['rank5_mean_depth_l1']:.6f}  (negative control)")
    print(f"  perturbation: sigma {row['perturbation_sigma_rad']:.4f} rad, seed {row['seed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
