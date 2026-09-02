"""Rung 1 MaskScore — Depth, Pose, Keypoints, Multimodal stubs, ETNF three-parquet emit.

Same walking-skeleton bootstrap the Mesh stub uses (rest ANNY vs rank1 identity + rank5
perturbed), reshaped into four more of the eight MASKSCORE.md stubs. Each stub gets its
own root row + candidates satellite + scores satellite -- 12 new parquets total, all
ZSTD-compressed. The score metric is `render-and-compare depth_l1` per MASKSCORE.md's
"one universal metric" line; stubs differ in what is INPUT (depth image, SOMA pose, 2D
keypoints, cross-modal), not in what is scored.

Text, Speech, and Video stubs are deferred to Rung 2 -- bootstrap has no transcript, no
audio, and no video, and CLAUDE.md's ETNF rule forbids putting a null in for the missing
input.

The Mesh stub's identity + negative controls carry over: rank1 vs input scores near-zero
at machine precision, rank5 strictly worse. Both are asserted here before write.

Usage:
    pixi run --environment anny-mac python maskscore_rung_1_stubs.py \
        --pose-dir build/bootstrap \
        --scores build/scores \
        --renders build/bootstrap/renders \
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

# One row per stub in one place, so a schema drift shows up in one diff rather than
# five copy-pasted docstrings. The MASKSCORE.md task_type + dimension per stub follow
# Rung 0's assignments verbatim.
STUBS = [
    # (stub_name, task_type, dimension, input_column, input_asset_kind)
    ("depth",      "depth_edit",           "instruction_following", "input_depth",      "aov_npz"),
    ("pose",       "pose_change",          "instruction_following", "input_pose",       "soma_rotations"),
    ("keypoints",  "expression_change",    "instruction_following", "input_keypoints",  "keypoints_json"),
    ("multimodal", "cross_modal_compose",  "overall",               "input_data",       "png"),
]

# The two candidates the bootstrap carries. Kept in a shared list so all four stubs
# populate their candidates satellite identically -- rank1 identity vs rank5 perturbed.
CANDIDATES = [
    ("rank1", 1),
    ("rank5", 5),
]


def assert_controls(scores: dict[str, dict]) -> None:
    r1, r5 = scores["rank1"], scores["rank5"]
    max_r1 = max(v["depth_l1"] for v in r1["views"])
    if max_r1 > 1e-6:
        raise SystemExit(f"identity control failed: rank1 max depth_l1 {max_r1:.6e} > 1e-6")
    if r5["mean_depth_l1"] <= r1["mean_depth_l1"]:
        raise SystemExit(f"negative control failed: rank5 mean depth_l1 "
                         f"({r5['mean_depth_l1']:.6f}) not strictly worse than rank1 "
                         f"({r1['mean_depth_l1']:.6f})")


def real(p: Path) -> str:
    if not p.is_file():
        raise SystemExit(f"missing: {p}")
    return str(p.relative_to(HERE))


def input_asset_path(stub: str, pose_dir: Path, renders_dir: Path) -> str:
    """Where each stub's INPUT lives on disk. Bootstrap sources only, no SpeakingFaces."""
    if stub == "depth":
        return real(renders_dir / "input" / "view_000.aov.npz")
    if stub == "pose":
        # SOMA rotations live inside the rest.npz (pose_soma, translation). The .npz IS
        # the pose asset -- pointing at it here is more honest than emitting a second
        # copy of the rotations into a JSON.
        return real(pose_dir / "rest.npz")
    if stub == "keypoints":
        return real(renders_dir / "input" / "view_000.keypoints.json")
    if stub == "multimodal":
        return real(renders_dir / "input" / "view_000.png")
    raise SystemExit(f"unknown stub: {stub}")


def candidate_asset_path(stub: str, cand: str, pose_dir: Path, renders_dir: Path) -> str:
    if stub == "depth":
        return real(renders_dir / cand / "view_000.aov.npz")
    if stub == "pose":
        return real(pose_dir / f"{cand}.npz")
    if stub == "keypoints":
        return real(renders_dir / cand / "view_000.keypoints.json")
    if stub == "multimodal":
        # Candidate in the multimodal (image -> mesh) shape is a mesh, not another image.
        return real(pose_dir / f"{cand}.npz")
    raise SystemExit(f"unknown stub: {stub}")


def build_scores_table(key: str, scores: dict[str, dict]) -> pa.Table:
    """Universal render-and-compare metric per MASKSCORE.md. Same 128-row shape for every stub."""
    rows_key, rows_cand, rows_view = [], [], []
    rows_dl1, rows_nl1, rows_ndot = [], [], []
    for cand_name, _rank in CANDIDATES:
        for i, v in enumerate(scores[cand_name]["views"]):
            rows_key.append(key)
            rows_cand.append(cand_name)
            rows_view.append(i)
            rows_dl1.append(v["depth_l1"])
            rows_nl1.append(v["normal_l1"])
            rows_ndot.append(v["normal_dot"])
    return pa.table({
        "row_key": rows_key,
        "candidate": rows_cand,
        "view_index": rows_view,
        "depth_l1": rows_dl1,
        "normal_l1": rows_nl1,
        "normal_dot": rows_ndot,
    })


def build_stub(stub: str, task_type: str, dimension: str, input_col: str,
               input_kind: str, pose_dir: Path, renders_dir: Path, scores: dict,
               ) -> tuple[str, pa.Table, pa.Table, pa.Table]:
    key = f"rung1/bootstrap/{stub}"

    # ROOT ROW. `input_asset` + `input_asset_kind` name the input's on-disk path and
    # its type from an interned vocabulary. That keeps every stub's root row shape
    # identical -- one schema, five stubs -- rather than five bespoke column names.
    root = pa.table({
        "key": [key],
        "task_type": [task_type],
        "dimension": [dimension],
        "input_column": [input_col],
        "input_asset": [input_asset_path(stub, pose_dir, renders_dir)],
        "input_asset_kind": [input_kind],
        "poses": [real(pose_dir / "poses.json")],
    })

    # CANDIDATES satellite. Same shape as Mesh's.
    cands = pa.table({
        "row_key": [key] * len(CANDIDATES),
        "candidate": [c for c, _ in CANDIDATES],
        "rank": [r for _, r in CANDIDATES],
        "candidate_asset": [candidate_asset_path(stub, c, pose_dir, renders_dir)
                            for c, _ in CANDIDATES],
    })

    scores_tbl = build_scores_table(key, scores)
    return key, root, cands, scores_tbl


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", type=Path, default=HERE / "build" / "bootstrap")
    ap.add_argument("--scores", type=Path, default=HERE / "build" / "scores")
    ap.add_argument("--renders", type=Path, default=HERE / "build" / "bootstrap" / "renders")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv[1:])

    scores = {
        "rank1": json.loads((a.scores / "rank1.json").read_text()),
        "rank5": json.loads((a.scores / "rank5.json").read_text()),
    }
    assert_controls(scores)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"emitting 4 stubs, 12 parquets, ZSTD:")
    for stub, task_type, dim, input_col, input_kind in STUBS:
        _, root, cands, scr = build_stub(stub, task_type, dim, input_col, input_kind,
                                          a.pose_dir, a.renders, scores)
        pq.write_table(root,  a.out_dir / f"maskscore_rung_1_{stub}.parquet",              compression="zstd")
        pq.write_table(cands, a.out_dir / f"maskscore_rung_1_{stub}_candidates.parquet",   compression="zstd")
        pq.write_table(scr,   a.out_dir / f"maskscore_rung_1_{stub}_scores.parquet",       compression="zstd")
        print(f"  {stub:11s}  root {root.num_rows}x{root.num_columns}, "
              f"cand {cands.num_rows}x{cands.num_columns}, "
              f"scores {scr.num_rows}x{scr.num_columns}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
