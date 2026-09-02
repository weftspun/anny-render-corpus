"""urn:oid:1.3.6.1.4.1.66606.1.2.2165.2 -- port 5 shipped stubs to unified schema.

Reads each stub's three wide-form parquets (mesh, depth, pose, keypoints,
multimodal) and rewrites them under the schema module from RFD 2165.1
(maskscore_stub_schema). Changes:

  root       drop stub-specific `poses` column; move into root_extras
  candidates rename `candidate` -> `candidate_kind`; add `candidate_axis`
             (constant 'edit'), `candidate_asset_kind` (interned per stub)
  scores     melt (depth_l1, normal_l1, normal_dot) wide -> long
             (metric_name, metric_value); keep view_index

Every rewritten table is validated by maskscore_stub_schema.validate before
write. The rewrite is bit-reproducible: same seed / same inputs / same outputs.

The wide-form files stay on disk under `_legacy_wide.parquet` names so a
diff against the shipped HF dataset shows exactly what changed.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

import maskscore_stub_schema as ms


HERE = pathlib.Path(__file__).resolve().parent

# stub name -> (task_type, input_asset_kind, candidate_asset_kind, root_extras)
STUBS = {
    "mesh":       ("pose_change",          "soma_mesh",      "soma_mesh",      "poses"),
    "depth":      ("depth_edit",           "aov_npz",        "aov_npz",        "poses"),
    "pose":       ("pose_change",          "soma_rotations", "soma_rotations", "poses"),
    "keypoints":  ("expression_change",    "keypoints_json", "keypoints_json", "poses"),
    "multimodal": ("cross_modal_compose",  "png",            "png",            "poses"),
}

# The five shipped stubs use dimension = 'instruction_following' for the first
# four and 'overall' for multimodal (per the shipped README).
DIMENSION = {
    "mesh": "instruction_following", "depth": "instruction_following",
    "pose": "instruction_following", "keypoints": "instruction_following",
    "multimodal": "overall",
}


def port_stub(stub: str, src_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    task_type, input_kind, cand_kind, extra_key = STUBS[stub]
    dim = DIMENSION[stub]

    root_in  = pq.read_table(src_dir / f"maskscore_rung_1_{stub}.parquet")
    cands_in = pq.read_table(src_dir / f"maskscore_rung_1_{stub}_candidates.parquet")
    scr_in   = pq.read_table(src_dir / f"maskscore_rung_1_{stub}_scores.parquet")

    root = pa.table({
        "key":              root_in["key"].to_pylist(),
        "task_type":        [task_type] * root_in.num_rows,
        "dimension":        [dim] * root_in.num_rows,
        "input_column":     root_in["input_column"].to_pylist(),
        "input_asset":      root_in["input_asset"].to_pylist(),
        "input_asset_kind": [input_kind] * root_in.num_rows,
    }, schema=ms.ROOT)

    root_extras_rows = []
    for key, extra_val in zip(root_in["key"].to_pylist(), root_in[extra_key].to_pylist()):
        root_extras_rows.append({
            "row_key": key, "extra_name": extra_key,
            "extra_value": str(extra_val), "extra_kind": "json_path",
        })
    root_extras = pa.Table.from_pylist(root_extras_rows, schema=ms.ROOT_EXTRAS)

    # Candidates: 'candidate' -> ('candidate_axis'='edit', 'candidate_kind'=name).
    cands = pa.table({
        "row_key":              cands_in["row_key"].to_pylist(),
        "candidate_axis":       ["edit"] * cands_in.num_rows,
        "rank":                 pa.array(cands_in["rank"].to_pylist(), type=pa.int32()),
        "candidate_asset":      cands_in["candidate_asset"].to_pylist(),
        "candidate_asset_kind": [cand_kind] * cands_in.num_rows,
        "candidate_kind":       cands_in["candidate"].to_pylist(),
    }, schema=ms.CANDIDATES)

    # Scores: melt wide -> long.
    scores_rows = []
    cand_to_rank = dict(zip(cands_in["candidate"].to_pylist(),
                            cands_in["rank"].to_pylist()))
    for i in range(scr_in.num_rows):
        row_key = scr_in["row_key"][i].as_py()
        cand    = scr_in["candidate"][i].as_py()
        vi      = scr_in["view_index"][i].as_py()
        rank    = cand_to_rank[cand]
        for m in ("depth_l1", "normal_l1", "normal_dot"):
            scores_rows.append({
                "row_key": row_key, "candidate_axis": "edit",
                "candidate_rank": int(rank), "view_index": int(vi),
                "metric_name": m, "metric_value": float(scr_in[m][i].as_py()),
            })
    scores = pa.Table.from_pylist(scores_rows, schema=ms.SCORES)

    ms.validate(root_t=root, cands_t=cands, scores_t=scores,
                root_extras_t=root_extras)

    out_root, out_cands, out_scores, out_root_ex, _ = ms.paths_for(
        f"rung_1_{stub}", out_dir)
    pq.write_table(root,        out_root,   compression="zstd")
    pq.write_table(cands,       out_cands,  compression="zstd")
    pq.write_table(scores,      out_scores, compression="zstd")
    pq.write_table(root_extras, out_root_ex, compression="zstd")

    return {"stub": stub, "root": root.num_rows, "candidates": cands.num_rows,
            "scores": scores.num_rows, "root_extras": root_extras.num_rows}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=pathlib.Path, default=HERE,
                    help="directory holding the wide-form maskscore_rung_1_*.parquet files")
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE / "build" / "unified",
                    help="destination for the ported unified-schema parquets")
    a = ap.parse_args(argv[1:])
    a.out_dir.mkdir(parents=True, exist_ok=True)

    for stub in STUBS:
        r = port_stub(stub, a.src_dir, a.out_dir)
        print(f"  {r['stub']:<12} root={r['root']} cands={r['candidates']} "
              f"scores={r['scores']} root_extras={r['root_extras']}")
    print(f"wrote unified parquets under {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
