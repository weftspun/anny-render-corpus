"""RFD 2197: join maskscore-rung-1-bootstrap into HF viewer-friendly configs."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


CONFIGS = {
    "depth":      {"prefix": ("maskscore_rung_1_depth",),
                   "cand_composite":  ("row_key", "candidate"),
                   "score_composite": ("row_key", "candidate")},
    "keypoints":  {"prefix": ("maskscore_rung_1_keypoints",),
                   "cand_composite":  ("row_key", "candidate"),
                   "score_composite": ("row_key", "candidate")},
    "mesh":       {"prefix": ("maskscore_rung_1_mesh",),
                   "cand_composite":  ("row_key", "candidate"),
                   "score_composite": ("row_key", "candidate")},
    "multimodal": {"prefix": ("maskscore_rung_1_multimodal",),
                   "cand_composite":  ("row_key", "candidate"),
                   "score_composite": ("row_key", "candidate")},
    "pose":       {"prefix": ("maskscore_rung_1_pose",),
                   "cand_composite":  ("row_key", "candidate"),
                   "score_composite": ("row_key", "candidate")},
    "speech":     {"prefix": ("speech", "speech", "maskscore_speech"),
                   "cand_composite":  ("row_key", "candidate_axis", "rank"),
                   "score_composite": ("row_key", "candidate_axis", "candidate_rank")},
}


def load_triple(source, prefix):
    stem = source.joinpath(*prefix)
    return (
        pq.read_table(f"{stem}.parquet").to_pandas(),
        pq.read_table(f"{stem}_candidates.parquet").to_pandas(),
        pq.read_table(f"{stem}_scores.parquet").to_pandas(),
    )


def verify_keys(base, cands, scores, cand_comp, score_comp):
    assert "key" in base.columns, f"base missing 'key': {list(base.columns)}"
    for col in cand_comp:
        assert col in cands.columns, f"cands missing {col!r}: {list(cands.columns)}"
    for col in score_comp:
        assert col in scores.columns, f"scores missing {col!r}: {list(scores.columns)}"
    b_keys, c_keys = set(base["key"]), set(cands["row_key"])
    assert b_keys == c_keys, f"base.key != cands.row_key: base={len(b_keys)}, cands={len(c_keys)}, common={len(b_keys & c_keys)}"
    cand_tuples = set(cands[list(cand_comp)].itertuples(index=False, name=None))
    score_tuples = set(scores[list(score_comp)].itertuples(index=False, name=None))
    orphans = score_tuples - cand_tuples
    assert not orphans, f"{len(orphans)} score composites not in cands: sample {list(orphans)[:3]}"


def build_wide(base, cands, scores, cand_comp, score_comp):
    verify_keys(base, cands, scores, cand_comp, score_comp)
    scores_grouped = scores.groupby(list(score_comp))
    score_drop, cand_drop = list(score_comp), ["row_key"]

    per_rk: dict = {}
    for _, r in cands.iterrows():
        rec = r.drop(labels=cand_drop).to_dict()
        key_vals = tuple(r[c] for c in cand_comp)
        try:
            g = scores_grouped.get_group(key_vals)
            rec["scores"] = g.drop(columns=score_drop).to_dict("records")
        except KeyError:
            rec["scores"] = []
        per_rk.setdefault(r["row_key"], []).append(rec)

    cand_series = pd.Series(per_rk, name="candidates")
    wide = base.set_index("key").join(cand_series, how="left")
    wide["candidates"] = wide["candidates"].apply(lambda v: v if isinstance(v, list) else [])
    return wide.reset_index()


def emit(wide, out_root, config):
    out_dir = out_root / "data" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pandas(wide, preserve_index=False), out_path, compression="zstd")
    return out_path


def run(source, out):
    for cfg, spec in CONFIGS.items():
        b, c, s = load_triple(source, spec["prefix"])
        wide = build_wide(b, c, s, spec["cand_composite"], spec["score_composite"])
        p = emit(wide, out, cfg)
        n_cand = int(sum(len(x) for x in wide["candidates"]))
        n_score = int(sum(len(sc["scores"]) for row in wide["candidates"] for sc in row))
        print(f"{cfg:<11}  rows={len(wide):>3}  cands={n_cand:>4}  scores={n_score:>5}  -> {p.relative_to(out)}")


def _fabricate(root):
    base = pd.DataFrame({
        "key": ["ex/a", "ex/b"], "task_type": ["toy"] * 2, "dimension": ["overall"] * 2,
        "input_column": ["input"] * 2, "input_asset": ["p/a.png", "p/b.png"],
        "input_asset_kind": ["png"] * 2, "poses": ["", ""],
    })
    cand = pd.DataFrame([
        {"row_key": k, "candidate": c, "rank": r, "candidate_asset": f"p/{k}/{c}.npz"}
        for k in ("ex/a", "ex/b") for c, r in (("rank1", 1), ("rank5", 5))
    ])
    scr = pd.DataFrame([
        {"row_key": k, "candidate": c, "view_index": v,
         "depth_l1": 0.0 if c == "rank1" else 0.5, "normal_l1": 0.0 if c == "rank1" else 0.5,
         "normal_dot": 1.0 if c == "rank1" else 0.5}
        for k in ("ex/a", "ex/b") for c in ("rank1", "rank5") for v in range(3)
    ])
    pq.write_table(pa.Table.from_pandas(base), root / "toy.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(cand), root / "toy_candidates.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(scr), root / "toy_scores.parquet", compression="zstd")

    (root / "sp" / "sp").mkdir(parents=True, exist_ok=True)
    sbase = pd.DataFrame({
        "key": ["sp/1"], "task_type": ["toy_speech"], "dimension": ["overall"],
        "input_column": ["input"], "input_asset": ["p/1.wav"],
        "input_asset_kind": ["wav_16khz"], "canonical_text": ["hi"],
    })
    scand = pd.DataFrame([
        {"row_key": "sp/1", "candidate_axis": "audio", "rank": r,
         "candidate_asset": f"p/1/{r}.wav", "candidate_asset_kind": "wav_16khz",
         "candidate_kind": "identity", "candidate_target_text": "hi"}
        for r in (1, 2)
    ])
    sscr = pd.DataFrame([
        {"row_key": "sp/1", "candidate_axis": "audio", "candidate_rank": r,
         "metric_name": m, "metric_value": 0.9 if m == "wavlm_cos" else 0.0}
        for r in (1, 2) for m in ("wavlm_cos", "wer")
    ])
    pq.write_table(pa.Table.from_pandas(sbase), root / "sp" / "sp" / "maskscore_toy_speech.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(scand), root / "sp" / "sp" / "maskscore_toy_speech_candidates.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(sscr), root / "sp" / "sp" / "maskscore_toy_speech_scores.parquet", compression="zstd")

    (root / "bad").mkdir(exist_ok=True)
    for name in ("toy.parquet", "toy_candidates.parquet"):
        pq.write_table(pq.read_table(root / name), root / "bad" / name)
    bad_scr = pd.concat([scr, pd.DataFrame([{
        "row_key": "ex/a", "candidate": "rank99", "view_index": 0,
        "depth_l1": 0.0, "normal_l1": 0.0, "normal_dot": 1.0,
    }])], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(bad_scr), root / "bad" / "toy_scores.parquet", compression="zstd")


def self_test():
    checks = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _fabricate(root)

        b, c, s = load_triple(root, ("toy",))
        wide = build_wide(b, c, s, ("row_key", "candidate"), ("row_key", "candidate"))
        checks.append(("wide join rows=2", len(wide) == 2))
        checks.append(("wide join cand_count=4", sum(len(x) for x in wide["candidates"]) == 4))
        checks.append(("wide join score_count=12", sum(len(sc["scores"]) for r in wide["candidates"] for sc in r) == 12))
        rec = wide.iloc[0]["candidates"][0]
        checks.append(("cand carries no row_key", "row_key" not in rec))
        checks.append(("score carries no join cols", set(rec["scores"][0].keys()) == {"view_index", "depth_l1", "normal_l1", "normal_dot"}))

        b, c, s = load_triple(root, ("sp", "sp", "maskscore_toy_speech"))
        wide = build_wide(b, c, s, ("row_key", "candidate_axis", "rank"), ("row_key", "candidate_axis", "candidate_rank"))
        checks.append(("speech join rows=1", len(wide) == 1))
        checks.append(("speech join cand_count=2", sum(len(x) for x in wide["candidates"]) == 2))
        checks.append(("speech join score_count=4", sum(len(sc["scores"]) for r in wide["candidates"] for sc in r) == 4))

        p = emit(wide, root / "out", "toy_speech")
        checks.append(("emit round-trip preserves rows", pq.read_table(str(p)).num_rows == 1))

        b, c, s = load_triple(root / "bad", ("toy",))
        raised = False
        try:
            build_wide(b, c, s, ("row_key", "candidate"), ("row_key", "candidate"))
        except AssertionError:
            raised = True
        checks.append(("orphan score rejected", raised))

    ok = sum(1 for _, v in checks if v)
    for name, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {name}")
    print(f"{ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.source or not args.out:
        ap.error("--source and --out required unless --self-test")
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.source, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
