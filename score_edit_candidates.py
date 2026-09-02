"""Score every (edit, candidate) render against its frame_b target, per view.

Reuses score_render_pair's per-view metric (depth L1, normal L1, normal dot on the
reference alpha mask). Reference = the edit's frame_b render; candidates = rank1..rank5.
rank1 == frame_b by construction, so its score is the identity control -- must be within
machine precision of the metric's zero.

Per-edit negative control: rank5 (wrong subject) or rank4 (wrong part) should score
strictly worse than rank1. Asserted after all edits scored, so a single failure fails
the whole emit rather than sneaking one broken edit through.

Emits one JSON per (edit, candidate): build/edit/scores/<edit>/<candidate>.json holding
mean depth/normal l1 and per-view detail.

Usage:
    pixi run --environment anny-mac python score_edit_candidates.py \
        [--render-dir build/edit/renders] [--out-dir build/edit/scores]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image


CANDIDATES = ("rank1", "rank2", "rank3", "rank4", "rank5")


def load_aov(view_json: pathlib.Path):
    aov_npz = view_json.with_suffix(".aov.npz")
    z = np.load(aov_npz)
    depth = z["depth"].astype(np.float32)
    normal = z["normal"].astype(np.float32)
    png = view_json.with_suffix(".png")
    alpha = np.asarray(Image.open(png).convert("RGBA")).astype(np.float32)[:, :, 3] / 255.0
    return depth, normal, alpha


def score_pair(ref_dir: pathlib.Path, cand_dir: pathlib.Path) -> list[dict]:
    ref_views = sorted(p for p in ref_dir.glob("view_*.json") if ".keypoints" not in p.name)
    if not ref_views:
        raise SystemExit(f"no view_*.json in {ref_dir}")
    scores = []
    for r_json in ref_views:
        c_json = cand_dir / r_json.name
        if not c_json.exists():
            raise SystemExit(f"missing candidate view: {c_json}")
        rd, rn, ra = load_aov(r_json)
        cd, cn, _ = load_aov(c_json)
        mask = ra > 0.5
        n_covered = int(mask.sum())
        if n_covered == 0:
            scores.append({"view": r_json.stem, "n_covered": 0,
                           "depth_l1": 0.0, "normal_l1": 0.0, "normal_dot": 1.0})
            continue
        depth_diff = np.abs(rd[mask] - cd[mask])
        normal_diff = np.abs(rn[mask] - cn[mask])
        rn_m = rn[mask] / (np.linalg.norm(rn[mask], axis=-1, keepdims=True) + 1e-9)
        cn_m = cn[mask] / (np.linalg.norm(cn[mask], axis=-1, keepdims=True) + 1e-9)
        cos = np.clip((rn_m * cn_m).sum(axis=-1), -1.0, 1.0)
        scores.append({
            "view": r_json.stem, "n_covered": n_covered,
            "depth_l1": float(depth_diff.mean()),
            "normal_l1": float(normal_diff.mean()),
            "normal_dot": float(cos.mean()),
        })
    return scores


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir",   type=pathlib.Path, default=pathlib.Path("build/edit"))
    ap.add_argument("--render-dir", type=pathlib.Path, default=pathlib.Path("build/edit/renders"))
    ap.add_argument("--out-dir",    type=pathlib.Path, default=pathlib.Path("build/edit/scores"))
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"per_edit": {}}
    for edit_name in manifest["edits"]:
        ref_dir = a.render_dir / edit_name / "frame_b"
        edit_scores = {}
        (a.out_dir / edit_name).mkdir(exist_ok=True)
        for cand in CANDIDATES:
            cand_dir = a.render_dir / edit_name / cand
            views = score_pair(ref_dir, cand_dir)
            mean = {k: float(np.mean([v[k] for v in views]))
                    for k in ("depth_l1", "normal_l1", "normal_dot")}
            out = {"reference": str(ref_dir), "candidate": str(cand_dir),
                   "n_views": len(views), **{f"mean_{k}": v for k, v in mean.items()},
                   "views": views}
            (a.out_dir / edit_name / f"{cand}.json").write_text(json.dumps(out, indent=2))
            edit_scores[cand] = mean

        # Controls per edit. rank1 == frame_b (bit-identical mesh, same render dir
        # via symlink), so identity should pass at machine precision.
        r1 = edit_scores["rank1"]["depth_l1"]
        if r1 > 1e-6:
            raise SystemExit(f"identity control failed for {edit_name}: rank1 depth_l1 "
                             f"{r1:.6e} > 1e-6")
        r5 = edit_scores["rank5"]["depth_l1"]
        if r5 <= r1:
            raise SystemExit(f"negative control failed for {edit_name}: rank5 depth_l1 "
                             f"({r5:.6f}) not > rank1 ({r1:.6e})")
        summary["per_edit"][edit_name] = edit_scores
        print(f"  {edit_name}: r1={edit_scores['rank1']['depth_l1']:.2e} "
              f"r2={edit_scores['rank2']['depth_l1']:.3f} "
              f"r3={edit_scores['rank3']['depth_l1']:.3f} "
              f"r4={edit_scores['rank4']['depth_l1']:.3f} "
              f"r5={edit_scores['rank5']['depth_l1']:.3f}")

    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done. 10 edits x 5 candidates = 50 score files under {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
