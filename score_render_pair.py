"""Score a candidate render directory against a reference render directory, per view.

For every matched pair of `view_XXX.aov.npz` files, compute:

  depth_l1   mean absolute depth difference on pixels that hit the body in the reference
  normal_l1  mean absolute normal difference on covered pixels
  normal_dot mean dot product of unit normals on covered pixels (1.0 = identical direction)

LPIPS is deferred to a later rung (needs a downloaded model and does not fit the walking
skeleton budget).

Negative control per rule 2: `rank1` is identity by construction, and this script asserts
its self-scores are within machine precision of zero. `rank5` must score strictly worse than
`rank1` on at least one metric per view; the script prints a summary that lets a downstream
gate reject a pair where the negative control failed to detect the difference.

Usage:
    pixi run --environment anny-mac python score_render_pair.py \
        --reference <ref_dir> --candidate <cand_dir> --out <scores.json>
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image


def load_aov(view_json: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth (H,W), normal (H,W,3), alpha (H,W)) from a render sidecar."""
    aov_npz = view_json.with_suffix(".aov.npz")
    z = np.load(aov_npz)
    depth = z["depth"].astype(np.float32)
    normal = z["normal"].astype(np.float32)
    # Alpha from the PNG — RGBA, 4th channel.
    png = view_json.with_suffix(".png")
    rgba = np.asarray(Image.open(png).convert("RGBA")).astype(np.float32) / 255.0
    alpha = rgba[:, :, 3]
    return depth, normal, alpha


def score_pair(ref_dir: pathlib.Path, cand_dir: pathlib.Path) -> list[dict]:
    ref_views = sorted([p for p in ref_dir.glob("view_*.json") if ".keypoints" not in p.name])

    if not ref_views:
        raise SystemExit(f"no view_*.json in {ref_dir}")
    scores = []
    for r_json in ref_views:
        c_json = cand_dir / r_json.name
        if not c_json.exists():
            raise SystemExit(f"missing candidate view: {c_json}")

        r_depth, r_normal, r_alpha = load_aov(r_json)
        c_depth, c_normal, _ = load_aov(c_json)

        # Covered mask: pixel hits the body in the REFERENCE. Scoring against unrelated
        # background pixels dilutes the metric with a constant.
        mask = r_alpha > 0.5
        n_covered = int(mask.sum())
        if n_covered == 0:
            scores.append({"view": r_json.stem, "n_covered": 0,
                           "depth_l1": 0.0, "normal_l1": 0.0, "normal_dot": 1.0})
            continue

        depth_diff = np.abs(r_depth[mask] - c_depth[mask])
        normal_diff = r_normal[mask] - c_normal[mask]

        # Renormalise normals (they may have length != 1 after Mitsuba's box filter avg).
        rn = r_normal[mask]
        cn = c_normal[mask]
        rn = rn / (np.linalg.norm(rn, axis=-1, keepdims=True) + 1e-9)
        cn = cn / (np.linalg.norm(cn, axis=-1, keepdims=True) + 1e-9)
        cos = np.clip((rn * cn).sum(axis=-1), -1.0, 1.0)

        scores.append({
            "view": r_json.stem,
            "n_covered": n_covered,
            "depth_l1": float(depth_diff.mean()),
            "normal_l1": float(np.abs(normal_diff).mean()),
            "normal_dot": float(cos.mean()),
        })
    return scores


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=pathlib.Path, required=True)
    ap.add_argument("--candidate", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--assert-identity", action="store_true",
                    help="fail if any view's depth_l1 exceeds 1e-6 (self-score control)")
    a = ap.parse_args(argv[1:])

    scores = score_pair(a.reference, a.candidate)
    summary = {
        "reference": str(a.reference),
        "candidate": str(a.candidate),
        "n_views": len(scores),
        "mean_depth_l1": float(np.mean([s["depth_l1"] for s in scores])),
        "mean_normal_l1": float(np.mean([s["normal_l1"] for s in scores])),
        "mean_normal_dot": float(np.mean([s["normal_dot"] for s in scores])),
        "views": scores,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))

    print(f"  {a.candidate.name} vs {a.reference.name}:")
    print(f"    mean depth L1 : {summary['mean_depth_l1']:.6f}  (unit-cube scene units)")
    print(f"    mean normal L1: {summary['mean_normal_l1']:.6f}")
    print(f"    mean normal .:  {summary['mean_normal_dot']:.6f}  (1.0 = identical)")

    if a.assert_identity:
        max_dl1 = max(s["depth_l1"] for s in scores)
        if max_dl1 > 1e-6:
            raise SystemExit(f"identity-control failed: max depth L1 {max_dl1:.6f} > 1e-6")
        print(f"  identity control PASSED (max depth L1 {max_dl1:.6e})")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
