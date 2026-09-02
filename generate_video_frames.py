"""Generate temporal 5-frame mesh sequences for the Video stub.

Each of the 10 edits has 5 candidates (rank1..rank5). The Video stub renders each
candidate as a temporal sequence A -> t=0.25 -> t=0.5 -> t=0.75 -> B where t linearly
interpolates the facial_actions vector between frame A (rest, all zeros) and the
candidate's endpoint. rank1's endpoint is frame_b; rank5's is frame_b with a phenotype
swap. Total: 10 edits x 5 candidates x 5 frames = 250 meshes.

Output layout:
  build/edit/video/<edit>/<candidate>/frame_0.npz  (t=0.0, same as frame_a)
  build/edit/video/<edit>/<candidate>/frame_1.npz  (t=0.25)
  ...
  build/edit/video/<edit>/<candidate>/frame_4.npz  (t=1.0, same as build/edit/<edit>/<candidate>.npz)

Frame 0 is identical across all candidates (rest); frame 4 is the still-image
candidate we already have. Both are re-emitted here for schema uniformity -- 250 npz
files, one per (edit, candidate, t). A downstream consumer walks the flat list.

Usage:
    pixi run --environment anny-mac python generate_video_frames.py [--n-frames 5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = ["rank1", "rank2", "rank3", "rank4", "rank5"]


def actions_at_t(target_actions: dict[str, float], t: float) -> dict[str, float]:
    """Linear interp of a rest-anchored action vector: t=0 -> zeros, t=1 -> target."""
    return {k: float(v * t) for k, v in target_actions.items()}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir", type=pathlib.Path, default=HERE / "build" / "edit")
    ap.add_argument("--out-dir",  type=pathlib.Path, default=HERE / "build" / "edit" / "video")
    ap.add_argument("--n-frames", type=int, default=5)
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)

    from anny import Anny
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref",
                 facial_actions="all")
    model.eval()

    fa_labels = model.facial_action_labels
    phen_labels = model.phenotype_labels
    faces = np.asarray(model.faces, dtype=np.int64)

    def to_action_tensor(d):
        return {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in d.items()}

    def phen_tensor(**overrides):
        return {l: torch.tensor([overrides.get(l, 0.5)], dtype=torch.float64) for l in phen_labels}

    def run(fa_dict, phen_kw):
        with torch.no_grad():
            out = model(facial_actions=to_action_tensor(fa_dict), phenotype_kwargs=phen_kw)
        return out["vertices"][0].numpy().astype(np.float64)

    baseline_phen = phen_tensor()
    ts = [i / (a.n_frames - 1) for i in range(a.n_frames)]  # [0.0, 0.25, 0.5, 0.75, 1.0]
    swap_attr = manifest["phenotype_swap_attribute"]
    swap_val = manifest["phenotype_swap_value"]

    manifest_out = {"n_frames": a.n_frames, "ts": ts,
                    "phenotype_swap_attribute": swap_attr,
                    "phenotype_swap_value": swap_val,
                    "edits": {}}
    total = 0
    for edit_name, edit_spec in manifest["edits"].items():
        target = edit_spec["target_actions"]
        wrong = edit_spec["wrong_actions"]
        edit_out = {}
        for cand in CANDIDATES:
            cand_out = a.out_dir / edit_name / cand
            cand_out.mkdir(parents=True, exist_ok=True)
            # Choose the actions and phenotype for this candidate's endpoint.
            if cand == "rank1":
                endpoint_actions, endpoint_phen = target, baseline_phen
            elif cand == "rank2":
                endpoint_actions, endpoint_phen = {k: v * 0.5 for k, v in target.items()}, baseline_phen
            elif cand == "rank3":
                endpoint_actions, endpoint_phen = {k: v * 0.3 for k, v in target.items()}, baseline_phen
            elif cand == "rank4":
                endpoint_actions, endpoint_phen = wrong, baseline_phen
            elif cand == "rank5":
                endpoint_actions, endpoint_phen = target, phen_tensor(**{swap_attr: swap_val})
            for i, t in enumerate(ts):
                fa_at_t = actions_at_t(endpoint_actions, t)
                verts = run(fa_at_t, endpoint_phen)
                np.savez_compressed(cand_out / f"frame_{i}.npz",
                                    verts=verts, faces=faces,
                                    t=np.float64(t),
                                    facial_actions=np.array([fa_at_t.get(l, 0.0)
                                                             for l in fa_labels], dtype=np.float64),
                                    phenotype=np.array([endpoint_phen[l].item()
                                                        for l in phen_labels], dtype=np.float64))
                total += 1
            edit_out[cand] = {"endpoint_actions": endpoint_actions,
                              "endpoint_phenotype_nondefault":
                                  {swap_attr: swap_val} if cand == "rank5" else {}}
        manifest_out["edits"][edit_name] = edit_out
        print(f"  {edit_name}: {len(CANDIDATES) * a.n_frames} meshes")

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"done. {total} meshes ({len(manifest['edits'])} edits x {len(CANDIDATES)} candidates "
          f"x {a.n_frames} frames) in {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
