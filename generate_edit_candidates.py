"""Generate frame A, frame B, and 5 ranked candidates for a MaskScore edit row.

MASKSCORE.md `## Edits are frame pairs, not perturbations` names the exact structure:
frame A (input) + frame B (target) + five graded candidates. This script instantiates
that on the ANNY bootstrap path -- no SpeakingFaces fit needed -- by using ANNY facial
actions to synthesise A and B and derive the five candidate meshes from them.

The edit driven here is a jawOpen action. See-Through's part vocabulary is deliberately
not used to label the parts: a downstream VLM step describes what changed in its own
words, and that description becomes the row's `instruction` column. The rank labels
below name the CANDIDATE'S relationship to (A, B), not the taxonomy of parts.

  frame_a.npz  rest pose (all facial actions zero)
  frame_b.npz  jawOpen = 1.0  -- the edit target
  rank1.npz    jawOpen = 1.0  -- correct execution of the edit (== frame_b)
  rank2.npz    jawOpen = 0.5  -- 50-percent interpolation between A and B
  rank3.npz    jawOpen = 0.3  -- partial severity, the edit is under-committed
  rank4.npz    eyeBlinkLeft = eyeBlinkRight = 1.0  -- the WRONG facial action perturbed
  rank5.npz    different phenotype (gender flipped) + jawOpen = 1.0  -- wrong subject

Every candidate is stored with its full 78-bone SOMA rotations (float64, zero for these
walking-skeleton meshes -- the deltas live in facial_actions) plus the facial_actions
vector and the phenotype used, so a downstream consumer can reproduce any candidate from
its record rather than only from this script's seed.

Usage:
    pixi run --environment anny-mac python generate_edit_candidates.py \
        [--out-dir build/edit] [--phenotype-b-attribute gender] [--phenotype-b-value 1.0]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent


def _run(model, facial_actions=None, phenotype_kwargs=None):
    kw = {}
    if facial_actions is not None:
        kw["facial_actions"] = facial_actions
    if phenotype_kwargs is not None:
        kw["phenotype_kwargs"] = phenotype_kwargs
    with torch.no_grad():
        out = model(**kw) if kw else model()
    verts_key = "vertices" if "vertices" in out else "rest_vertices"
    return out[verts_key][0].numpy().astype(np.float64)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE / "build" / "edit")
    ap.add_argument("--target-action", default="jawOpen",
                    help="ANNY facial action driving the A->B edit (default jawOpen)")
    ap.add_argument("--wrong-actions", nargs="+",
                    default=["eyeBlinkLeft", "eyeBlinkRight"],
                    help="actions activated for rank4 (wrong part edited)")
    ap.add_argument("--phenotype-swap-attribute", default="gender",
                    help="phenotype dim to flip for rank5 (wrong subject)")
    ap.add_argument("--phenotype-swap-value", type=float, default=1.0)
    a = ap.parse_args(argv[1:])

    a.out_dir.mkdir(parents=True, exist_ok=True)

    from anny import Anny
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref",
                 facial_actions="all")
    model.eval()

    fa_labels = model.facial_action_labels
    if a.target_action not in fa_labels:
        raise SystemExit(f"target action {a.target_action!r} not in {len(fa_labels)} labels")
    for w in a.wrong_actions:
        if w not in fa_labels:
            raise SystemExit(f"wrong action {w!r} not in {len(fa_labels)} labels")

    phen_labels = model.phenotype_labels
    if a.phenotype_swap_attribute not in phen_labels:
        raise SystemExit(f"phenotype attribute {a.phenotype_swap_attribute!r} "
                         f"not in {phen_labels}")

    faces = np.asarray(model.faces, dtype=np.int64)

    def actions(**kw):
        return {label: float(kw.get(label, 0.0)) for label in fa_labels}

    def phenotype(**overrides):
        return {label: torch.tensor([overrides.get(label, 0.5)], dtype=torch.float64)
                for label in phen_labels}

    def to_tensor(d):
        return {k: torch.tensor([v], dtype=torch.float64) for k, v in d.items()}

    # (name, facial_actions_dict, phenotype_kwargs, note)
    specs = [
        ("frame_a", {},                                   None,
         "rest pose, all facial actions zero, phenotype = adult average (all 0.5)"),
        ("frame_b", {a.target_action: 1.0},               None,
         f"target: {a.target_action} = 1.0"),
        ("rank1",   {a.target_action: 1.0},               None,
         f"correct edit: {a.target_action} = 1.0 (equal to frame_b)"),
        ("rank2",   {a.target_action: 0.5},               None,
         f"interpolation halfway between A and B: {a.target_action} = 0.5"),
        ("rank3",   {a.target_action: 0.3},               None,
         f"partial severity: {a.target_action} = 0.3 (under-committed)"),
        ("rank4",   {w: 1.0 for w in a.wrong_actions},    None,
         f"wrong facial action perturbed: {a.wrong_actions}"),
        ("rank5",   {a.target_action: 1.0},
                    {a.phenotype_swap_attribute: a.phenotype_swap_value},
         f"wrong subject: same edit ({a.target_action}=1.0) but phenotype "
         f"{a.phenotype_swap_attribute}={a.phenotype_swap_value}"),
    ]

    manifest = {"target_action": a.target_action,
                "wrong_actions": a.wrong_actions,
                "phenotype_swap_attribute": a.phenotype_swap_attribute,
                "phenotype_swap_value": a.phenotype_swap_value,
                "phenotype_baseline": 0.5,
                "candidates": {}}

    for name, fa_kw, phen_over, note in specs:
        fa_dict = actions(**fa_kw)
        phen_kw = phenotype(**(phen_over or {}))
        verts = _run(model, facial_actions=to_tensor(fa_dict), phenotype_kwargs=phen_kw)
        fa_vec = np.array([fa_dict[l] for l in fa_labels], dtype=np.float64)
        phen_vec = np.array([phen_kw[l].item() for l in phen_labels], dtype=np.float64)
        p = a.out_dir / f"{name}.npz"
        np.savez_compressed(p, verts=verts, faces=faces,
                            facial_actions=fa_vec,
                            facial_action_labels=np.array(fa_labels),
                            phenotype=phen_vec,
                            phenotype_labels=np.array(phen_labels))
        manifest["candidates"][name] = {
            "path": p.name, "note": note,
            "facial_actions_nonzero": {k: v for k, v in fa_dict.items() if v != 0.0},
            "phenotype_nondefault": {k: phen_kw[k].item() for k in phen_labels
                                     if phen_kw[k].item() != 0.5},
        }
        print(f"  {name}: {note}  verts={verts.shape}")

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {len(specs)} meshes + manifest.json to {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
