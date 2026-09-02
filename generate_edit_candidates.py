"""Generate frame A and 10 edit families (frame_b + 5 candidates each) for MaskScore.

MASKSCORE.md `## Edits are frame pairs, not perturbations` defines: frame A (input),
frame B (target), five graded candidates per edit. This script instantiates that on the
ANNY bootstrap path with M=10 in-distribution single-FACS-action edits — one blendshape
at strength 1.0 per edit, drawn from ANNY's 52-action ARKit-compatible set. No random
multi-action combinations; every frame B is a real, anatomically-plausible expression.

The 10 edits (see EDITS below) span mouth + brow + eye + cheek + nose regions. Each
edit gets its own directory under out_dir/<edit_name>/ containing frame_b + 5 ranks;
frame A lives once at out_dir/frame_a.npz because it is the same rest pose for every
edit.

Rank labels per MASKSCORE.md's 5-rank spec:
  rank1: correct edit — frame B's exact facial_actions vector
  rank2: 50% interpolation between A and B
  rank3: 30% interpolation (under-committed)
  rank4: a DIFFERENT edit's target (anatomically distant, see WRONG_PART_MAP)
  rank5: same edit + phenotype gender flipped to 1.0 (wrong subject)

Every mesh stores its facial_actions vector and phenotype in the npz so a downstream
consumer reproduces any candidate from its record rather than from this script.

Usage:
    pixi run --environment anny-mac python generate_edit_candidates.py \
        [--out-dir build/edit] [--phenotype-swap-attribute gender] [--swap-value 1.0]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent


# In-distribution edits: each is one FACS blendshape at strength 1.0 (or a symmetric pair
# where the action naturally moves symmetric parts). Coverage: 4 mouth + 2 brow + 2 eye +
# 1 cheek + 1 nose = 10 anatomical regions. No random combinations.
EDITS = [
    ("jaw_open",         {"jawOpen": 1.0}),
    ("smile",            {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0}),
    ("pucker",           {"mouthPucker": 1.0}),
    ("funnel",           {"mouthFunnel": 1.0}),
    ("brow_up",          {"browInnerUp": 1.0}),
    ("brow_down",        {"browDownLeft": 1.0, "browDownRight": 1.0}),
    ("eye_blink",        {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}),
    ("eye_wide",         {"eyeWideLeft": 1.0, "eyeWideRight": 1.0}),
    ("cheek_puff",       {"cheekPuff": 1.0}),
    ("nose_sneer",       {"noseSneerLeft": 1.0, "noseSneerRight": 1.0}),
]

# rank4 uses each edit's semantically distant "wrong part". Hand-picked so the wrong
# activation is anatomically clearly not the target — a downstream consumer can rely on
# "rank4 activates a different action than rank1" as ground truth, not just "different".
WRONG_PART_MAP = {
    "jaw_open":   "eye_blink",
    "smile":      "brow_up",
    "pucker":     "cheek_puff",
    "funnel":     "eye_wide",
    "brow_up":    "smile",
    "brow_down":  "pucker",
    "eye_blink":  "jaw_open",
    "eye_wide":   "nose_sneer",
    "cheek_puff": "funnel",
    "nose_sneer": "brow_down",
}


def _run(model, facial_actions, phenotype_kwargs) -> np.ndarray:
    with torch.no_grad():
        out = model(facial_actions=facial_actions, phenotype_kwargs=phenotype_kwargs)
    return out["vertices"][0].numpy().astype(np.float64)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE / "build" / "edit")
    ap.add_argument("--phenotype-swap-attribute", default="gender")
    ap.add_argument("--swap-value", type=float, default=1.0)
    a = ap.parse_args(argv[1:])

    a.out_dir.mkdir(parents=True, exist_ok=True)

    from anny import Anny
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref",
                 facial_actions="all")
    model.eval()

    fa_labels = model.facial_action_labels
    phen_labels = model.phenotype_labels
    if a.phenotype_swap_attribute not in phen_labels:
        raise SystemExit(f"phenotype attribute {a.phenotype_swap_attribute!r} not in "
                         f"{phen_labels}")

    # Sanity: every action named in EDITS and WRONG_PART_MAP resolves.
    edit_dict = dict(EDITS)
    for name, actions in EDITS:
        for act in actions:
            if act not in fa_labels:
                raise SystemExit(f"edit {name!r}: action {act!r} not in {len(fa_labels)} labels")
        wrong = WRONG_PART_MAP[name]
        if wrong not in edit_dict:
            raise SystemExit(f"edit {name!r}: wrong-part target {wrong!r} not an edit")

    faces = np.asarray(model.faces, dtype=np.int64)

    def to_action_vector(d):
        v = np.zeros(len(fa_labels), dtype=np.float64)
        for k, val in d.items():
            v[fa_labels.index(k)] = val
        return v

    def to_action_tensor(d):
        return {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in d.items()}

    def phenotype(**overrides):
        return {label: torch.tensor([overrides.get(label, 0.5)], dtype=torch.float64)
                for label in phen_labels}

    def scale_dict(d, s):
        return {k: v * s for k, v in d.items()}

    def phen_to_vector(phen):
        return np.array([phen[l].item() for l in phen_labels], dtype=np.float64)

    def save_mesh(path, verts, fa_dict, phen):
        np.savez_compressed(path, verts=verts, faces=faces,
                            facial_actions=to_action_vector(fa_dict),
                            facial_action_labels=np.array(fa_labels),
                            phenotype=phen_to_vector(phen),
                            phenotype_labels=np.array(phen_labels))

    # Frame A — same rest for every edit family, saved once at out_dir root.
    baseline_phen = phenotype()
    verts_a = _run(model, to_action_tensor({}), baseline_phen)
    save_mesh(a.out_dir / "frame_a.npz", verts_a, {}, baseline_phen)
    print(f"  frame_a: rest, {verts_a.shape}")

    manifest = {"n_edits": len(EDITS), "phenotype_swap_attribute": a.phenotype_swap_attribute,
                "phenotype_swap_value": a.swap_value, "phenotype_baseline": 0.5,
                "edits": {}}

    for edit_name, target_actions in EDITS:
        edir = a.out_dir / edit_name
        edir.mkdir(exist_ok=True)

        # Frame B — the target, all named actions at their strength.
        verts_b = _run(model, to_action_tensor(target_actions), baseline_phen)
        save_mesh(edir / "frame_b.npz", verts_b, target_actions, baseline_phen)

        # rank1 — same as frame_b; save a copy so downstream can address it as a candidate
        # even though it duplicates frame_b's bytes. Tiny cost, keeps the schema uniform.
        save_mesh(edir / "rank1.npz", verts_b, target_actions, baseline_phen)

        # rank2 — 50% interpolation of the target actions.
        r2_actions = scale_dict(target_actions, 0.5)
        verts_r2 = _run(model, to_action_tensor(r2_actions), baseline_phen)
        save_mesh(edir / "rank2.npz", verts_r2, r2_actions, baseline_phen)

        # rank3 — 30% interpolation, under-committed.
        r3_actions = scale_dict(target_actions, 0.3)
        verts_r3 = _run(model, to_action_tensor(r3_actions), baseline_phen)
        save_mesh(edir / "rank3.npz", verts_r3, r3_actions, baseline_phen)

        # rank4 — the WRONG edit's target actions, at strength 1.0.
        wrong_name = WRONG_PART_MAP[edit_name]
        wrong_actions = dict(edit_dict[wrong_name])
        verts_r4 = _run(model, to_action_tensor(wrong_actions), baseline_phen)
        save_mesh(edir / "rank4.npz", verts_r4, wrong_actions, baseline_phen)

        # rank5 — same edit + phenotype swap (wrong subject).
        swap_phen = phenotype(**{a.phenotype_swap_attribute: a.swap_value})
        verts_r5 = _run(model, to_action_tensor(target_actions), swap_phen)
        save_mesh(edir / "rank5.npz", verts_r5, target_actions, swap_phen)

        manifest["edits"][edit_name] = {
            "target_actions": target_actions,
            "wrong_part": wrong_name, "wrong_actions": wrong_actions,
            "rank2_scale": 0.5, "rank3_scale": 0.3,
        }
        print(f"  {edit_name}: 5 candidates written (target = {list(target_actions.keys())})")

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  done. 1 frame_a + {len(EDITS)} edits x 5 candidates = "
          f"{1 + len(EDITS) * 5} meshes in {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
