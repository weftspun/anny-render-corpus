"""Generate frame A and 15 edit families (frame_b + 5 candidates each) for MaskScore.

MASKSCORE.md `## Edits are frame pairs, not perturbations` defines: frame A (input),
frame B (target), five graded candidates per edit. This script instantiates that on the
ANNY bootstrap path with 15 in-distribution edits: 10 FACS single-action blendshapes
(mouth/brow/eye/cheek/nose) and 5 SOMA bone-rotation poses (head_tilt, head_nod,
head_turn, jaw_open_bone, shoulders_up). No random multi-action combinations; every
frame B is a real, anatomically-plausible expression or pose.

The 15 edits span both modalities MaskScore's Face + Pose stubs care about:
  * face edits (kind='facial_action'): activate 1-2 FACS blendshapes at strength 1.0
  * pose edits (kind='bone_rotation'): rotate 1-2 SOMA bones by a small angle in radians

Rank labels per MASKSCORE.md's 5-rank spec (same for both kinds):
  rank1: correct edit -- exact facial_actions/pose parameters of the target
  rank2: 50% interpolation between A and B (halved action strength or halved bone angle)
  rank3: 30% interpolation (under-committed)
  rank4: a DIFFERENT edit's target (anatomically distant, see WRONG_PART_MAP)
  rank5: same edit + phenotype gender flipped to 1.0 (wrong subject)

Every mesh stores its facial_actions vector, SOMA pose (bone rotations), and phenotype
in the npz so a downstream consumer reproduces any candidate from its record.

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
# where the action naturally moves symmetric parts), OR one SOMA bone rotated by a small
# angle in radians (axis-angle). Coverage: 4 mouth + 2 brow + 2 eye + 1 cheek + 1 nose
# for face; 3 head + 1 jaw + 1 shoulders for pose.
# Structure: (name, kind, params) where params for 'facial_action' is
# {action_label: strength} and for 'bone_rotation' is {bone_label: (rx, ry, rz)}.
POSE_ROT_RAD = 0.25  # about 14 degrees, in-distribution head movement.

EDITS = [
    # face -- 10 FACS single-action edits, unchanged from Rung 1.5 first pass
    ("jaw_open",       "facial_action", {"jawOpen": 1.0}),
    ("smile",          "facial_action", {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0}),
    ("pucker",         "facial_action", {"mouthPucker": 1.0}),
    ("funnel",         "facial_action", {"mouthFunnel": 1.0}),
    ("brow_up",        "facial_action", {"browInnerUp": 1.0}),
    ("brow_down",      "facial_action", {"browDownLeft": 1.0, "browDownRight": 1.0}),
    ("eye_blink",      "facial_action", {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}),
    ("eye_wide",       "facial_action", {"eyeWideLeft": 1.0, "eyeWideRight": 1.0}),
    ("cheek_puff",     "facial_action", {"cheekPuff": 1.0}),
    ("nose_sneer",     "facial_action", {"noseSneerLeft": 1.0, "noseSneerRight": 1.0}),
    # pose -- 5 SOMA bone-rotation edits, new for the "pose changes are also recovered
    # from the anny mesh" scope expansion. head/neck/jaw/shoulders only; camera is
    # near-front so foot/hand bones would render off-screen.
    ("head_tilt",      "bone_rotation", {"Head":  (0.0, 0.0, POSE_ROT_RAD)}),
    ("head_nod",       "bone_rotation", {"Neck2": (POSE_ROT_RAD, 0.0, 0.0)}),
    ("head_turn",      "bone_rotation", {"Neck1": (0.0, POSE_ROT_RAD, 0.0)}),
    ("jaw_open_bone",  "bone_rotation", {"Jaw":   (POSE_ROT_RAD, 0.0, 0.0)}),
    ("shoulders_up",   "bone_rotation", {"LeftShoulder":  (0.0, 0.0,  POSE_ROT_RAD),
                                          "RightShoulder": (0.0, 0.0, -POSE_ROT_RAD)}),
]

# rank4's WRONG_PART_MAP now includes cross-kind pairings so a face edit's "wrong part"
# can be a pose action and vice versa. Anatomically distant either way.
WRONG_PART_MAP = {
    "jaw_open":       "eye_blink",
    "smile":          "brow_up",
    "pucker":         "cheek_puff",
    "funnel":         "eye_wide",
    "brow_up":        "smile",
    "brow_down":      "pucker",
    "eye_blink":      "jaw_open",
    "eye_wide":       "nose_sneer",
    "cheek_puff":     "funnel",
    "nose_sneer":     "brow_down",
    "head_tilt":      "shoulders_up",
    "head_nod":       "jaw_open_bone",
    "head_turn":      "head_tilt",
    "jaw_open_bone":  "head_nod",
    "shoulders_up":   "head_turn",
}


def _run(model, facial_actions, phenotype_kwargs, pose_params=None) -> np.ndarray:
    """Run ANNY forward with optional pose_parameters override (4x4 per-bone transforms)."""
    kw = {"facial_actions": facial_actions, "phenotype_kwargs": phenotype_kwargs}
    if pose_params is not None:
        kw["pose_parameters"] = pose_params
    with torch.no_grad():
        out = model(**kw)
    return out["vertices"][0].numpy().astype(np.float64)


def bone_rotations_to_pose_matrix(bone_rotations: dict, bone_labels: list) -> torch.Tensor:
    """Build a [1, n_bones, 4, 4] tensor where every bone is identity except the named
    ones, which carry the axis-angle rotation from the dict.
    """
    import roma
    n = len(bone_labels)
    rotvecs = torch.zeros((1, n, 3), dtype=torch.float64)
    for label, (rx, ry, rz) in bone_rotations.items():
        i = bone_labels.index(label)
        rotvecs[0, i, 0] = rx
        rotvecs[0, i, 1] = ry
        rotvecs[0, i, 2] = rz
    R = roma.rotvec_to_rotmat(rotvecs)                                 # [1, n, 3, 3]
    T = torch.zeros(1, n, 4, 4, dtype=torch.float64)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    return T


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
    bone_labels = model.bone_labels
    phen_labels = model.phenotype_labels
    if a.phenotype_swap_attribute not in phen_labels:
        raise SystemExit(f"phenotype attribute {a.phenotype_swap_attribute!r} not in "
                         f"{phen_labels}")

    # Sanity every edit resolves against the rig.
    edit_lookup = {name: (kind, params) for name, kind, params in EDITS}
    for name, kind, params in EDITS:
        if kind == "facial_action":
            for act in params:
                if act not in fa_labels:
                    raise SystemExit(f"edit {name!r}: facial action {act!r} not in "
                                     f"{len(fa_labels)} labels")
        elif kind == "bone_rotation":
            for bone in params:
                if bone not in bone_labels:
                    raise SystemExit(f"edit {name!r}: bone {bone!r} not in {len(bone_labels)} bones")
        else:
            raise SystemExit(f"edit {name!r}: unknown kind {kind!r}")
        wrong = WRONG_PART_MAP[name]
        if wrong not in edit_lookup:
            raise SystemExit(f"edit {name!r}: wrong-part target {wrong!r} not an edit")

    faces = np.asarray(model.faces, dtype=np.int64)

    def action_vector(d):
        v = np.zeros(len(fa_labels), dtype=np.float64)
        for k, val in d.items():
            v[fa_labels.index(k)] = val
        return v

    def action_tensor(d):
        return {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in d.items()}

    def pose_vector(bone_rotations):
        v = np.zeros((len(bone_labels), 3), dtype=np.float64)
        for label, (rx, ry, rz) in bone_rotations.items():
            i = bone_labels.index(label)
            v[i] = (rx, ry, rz)
        return v

    def phenotype(**overrides):
        return {label: torch.tensor([overrides.get(label, 0.5)], dtype=torch.float64)
                for label in phen_labels}

    def phen_vector(phen):
        return np.array([phen[l].item() for l in phen_labels], dtype=np.float64)

    def build_state(edit_kind, params, scale=1.0):
        """Return (facial_actions_dict, bone_rotations_dict) for a candidate at the given scale."""
        if edit_kind == "facial_action":
            fa_scaled = {k: v * scale for k, v in params.items()}
            return fa_scaled, {}
        else:  # bone_rotation
            br_scaled = {k: (rx * scale, ry * scale, rz * scale)
                         for k, (rx, ry, rz) in params.items()}
            return {}, br_scaled

    def run(fa_dict, br_dict, phen):
        pose_params = (bone_rotations_to_pose_matrix(br_dict, bone_labels)
                       if br_dict else None)
        return _run(model, action_tensor(fa_dict), phen, pose_params=pose_params)

    def save_mesh(path, verts, fa_dict, br_dict, phen, edit_kind):
        np.savez_compressed(path, verts=verts, faces=faces,
                            facial_actions=action_vector(fa_dict),
                            facial_action_labels=np.array(fa_labels),
                            pose_soma=pose_vector(br_dict),   # (78, 3) axis-angle radians
                            bone_labels=np.array(bone_labels),
                            phenotype=phen_vector(phen),
                            phenotype_labels=np.array(phen_labels),
                            edit_kind=np.array(edit_kind))

    # Frame A -- rest for every edit family.
    baseline_phen = phenotype()
    verts_a = run({}, {}, baseline_phen)
    save_mesh(a.out_dir / "frame_a.npz", verts_a, {}, {}, baseline_phen, "rest")
    print(f"  frame_a: rest, {verts_a.shape}")

    manifest = {"n_edits": len(EDITS),
                "phenotype_swap_attribute": a.phenotype_swap_attribute,
                "phenotype_swap_value": a.swap_value, "phenotype_baseline": 0.5,
                "pose_rot_rad_default": POSE_ROT_RAD,
                "edits": {}}

    for edit_name, edit_kind, params in EDITS:
        edir = a.out_dir / edit_name
        edir.mkdir(exist_ok=True)

        fa1, br1 = build_state(edit_kind, params, scale=1.0)
        verts_b = run(fa1, br1, baseline_phen)
        save_mesh(edir / "frame_b.npz", verts_b, fa1, br1, baseline_phen, edit_kind)
        save_mesh(edir / "rank1.npz",   verts_b, fa1, br1, baseline_phen, edit_kind)

        fa2, br2 = build_state(edit_kind, params, scale=0.5)
        verts_r2 = run(fa2, br2, baseline_phen)
        save_mesh(edir / "rank2.npz", verts_r2, fa2, br2, baseline_phen, edit_kind)

        fa3, br3 = build_state(edit_kind, params, scale=0.3)
        verts_r3 = run(fa3, br3, baseline_phen)
        save_mesh(edir / "rank3.npz", verts_r3, fa3, br3, baseline_phen, edit_kind)

        # rank4 uses the WRONG edit's action/rotation set, applied at strength 1.0. Kind
        # may differ from this edit's kind -- a facial edit's wrong part may be a pose
        # action or vice versa.
        wrong_name = WRONG_PART_MAP[edit_name]
        wrong_kind, wrong_params = edit_lookup[wrong_name]
        fa4, br4 = build_state(wrong_kind, wrong_params, scale=1.0)
        verts_r4 = run(fa4, br4, baseline_phen)
        save_mesh(edir / "rank4.npz", verts_r4, fa4, br4, baseline_phen, wrong_kind)

        # rank5: same edit, phenotype swap -> "wrong subject".
        swap_phen = phenotype(**{a.phenotype_swap_attribute: a.swap_value})
        verts_r5 = run(fa1, br1, swap_phen)
        save_mesh(edir / "rank5.npz", verts_r5, fa1, br1, swap_phen, edit_kind)

        manifest["edits"][edit_name] = {
            "kind": edit_kind,
            "target_params": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in params.items()},
            "wrong_part": wrong_name, "wrong_kind": wrong_kind,
            "wrong_params": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in wrong_params.items()},
            "rank2_scale": 0.5, "rank3_scale": 0.3,
            # Kept for backward compatibility with the older schema readers.
            "target_actions": params if edit_kind == "facial_action" else {},
            "wrong_actions": wrong_params if wrong_kind == "facial_action" else {},
        }
        print(f"  {edit_name} ({edit_kind}): 5 candidates written "
              f"(target = {list(params.keys())})")

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  done. 1 frame_a + {len(EDITS)} edits x 5 candidates = "
          f"{1 + len(EDITS) * 5} meshes in {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
