"""Generate ANNY input + candidate poses for the MaskScore Rung 1 bootstrap.

MaskScore Rung 1 walking-skeleton: no SpeakingFaces fit yet -- rf-detr-keypoint is being
bootstrapped from synthetic renders and needs training data before it can detect anything
back. This script produces that data at the pose level:

  input_mesh    the rest pose (identity), no perturbation
  rank1         same as input -- a candidate that is exactly right
  rank5         random small pose perturbation -- a candidate that is not

Poses are recorded in SOMA format (78 bones x 3 axis-angle rotations + 3 translation,
float64) so the fields the corpus schema names are populated by measurement rather than
promise. The negative control that rule 2 asks for is baked into the pair: rank1 vs
rank5 must score differently under any working metric, and if they do not the metric is
broken rather than the poses.

Outputs to build/bootstrap/:
  rest.npz          verts, faces, pose_soma, translation
  rank1.npz         same as rest
  rank5.npz         rest + random perturbation, same faces
  poses.json        SOMA rotations + translation for both candidates, plus the seed and
                    the perturbation magnitude in radians (~ one credit-card thickness of
                    a rotation, roughly 0.05 rad = 2.9 degrees per bone).

Usage:
    pixi run --environment anny-mac python generate_bootstrap_poses.py [--seed N] [--sigma R]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


PERTURBATION_SIGMA_RAD = 0.05  # 2.9 degrees per bone axis-angle component.
DEFAULT_SEED = 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("build/bootstrap"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sigma-rad", type=float, default=PERTURBATION_SIGMA_RAD,
                    help="std of per-bone axis-angle perturbation")
    a = ap.parse_args(argv[1:])

    a.out_dir.mkdir(parents=True, exist_ok=True)

    from anny import Anny
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref")
    model.eval()

    # BONE COUNT. Persona named "SOMA format, 78 bones, float64". Assert it, so a future
    # rig change that moves the count fails here rather than silently as a shape mismatch
    # downstream in the parquet schema.
    n_bones = len(model.bone_labels)
    if n_bones != 78:
        raise SystemExit(f"expected 78 bones for soma rig, got {n_bones}: {model.bone_labels[:5]}...")

    with torch.no_grad():
        rest = model()
    verts_rest = rest["rest_vertices"][0].numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)

    # REST POSE. Zero rotation on every bone, zero translation. Same identity, same
    # phenotype -- everything downstream can locate the input.
    pose_rest = np.zeros((n_bones, 3), dtype=np.float64)
    trans_rest = np.zeros(3, dtype=np.float64)

    # RANK-5 CANDIDATE. Same identity, a small random pose perturbation. The seed is
    # recorded so the perturbation reproduces from its two numbers rather than a run
    # nobody kept. The magnitude is small enough that the mesh stays anatomically
    # plausible but large enough that a working metric moves on it.
    rng = np.random.default_rng(a.seed)
    pose_rank5 = rng.normal(0.0, a.sigma_rad, size=(n_bones, 3)).astype(np.float64)
    trans_rank5 = np.zeros(3, dtype=np.float64)

    # POSED MESH FOR RANK-5. Run ANNY forward with the perturbation to get vertices at
    # that pose. bone_poses in local-ref parameterization takes axis-angle per bone.
    # pose_parameters shape is [B, J, 4, 4] -- homogeneous transforms per bone. Build from
    # axis-angle via roma to keep the perturbation record (78x3 rotations) as the source of
    # truth in poses.json; the 4x4 form is derived on the way in and not stored twice.
    # Anny runs in float64 -- pass float32 and it hits "expected scalar type Double but
    # found Float" in the bone-orientation solve. Keep the SOMA record float64 too, per
    # persona's spec.
    import roma
    rotvec = torch.from_numpy(pose_rank5[None])                      # [1, 78, 3] float64
    R = roma.rotvec_to_rotmat(rotvec)                                # [1, 78, 3, 3]
    T = torch.zeros(R.shape[0], R.shape[1], 4, 4, dtype=torch.float64)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    with torch.no_grad():
        posed = model(pose_parameters=T)
    verts_rank5 = posed["vertices"][0].numpy().astype(np.float64)

    def write(name, verts, pose, trans):
        p = a.out_dir / f"{name}.npz"
        np.savez_compressed(p, verts=verts, faces=faces, pose_soma=pose, translation=trans)
        return p

    p_rest = write("rest", verts_rest, pose_rest, trans_rest)
    p_rank1 = write("rank1", verts_rest, pose_rest, trans_rest)     # exactly rest
    p_rank5 = write("rank5", verts_rank5, pose_rank5, trans_rank5)

    meta = {
        "rig": "soma",
        "topology": "soma",
        "pose_parameterization": "local-ref",
        "n_bones": n_bones,
        "n_vertices": int(verts_rest.shape[0]),
        "n_faces": int(faces.shape[0]),
        "seed": a.seed,
        "perturbation_sigma_rad": a.sigma_rad,
        "perturbation_sigma_deg": float(np.degrees(a.sigma_rad)),
        "candidates": {
            "input": {"path": p_rest.name, "pose": "rest (identity, all zeros)"},
            "rank1": {"path": p_rank1.name, "pose": "rest (identity, all zeros) -- deliberately equal to input"},
            "rank5": {"path": p_rank5.name, "pose": "rest + N(0, sigma_rad) per bone axis-angle"},
        },
    }
    (a.out_dir / "poses.json").write_text(json.dumps(meta, indent=2))

    print(f"  bones: {n_bones}  verts: {verts_rest.shape[0]}  faces: {faces.shape[0]}")
    print(f"  wrote: {p_rest.name}, {p_rank1.name}, {p_rank5.name}")
    print(f"  perturbation sigma: {a.sigma_rad:.4f} rad ({np.degrees(a.sigma_rad):.2f} deg)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
