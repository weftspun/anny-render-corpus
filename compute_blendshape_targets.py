"""Compute the 52 FACS blendshape vertex deltas ANNY exposes via facial_actions.

Each of the 52 facial-action labels drives a per-vertex offset from the rest mesh when
activated at strength 1.0. Those offsets are the UsdSkelBlendShape target deltas -- one
Vec3 array per action, referenced by SkelAnimation's per-frame `blendShapeWeights`.

The output is a single npz file the USD-emit step reads:
  rest_verts:       (V, 3) float64
  faces:            (F, 3) int64
  labels:           (52,)  str
  deltas:           (52, V, 3) float64   -- per-action (posed - rest)

USD stores blendshape offsets ONLY on the vertices that moved (sparse indices). We
compute both dense and sparse forms here; the USD emitter chooses the sparse form for
each target so the .usdz file stays small.

Usage:
    pixi run --environment anny-mac python compute_blendshape_targets.py \
        [--out build/blendshapes.npz]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "build" / "blendshapes.npz")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="minimum per-vertex movement to keep in the sparse target")
    a = ap.parse_args(argv[1:])

    a.out.parent.mkdir(parents=True, exist_ok=True)

    from anny import Anny
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref",
                 facial_actions="all")
    model.eval()
    labels = list(model.facial_action_labels)
    phen_labels = model.phenotype_labels

    def phen_default():
        return {l: torch.tensor([0.5], dtype=torch.float64) for l in phen_labels}

    def actions(**kw):
        return {l: torch.tensor([float(kw.get(l, 0.0))], dtype=torch.float64) for l in labels}

    with torch.no_grad():
        rest = model(facial_actions=actions(), phenotype_kwargs=phen_default())
    rest_verts = rest["vertices"][0].numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)

    deltas = np.zeros((len(labels), rest_verts.shape[0], 3), dtype=np.float64)
    for i, label in enumerate(labels):
        with torch.no_grad():
            posed = model(facial_actions=actions(**{label: 1.0}), phenotype_kwargs=phen_default())
        posed_verts = posed["vertices"][0].numpy().astype(np.float64)
        deltas[i] = posed_verts - rest_verts

    per_shape_norm = np.linalg.norm(deltas, axis=2).max(axis=1)  # per-shape max vertex move
    per_shape_nnz  = (np.linalg.norm(deltas, axis=2) > a.eps).sum(axis=1)  # sparse index count

    np.savez_compressed(a.out,
                        rest_verts=rest_verts, faces=faces,
                        labels=np.array(labels),
                        deltas=deltas.astype(np.float32),
                        per_shape_max_move=per_shape_norm,
                        per_shape_nnz=per_shape_nnz)

    print(f"  wrote {len(labels)} blendshape targets, "
          f"rest verts {rest_verts.shape}, faces {faces.shape}")
    print(f"  max per-shape movement (top 5): "
          f"{sorted(zip(labels, per_shape_norm.tolist()), key=lambda x: -x[1])[:5]}")
    print(f"  mean sparse-index count per shape: {per_shape_nnz.mean():.0f} / {rest_verts.shape[0]}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
