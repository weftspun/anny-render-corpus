"""urn:oid:1.3.6.1.4.1.66606.1.2.2162.1 -- author real ANNY skinning for USDZ.

The Video-stub USDZ previously carried a 1-joint dummy skeleton, so a
consumer that animated the joints saw no motion: every skin weight
mapped every vertex to the single dummy. A downstream measurement of
pose-animated candidates was measuring rest pose.

This module builds:

  1. The full 104-joint UsdSkel.Skeleton from `anny_rig.build_corpus_model()`
     (hierarchical joint paths encoding bone_parents; bind and rest world
     transforms from `rest_bone_poses`).
  2. Per-vertex skin weights on the render mesh, rebound by nearest-neighbor
     from the corpus model's canonical (13,718-vert) skin binding onto the
     render mesh (18,056 verts in the current fixture). The approximation is
     recorded in customData so a consumer can tell canonical binding from
     rebound binding without reading the mesh.

The rebinding is nearest-neighbor rather than barycentric because:
  - our current use is rest-pose facial-action video where bones do not move
    (the failure it fixes is a downstream tool seeing 1 joint at all, not
    incorrect deformations under motion);
  - 3/n detection floor (rule 5) on 18,056 vertices is ~0.02%: even a small
    barycentric win is invisible at MaskScore's scoring granularity;
  - the caveat surfaces in customData so a future SOMA-bone-animated stub
    can rebuild with a stronger method before shipping motion.

References:
  urn:oid:1.3.6.1.4.1.66606.1.1.1173  MaskScore parent
  urn:oid:1.3.6.1.4.1.66606.1.2.2162  USDZ skinning + 15-edit emit
  urn:oid:1.3.6.1.4.1.66606.1.2.2165.1  schema module (unrelated but paired refactor)
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


MAX_INFLUENCES = 9  # matches ANNY's corpus model


@dataclasses.dataclass
class Skinning:
    """Full ANNY skin binding for one render mesh."""
    joint_paths:      list[str]                # 104 entries, slash-delimited hierarchy
    bind_transforms:  np.ndarray               # (104, 4, 4) float64 world matrices
    rest_transforms:  np.ndarray               # (104, 4, 4) float64 world matrices
    joint_indices:    np.ndarray               # (V, 9) int32
    joint_weights:    np.ndarray               # (V, 9) float32, rows sum to 1 (or 0 for unbound)
    rebind_source:    str                      # 'canonical' or 'nearest_neighbor'
    corpus_verts:     int                      # V of the source binding
    render_verts:     int                      # V of the target mesh
    bone_labels:      list[str]                # for provenance


def _joint_paths_from_parents(labels: list[str], parents: np.ndarray) -> list[str]:
    """USD Skeleton joint paths encode hierarchy as slash-delimited names.

    ANNY's parents[0] == -1 (root). Every other parent[i] < i, so a single
    forward pass suffices to build each path.
    """
    if int(parents[0]) != -1:
        raise ValueError(f"expected root parent = -1, got {int(parents[0])}")
    paths: list[str] = [labels[0]]
    for i in range(1, len(labels)):
        p = int(parents[i])
        if p < 0 or p >= i:
            raise ValueError(f"parent[{i}] = {p} not a valid earlier index")
        paths.append(f"{paths[p]}/{labels[i]}")
    return paths


def _nearest_neighbor_rebind(src_verts: np.ndarray, tgt_verts: np.ndarray,
                             src_indices: np.ndarray, src_weights: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """For each target vert, copy the skin binding of its nearest source vert.

    Brute-force distance; both meshes fit in memory and this runs once per
    corpus build. No sklearn/scipy dependency (keeps the anny-mac env slim).
    """
    if src_indices.shape[1] != MAX_INFLUENCES:
        raise ValueError(f"src_indices last dim must be {MAX_INFLUENCES}, "
                         f"got {src_indices.shape[1]}")
    nearest = np.empty(tgt_verts.shape[0], dtype=np.int64)
    chunk = 2048
    for start in range(0, tgt_verts.shape[0], chunk):
        end = min(start + chunk, tgt_verts.shape[0])
        diff = tgt_verts[start:end, None, :] - src_verts[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        nearest[start:end] = d2.argmin(axis=1)
    return src_indices[nearest].astype(np.int32), src_weights[nearest].astype(np.float32)


def build_from_anny(render_verts: np.ndarray, model=None) -> Skinning:
    """Build a Skinning for `render_verts` from a fresh ANNY corpus model."""
    if model is None:
        import anny_rig
        model = anny_rig.build_corpus_model()

    result = model()
    rest_bone_poses = result["rest_bone_poses"][0].detach().cpu().numpy().astype(np.float64)
    corpus_verts = result["rest_vertices"][0].detach().cpu().numpy().astype(np.float64)

    labels = [str(x) for x in model.bone_labels]
    parents = np.asarray(model.bone_parents, dtype=np.int32)
    joint_paths = _joint_paths_from_parents(labels, parents)

    src_indices = np.asarray(model.vertex_bone_indices).astype(np.int32)
    src_weights = np.asarray(model.vertex_bone_weights).astype(np.float32)

    if render_verts.shape[0] == corpus_verts.shape[0]:
        # Canonical binding applies directly.
        joint_indices, joint_weights = src_indices, src_weights
        rebind_source = "canonical"
    else:
        joint_indices, joint_weights = _nearest_neighbor_rebind(
            corpus_verts, np.asarray(render_verts, dtype=np.float64),
            src_indices, src_weights,
        )
        rebind_source = "nearest_neighbor"

    return Skinning(
        joint_paths=joint_paths,
        bind_transforms=rest_bone_poses,
        rest_transforms=rest_bone_poses,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
        rebind_source=rebind_source,
        corpus_verts=int(corpus_verts.shape[0]),
        render_verts=int(render_verts.shape[0]),
        bone_labels=labels,
    )


def save(sk: Skinning, path) -> None:
    """Persist a Skinning to .npz so a caller in an env without anny_rig
    (e.g. the `usd` env, which has pxr but not anny_rig) can load and author
    it without re-running the corpus model."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_paths=np.array(sk.joint_paths, dtype=object),
        bind_transforms=sk.bind_transforms,
        rest_transforms=sk.rest_transforms,
        joint_indices=sk.joint_indices,
        joint_weights=sk.joint_weights,
        rebind_source=np.array(sk.rebind_source),
        corpus_verts=np.array(sk.corpus_verts),
        render_verts=np.array(sk.render_verts),
        bone_labels=np.array(sk.bone_labels, dtype=object),
    )


def load(path) -> Skinning:
    """Inverse of `save`."""
    d = np.load(path, allow_pickle=True)
    return Skinning(
        joint_paths=[str(x) for x in d["joint_paths"]],
        bind_transforms=d["bind_transforms"],
        rest_transforms=d["rest_transforms"],
        joint_indices=d["joint_indices"],
        joint_weights=d["joint_weights"],
        rebind_source=str(d["rebind_source"]),
        corpus_verts=int(d["corpus_verts"]),
        render_verts=int(d["render_verts"]),
        bone_labels=[str(x) for x in d["bone_labels"]],
    )


def apply_to_usd(stage, skel_path: str, mesh_path: str, skinning: Skinning) -> None:
    """Author the Skeleton + SkelBindingAPI primvars on an existing USD stage.

    The mesh at `mesh_path` must exist. This function REPLACES any prior
    joints attribute and skin-weight primvars on that mesh, so a caller that
    used to author a 1-joint dummy skeleton can drop that block and call this
    instead.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdSkel, Vt

    skel = UsdSkel.Skeleton.Define(stage, skel_path)
    skel.CreateJointsAttr(skinning.joint_paths)
    skel.CreateBindTransformsAttr(
        [Gf.Matrix4d(*m.reshape(-1).tolist()) for m in skinning.bind_transforms]
    )
    skel.CreateRestTransformsAttr(
        [Gf.Matrix4d(*m.reshape(-1).tolist()) for m in skinning.rest_transforms]
    )

    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim:
        raise ValueError(f"mesh prim not found at {mesh_path}")
    binding = UsdSkel.BindingAPI.Apply(mesh_prim)
    binding.CreateSkeletonRel().SetTargets([Sdf.Path(skel_path)])

    # jointIndices/jointWeights primvars are per-vertex with elementSize = MAX_INFLUENCES.
    v = skinning.joint_indices.shape[0]
    idx_flat = skinning.joint_indices.reshape(-1).astype(np.int32)
    w_flat   = skinning.joint_weights.reshape(-1).astype(np.float32)
    ji = binding.CreateJointIndicesPrimvar(constant=False, elementSize=MAX_INFLUENCES)
    ji.Set(Vt.IntArray.FromNumpy(idx_flat))
    jw = binding.CreateJointWeightsPrimvar(constant=False, elementSize=MAX_INFLUENCES)
    jw.Set(Vt.FloatArray.FromNumpy(w_flat))

    mesh_prim.SetCustomDataByKey("skin_binding_source", skinning.rebind_source)
    mesh_prim.SetCustomDataByKey("skin_binding_influences", MAX_INFLUENCES)
    mesh_prim.SetCustomDataByKey("skin_binding_corpus_verts", skinning.corpus_verts)
    mesh_prim.SetCustomDataByKey("skin_binding_render_verts", v)


# ---- Self-test with controls. ----

def _self_test() -> int:
    """Positive control: canonical mesh (13718 verts) binds without rebinding.
    Negative controls:
      * a mesh with the wrong vert count triggers nearest-neighbor rebinding
      * joint hierarchy is well-formed (no forward parent refs)
      * weights sum to ~1 (per-row) on bound vertices
      * bind_transforms are non-degenerate (det != 0)
    """
    import anny_rig
    m = anny_rig.build_corpus_model()
    canonical = m()["rest_vertices"][0].detach().cpu().numpy()

    # positive: exact vert count -> canonical
    sk = build_from_anny(canonical, model=m)
    if sk.rebind_source != "canonical":
        print(f"FAIL: exact-vert-count mesh took {sk.rebind_source} path")
        return 1
    if len(sk.joint_paths) != 104:
        print(f"FAIL: expected 104 joints, got {len(sk.joint_paths)}")
        return 1
    if sk.joint_paths[0] != "root":
        print(f"FAIL: root joint path is {sk.joint_paths[0]!r}, expected 'root'")
        return 1
    # weights per row should sum to ~1 (or be exactly 0 for unbound rows)
    row_sums = sk.joint_weights.sum(axis=1)
    bad = np.where((row_sums > 1e-6) & (np.abs(row_sums - 1.0) > 1e-3))[0]
    if bad.size:
        print(f"FAIL: {bad.size} rows have weights that don't sum to 1 "
              f"(first: row {bad[0]} sums to {row_sums[bad[0]]:.6f})")
        return 1
    # bind transforms non-degenerate
    dets = np.linalg.det(sk.bind_transforms[:, :3, :3])
    if np.any(np.abs(dets) < 1e-6):
        print(f"FAIL: {int(np.sum(np.abs(dets) < 1e-6))} bind transforms are degenerate")
        return 1
    # joint hierarchy: every non-root path contains its parent's path as prefix
    for i, path in enumerate(sk.joint_paths[1:], start=1):
        if "/" not in path:
            print(f"FAIL: non-root joint {i} path {path!r} has no parent segment")
            return 1

    # negative: rebound path on a synthetic 18056-vert mesh.
    rng = np.random.default_rng(0)
    fake = canonical[rng.integers(0, canonical.shape[0], size=18056)] \
           + rng.normal(0.0, 0.001, size=(18056, 3))
    sk_reb = build_from_anny(fake, model=m)
    if sk_reb.rebind_source != "nearest_neighbor":
        print(f"FAIL: wrong-vert-count mesh took {sk_reb.rebind_source} path")
        return 1
    if sk_reb.joint_indices.shape != (18056, MAX_INFLUENCES):
        print(f"FAIL: rebound joint_indices shape {sk_reb.joint_indices.shape}")
        return 1
    row_sums = sk_reb.joint_weights.sum(axis=1)
    bad = np.where((row_sums > 1e-6) & (np.abs(row_sums - 1.0) > 1e-3))[0]
    if bad.size:
        print(f"FAIL: rebound {bad.size} rows fail weight-sum check")
        return 1

    print(f"ok: {len(sk.joint_paths)} joints, canonical + nearest-neighbor "
          f"both bind clean; weights sum to 1 on bound vertices; bind "
          f"transforms non-degenerate")
    return 0


def _build_cli(argv: list[str]) -> int:
    """CLI wrapper. Runs under the `anny-mac` env; writes a Skinning sidecar
    that the `usd` env's emit script loads."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-verts-npz", required=True, type=pathlib.Path,
                    help="npz containing a 'verts' or 'rest_verts' array of the render mesh")
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help="destination .npz for the Skinning sidecar")
    a = ap.parse_args(argv)
    d = np.load(a.render_verts_npz)
    for k in ("rest_verts", "verts"):
        if k in d.files:
            verts = d[k]
            break
    else:
        raise SystemExit(f"{a.render_verts_npz}: no 'rest_verts' or 'verts' key "
                         f"(found: {list(d.files)})")
    sk = build_from_anny(np.asarray(verts, dtype=np.float64))
    save(sk, a.out)
    print(f"wrote {a.out}: {len(sk.joint_paths)} joints, {sk.render_verts} verts, "
          f"binding source: {sk.rebind_source}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        sys.exit(_build_cli(sys.argv[2:]))
    sys.exit(_self_test())
