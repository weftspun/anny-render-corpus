"""Write a posed ANNY mesh and its skeleton as OpenUSD text.

CLAUDE.md's archive rule asks for `.usda` where a thing should stay text editable, and a
corpus subject is exactly that: the mesh a hundred renders were made from should be
readable and diffable rather than only loadable. The npz beside it stays as the renderer's
input; this is the archival form.

WHAT IS WRITTEN. A `UsdGeom.Mesh` with points and topology, and a `UsdSkel.Skeleton`
carrying all 104 joints with their rest and bind transforms, under a `SkelRoot`. The joints
are the same array the labels are projected from, so the USD and the sidecars cannot
disagree about where a joint is.

A NOTE ON NAMES, BECAUSE THE RENAME IS LOSSY AND SILENT OTHERWISE. USD path components are
alphanumeric and underscore, and ANNY's labels carry dots: `pelvis.L`, `toe3-1.R`. The
joint paths therefore hold sanitised names, and `annyJointNames` on the skeleton holds the
originals in the same order. Reading the sanitised name back as an ANNY label is wrong;
read the attribute.

    python mesh_to_usda.py <rest-skel.npz> <out.usda> [--names names.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

SAFE = re.compile(r"[^A-Za-z0-9_]")


def sanitise(name: str) -> str:
    """A USD path component. Digits cannot lead, so a leading digit is prefixed."""
    out = SAFE.sub("_", name)
    return f"j_{out}" if not out or out[0].isdigit() else out


def joint_paths(names, parents) -> list[str]:
    """Full paths in the skeleton's own space, parents before children."""
    paths: list[str] = []
    for i, name in enumerate(names):
        parent = parents[i]
        stem = sanitise(name)
        paths.append(stem if parent < 0 else f"{paths[parent]}/{stem}")
    return paths


def matrix(rows) -> Gf.Matrix4d:
    """A row-major 4x4 into USD's row-vector convention, which is the transpose."""
    return Gf.Matrix4d(*[float(v) for v in np.asarray(rows).T.reshape(-1)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--names", default="")
    ap.add_argument("--subject", default="ANNY rest pose")
    args = ap.parse_args()

    data = np.load(args.npz)
    verts, faces = data["verts"], data["faces"]
    bone_poses, parents = data["bone_poses"], data["parents"]
    names_file = args.names or str(pathlib.Path(args.npz).with_suffix(".names.json"))
    names = json.loads(pathlib.Path(names_file).read_text(encoding="utf-8"))
    if len(names) != len(parents):
        raise SystemExit(f"{len(names)} names against {len(parents)} joints")

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)   # measured: the rig's tall axis is Z
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("comment", f"{args.subject}: {verts.shape[0]} vertices, "
                                 f"{faces.shape[0]} triangles, {len(names)} joints")

    root = UsdSkel.Root.Define(stage, Sdf.Path("/Subject"))
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/Subject/Mesh"))
    # float() per component: Gf.Vec3f takes Python floats, and a numpy scalar is not one.
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z))
                                         for x, y, z in verts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.reshape(-1)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * faces.shape[0]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
        Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2]))]))

    skel = UsdSkel.Skeleton.Define(stage, Sdf.Path("/Subject/Skeleton"))
    paths = joint_paths(names, parents)
    skel.CreateJointsAttr(Vt.TokenArray(paths))
    world = [matrix(m) for m in bone_poses]
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray(world))
    # Rest transforms are joint-local, which is the world transform of a joint expressed in
    # its parent. The root's parent is the skeleton space, so its local is its world.
    rest = []
    for i, parent in enumerate(parents):
        rest.append(world[i] if parent < 0 else world[i] * world[int(parent)].GetInverse())
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray(rest))

    prim = skel.GetPrim()
    attr = prim.CreateAttribute("annyJointNames", Sdf.ValueTypeNames.TokenArray)
    attr.Set(Vt.TokenArray(list(names)))
    prim.CreateAttribute("annyJointNamesNote", Sdf.ValueTypeNames.String).Set(
        "ANNY's own labels, in joint order. The joint paths above are sanitised for USD, "
        "which has no dot in a path component, so pelvis.L becomes pelvis_L. Read labels "
        "from here rather than from the paths.")

    stage.GetRootLayer().Save()
    print(f"wrote {args.out}: {verts.shape[0]} points, {faces.shape[0]} triangles, "
          f"{len(names)} joints, up axis Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
