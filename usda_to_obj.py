# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Write a UsdGeom.Mesh out as OBJ, so Mitsuba and three.js can load the same geometry.

    python usda_to_obj.py <in.usda> <out.obj> [--self-test]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def read_mesh(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise SystemExit("FAIL  %s does not open" % path)
    prim = next((p for p in stage.TraverseAll() if p.IsA(UsdGeom.Mesh)), None)
    if prim is None:
        raise SystemExit("FAIL  %s carries no UsdGeom.Mesh" % path)
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if counts.sum() != len(indices):
        raise SystemExit("FAIL  %d face-vertex counts against %d indices"
                         % (counts.sum(), len(indices)))
    return str(prim.GetPath()), points, counts, indices


def fan(counts, indices):
    """Triangulate by fan. Convex faces only, which a quad base mesh satisfies."""
    tris, at = [], 0
    for n in counts:
        face = indices[at:at + n]
        at += n
        for k in range(1, n - 1):
            tris.append((face[0], face[k], face[k + 1]))
    return np.asarray(tris, dtype=np.int64)


def normalise(points, scale=1.0):
    """Centre on the origin and fit the longest axis to `scale`, so a camera framed for one
    mesh frames any of them."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = float((hi - lo).max())
    return (points - (lo + hi) / 2.0) * (scale / extent)


def vertex_normals(points, tris):
    """Area-weighted, written into the OBJ so every renderer shades the same normal: three.js
    would smooth its own and Mitsuba would take the face normal."""
    n = np.zeros_like(points)
    t = points[tris]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def write(path, points, tris, normals=None):
    with open(path, "w", encoding="utf-8") as fh:
        for p in points:
            fh.write("v %.8f %.8f %.8f\n" % tuple(p))
        if normals is None:
            for t in tris + 1:
                fh.write("f %d %d %d\n" % tuple(t))
            return
        for q in normals:
            fh.write("vn %.8f %.8f %.8f\n" % tuple(q))
        for t in tris + 1:
            fh.write("f %d//%d %d//%d %d//%d\n" % (t[0], t[0], t[1], t[1], t[2], t[2]))


def convert(src, dst, scale=1.0):
    name, points, counts, indices = read_mesh(src)
    tris = fan(counts, indices)
    points = normalise(points, scale)
    write(dst, points, tris, vertex_normals(points, tris))
    return name, points, tris


def self_test():
    """Ten controls. Five must reject a conversion that changed the geometry."""
    import pathlib
    import tempfile
    r = []
    counts = np.array([4, 3])
    indices = np.array([0, 1, 2, 3, 0, 2, 4])
    tris = fan(counts, indices)
    r.append(("a quad becomes two triangles and a triangle stays one", len(tris) == 3))
    r.append(("every triangulated corner came from its own face",
              set(map(tuple, tris)) == {(0, 1, 2), (0, 2, 3), (0, 2, 4)}))

    pts = np.array([[0., 0, 0], [2, 0, 0], [2, 4, 0], [0, 4, 0], [1, 6, 0]])
    n = normalise(pts, 1.0)
    # The BOUNDING BOX, not the centroid. A symmetric fixture makes the two coincide, which
    # is how the first version of this control passed while asserting the wrong property.
    r.append(("normalise centres the bounding box",
              np.allclose((n.max(0) + n.min(0)) / 2, 0, atol=1e-12)))
    r.append(("normalise fits the longest axis", abs((n.max(0) - n.min(0)).max() - 1.0) < 1e-12))
    want = (pts.max(0) - pts.min(0))[0] / (pts.max(0) - pts.min(0))[1]
    r.append(("normalise keeps the aspect ratio",
              abs((n.max(0) - n.min(0))[0] / (n.max(0) - n.min(0))[1] - want) < 1e-12))
    r.append(("the centroid is NOT what moved, so the two are told apart",
              not np.allclose(n.mean(axis=0), 0, atol=1e-9)))

    nrm = vertex_normals(n, tris)
    r.append(("normals are unit length", np.allclose(np.linalg.norm(nrm, axis=1), 1.0)))
    r.append(("a planar fixture has one normal direction",
              np.allclose(np.abs(nrm @ nrm[0]), 1.0)))
    d = pathlib.Path(tempfile.mkdtemp()) / "t.obj"
    write(d, n, tris, nrm)
    body = d.read_text(encoding="utf-8")
    r.append(("obj indices are one-based", "f 1//1 2//2 3//3" in body))
    r.append(("the obj carries normals, so no renderer invents its own",
              body.count("vn ") == len(n)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.src or not args.dst:
        return ap.error("give a source and a destination")
    name, points, tris = convert(args.src, args.dst, args.scale)
    print("  %s -> %s   %d points, %d triangles, longest axis %.3f"
          % (name, args.dst, len(points), len(tris), (points.max(0) - points.min(0)).max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
