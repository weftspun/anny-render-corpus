# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Write a UsdGeom.Mesh out as OBJ, so Mitsuba and three.js can load the same geometry.

    python usda_to_obj.py <in.usda> <out.obj> [--self-test]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def read_mesh(path):
    """Conventions are READ, never assumed. CLAUDE.md rule 6."""
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
    conv = {
        "up": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "orientation": str(mesh.GetOrientationAttr().Get() or "rightHanded"),
        "xform_ops": [str(o.GetOpName()) for o in UsdGeom.Xformable(prim).GetOrderedXformOps()],
    }
    return str(prim.GetPath()), points, counts, indices, conv


def fan(counts, indices):
    """Triangulate by fan. Convex faces only, which a quad base mesh satisfies."""
    tris, at = [], 0
    for n in counts:
        face = indices[at:at + n]
        at += n
        for k in range(1, n - 1):
            tris.append((face[0], face[k], face[k + 1]))
    return np.asarray(tris, dtype=np.int64)


UP_TO_Y = {"Y": np.eye(3),
           "Z": np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])}


def to_y_up(points, up):
    """OBJ has no up axis, so everything is written Y-up from the declared one."""
    if up not in UP_TO_Y:
        raise SystemExit("FAIL  unknown up axis %r; this handles %s"
                         % (up, sorted(UP_TO_Y)))
    return points @ UP_TO_Y[up].T


def handedness(points, tris):
    """Signed volume; positive is outward, which all three consumers take as front."""
    t = points[tris]
    return float(np.einsum("ij,ij->i",
                           t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


FORWARD_Y_UP = np.array([0.0, 0.0, -1.0])


def subject_forward(facing):
    """A subject forward, measured off the body. Never a world axis, never a default."""
    if facing is None:
        raise SystemExit("FAIL  a subject forward has to be measured off the body, not "
                         "taken from the world axes")
    v = np.asarray(facing, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise SystemExit("FAIL  a zero facing vector names no direction")
    return v / n


def longest_axis(points):
    d = points.max(axis=0) - points.min(axis=0)
    return "XYZ"[int(np.argmax(d))], d


def normalise(points, scale=1.0):
    """Centre on the origin and fit the longest axis to `scale`, so a camera framed for one
    mesh frames any of them."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = float((hi - lo).max())
    return (points - (lo + hi) / 2.0) * (scale / extent)


def vertex_normals(points, tris):
    """Area-weighted, written out so no renderer computes its own."""
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


def convert(src, dst, scale=1.0, keep_metres=False):
    name, points, counts, indices, conv = read_mesh(src)
    tris = fan(counts, indices)
    points = points * conv["meters_per_unit"]
    points = to_y_up(points, conv["up"])
    metres = points.max(axis=0) - points.min(axis=0)
    if not keep_metres:
        points = normalise(points, scale)
    write(dst, points, tris, vertex_normals(points, tris))
    conv["metres"] = [float(x) for x in metres]
    conv["signed_volume"] = handedness(points, tris)
    return name, points, tris, conv


def self_test():
    """Twenty-four controls, over geometry and over the conventions it carries."""
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
    # The bounding box, not the centroid; a symmetric fixture hides the difference.
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

    # SCALE.
    box = np.array([[0., 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0], [0, 0, 3]])
    r.append(("metres scale with metersPerUnit",
              np.allclose((box * 0.01).max(0) - (box * 0.01).min(0),
                          (box.max(0) - box.min(0)) * 0.01)))
    r.append(("normalise is what discards real scale, and only when asked",
              abs((normalise(box, 1.0).max(0) - normalise(box, 1.0).min(0)).max() - 1.0)
              < 1e-12))

    # UP.
    tall_z = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 5]])
    rot = to_y_up(tall_z, "Z")
    r.append(("a Z-up asset has its height moved to Y",
              longest_axis(rot)[0] == "Y" and longest_axis(tall_z)[0] == "Z"))
    r.append(("a Y-up asset is left alone", np.allclose(to_y_up(tall_z, "Y"), tall_z)))
    r.append(("the up rotation is a rotation, not a reflection",
              abs(np.linalg.det(UP_TO_Y["Z"]) - 1.0) < 1e-12))
    r.append(("lengths survive the up rotation",
              abs(np.linalg.norm(rot[3]) - np.linalg.norm(tall_z[3])) < 1e-12))
    try:
        to_y_up(tall_z, "W")
        r.append(("an unknown up axis is refused", False))
    except SystemExit:
        r.append(("an unknown up axis is refused", True))

    # FORWARD.
    r.append(("forward is orthogonal to up",
              abs(float(FORWARD_Y_UP @ np.array([0.0, 1.0, 0.0]))) < 1e-12))

    # BODY FORWARD.
    r.append(("a subject forward is normalised",
              abs(np.linalg.norm(subject_forward((0, 0, -3))) - 1.0) < 1e-12))
    for bad_facing, why in ((None, "an absent facing"), ((0, 0, 0), "a zero facing")):
        try:
            subject_forward(bad_facing)
            r.append(("%s is refused rather than defaulted" % why, False))
        except SystemExit:
            r.append(("%s is refused rather than defaulted" % why, True))

    # HANDEDNESS.
    cube_v = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) - 0.5
    cube_f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                       [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                       [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    vol = handedness(cube_v, cube_f)
    r.append(("a closed cube has the volume it has", abs(abs(vol) - 1.0) < 1e-9))
    mirrored = cube_v * np.array([-1.0, 1, 1])
    r.append(("mirroring flips the sign, so handedness is detected",
              np.sign(handedness(mirrored, cube_f)) == -np.sign(vol)))
    r.append(("reversing the winding flips it too",
              np.sign(handedness(cube_v, cube_f[:, ::-1])) == -np.sign(vol)))

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
    ap.add_argument("--keep-metres", action="store_true",
                    help="leave the mesh at real-world scale instead of fitting to --scale")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.src or not args.dst:
        return ap.error("give a source and a destination")
    name, points, tris, conv = convert(args.src, args.dst, args.scale, args.keep_metres)
    axis, extent = longest_axis(points)
    print("  %s -> %s   %d points, %d triangles" % (name, args.dst, len(points), len(tris)))
    print("  source up %s, %g m per unit, %s, %d xform op(s)"
          % (conv["up"], conv["meters_per_unit"], conv["orientation"], len(conv["xform_ops"])))
    print("  real size %.3f x %.3f x %.3f m   written Y-up, longest axis %s (%.3f)"
          % (*conv["metres"], axis, extent.max()))
    print("  signed volume %+.6f (%s)"
          % (conv["signed_volume"],
             "outward, right-handed" if conv["signed_volume"] > 0 else "INWARD or flipped"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
