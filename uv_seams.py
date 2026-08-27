# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Split a mesh at its UV seams so texcoords can be carried per vertex.

    python uv_seams.py --self-test
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def to_mitsuba_uv(uv):
    """ANNY's v origin is at the bottom, Mitsuba's obj loader puts it at the top."""
    uv = np.asarray(uv, dtype=np.float64).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def split(verts, faces, uv, face_uv):
    """Returns (verts2, faces2, uv2) with one texcoord per vertex."""
    verts = np.asarray(verts)
    faces = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    face_uv = np.asarray(face_uv, dtype=np.int64)
    if faces.shape != face_uv.shape:
        raise ValueError("faces %s and face_uv %s must have the same shape"
                         % (faces.shape, face_uv.shape))
    if face_uv.size and (face_uv.max() >= len(uv) or face_uv.min() < 0):
        raise ValueError("face_uv indexes outside the %d texture coordinates" % len(uv))
    if faces.size and (faces.max() >= len(verts) or faces.min() < 0):
        raise ValueError("faces indexes outside the %d vertices" % len(verts))

    corner = np.stack([faces.reshape(-1), face_uv.reshape(-1)], axis=1)
    pairs, inverse = np.unique(corner, axis=0, return_inverse=True)
    verts2 = verts[pairs[:, 0]]
    uv2 = uv[pairs[:, 1]]
    faces2 = inverse.reshape(faces.shape).astype(np.int64)
    return verts2, faces2, uv2


def naive(verts, faces, uv, face_uv):
    """Last writer wins, as an unsplit bind does."""
    out = np.zeros((len(verts), 2), dtype=np.float64)
    out[np.asarray(faces).reshape(-1)] = np.asarray(uv)[np.asarray(face_uv).reshape(-1)]
    return out


def seam_vertices(faces, face_uv):
    """Vertices owning more than one texcoord."""
    corner = np.stack([np.asarray(faces).reshape(-1),
                       np.asarray(face_uv).reshape(-1)], axis=1)
    pairs = np.unique(corner, axis=0)
    vert, count = np.unique(pairs[:, 0], return_counts=True)
    return vert[count > 1]


def tri_area(verts, faces):
    """Total surface area."""
    t = np.asarray(verts)[np.asarray(faces)]
    return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)


def uv_winding(uv, faces):
    """Signed UV areas; a fold flips a sign."""
    t = np.asarray(uv)[np.asarray(faces)]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _seamless():
    """One texcoord per vertex; splitting must change nothing."""
    verts = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0., 0], [1, 0], [1, 1], [0, 1]])
    return verts, faces, uv, faces.copy()


def _quad():
    """Two triangles sharing an edge, shared vertices on a UV seam."""
    verts = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0., 0], [1, 0], [1, 1], [0.5, 0], [0.5, 1], [0, 1]])
    face_uv = np.array([[0, 1, 2], [3, 4, 5]])   # vertices 0 and 2 each own two UVs
    return verts, faces, uv, face_uv


def self_test():
    """Twenty-two controls. Five must reject input a naive bind would have accepted."""
    results = []
    v, f, uv, fuv = _quad()

    seams = seam_vertices(f, fuv)
    results.append(("the fixture has seam vertices to split", sorted(seams) == [0, 2]))

    v2, f2, uv2 = split(v, f, uv, fuv)
    results.append(("split gives one texcoord per vertex", len(v2) == len(uv2) == 6))
    results.append(("triangle count is unchanged", f2.shape == f.shape))
    results.append(("positions are duplicated, never moved",
                    np.allclose(v2[f2], v[f])))
    results.append(("every corner keeps its authored uv",
                    np.allclose(uv2[f2], uv[fuv])))
    bad = naive(v, f, uv, fuv)
    results.append(("a naive per-vertex bind is shown to lose a uv",
                    not np.allclose(bad[f], uv[fuv])))

    for args, why in (((v, f, uv, fuv[:1]), "mismatched face counts"),
                      ((v, f, uv[:2], fuv), "a uv index out of range"),
                      ((v[:2], f, uv, fuv), "a vertex index out of range")):
        try:
            split(*args)
            results.append(("split rejects %s" % why, False))
        except ValueError:
            results.append(("split rejects %s" % why, True))

    results.append(("the v flip is an involution",
                    np.allclose(to_mitsuba_uv(to_mitsuba_uv(uv)), uv)))
    results.append(("the v flip actually changes v",
                    not np.allclose(to_mitsuba_uv(uv)[:, 1], uv[:, 1])))
    results.append(("the v flip leaves u alone",
                    np.allclose(to_mitsuba_uv(uv)[:, 0], uv[:, 0])))
    sv, sf, suv, sfuv = _seamless()
    sv2, sf2, suv2 = split(sv, sf, suv, sfuv)
    results.append(("a seamless mesh is not grown by the split", len(sv2) == len(sv)))
    results.append(("a seamless mesh keeps its uv", np.allclose(suv2[sf2], suv[sfuv])))
    results.append(("the seam fixture is not accidentally seamless",
                    len(seam_vertices(f, fuv)) > len(seam_vertices(sf, sfuv))))

    results.append(("the split is deterministic across calls",
                    all(np.array_equal(a, b) for a, b in
                        zip(split(v, f, uv, fuv), split(v, f, uv, fuv)))))
    results.append(("surface area is preserved",
                    abs(tri_area(v2, f2) - tri_area(v, f)) < 1e-12))
    results.append(("no output vertex is left unreferenced",
                    len(np.unique(f2)) == len(v2)))
    results.append(("corner order within a face is unchanged",
                    np.array_equal(np.unique(np.stack([f.reshape(-1), fuv.reshape(-1)], 1),
                                             axis=0)[f2.reshape(-1), 0].reshape(f.shape), f)))
    results.append(("uv winding signs are unchanged",
                    np.array_equal(np.sign(uv_winding(uv2, f2)),
                                   np.sign(uv_winding(uv, fuv)))))
    perm = fuv.copy(); perm[0] = perm[0][::-1]
    results.append(("permuting a face's uv corners changes the split",
                    not np.allclose(split(v, f, uv, perm)[2][split(v, f, uv, perm)[1]],
                                    uv2[f2])))

    results.append(_against_mitsuba())

    bad_count = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(results) - bad_count, len(results)))
    return 1 if bad_count else 0


def _against_mitsuba():
    """Our split against Mitsuba's obj loader. Unavailable is a FAIL, never a skip."""
    import pathlib
    import tempfile
    try:
        import mitsuba as mi
        mi.set_variant("scalar_rgb")
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                               / "3-interactor" / "anny" / "src"))
        from anny import Anny
    except Exception as error:  # noqa: BLE001
        return ("upstream obj loader reachable for the differential control: %s"
                % type(error).__name__, False)

    m = Anny()
    verts = m.template_vertices.detach().cpu().numpy().astype(np.float64)
    faces = m.faces.detach().cpu().numpy().astype(np.int64)
    uv = m.texture_coordinates.detach().cpu().numpy().astype(np.float64)
    fuv = m.face_texture_coordinate_indices.detach().cpu().numpy().astype(np.int64)
    v2, f2, uv2 = split(verts, faces, uv, fuv)

    path = pathlib.Path(tempfile.mkdtemp()) / "anny.obj"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join("v %.8f %.8f %.8f\n" % tuple(p) for p in verts))
        fh.write("".join("vt %.8f %.8f\n" % tuple(t) for t in uv))
        for tri, tuv in zip(faces + 1, fuv + 1):
            fh.write("f %d/%d %d/%d %d/%d\n"
                     % (tri[0], tuv[0], tri[1], tuv[1], tri[2], tuv[2]))
    mesh = mi.load_dict({"type": "obj", "filename": str(path)})
    theirs = np.array(mesh.vertex_texcoords_buffer()).reshape(-1, 2).astype(np.float64)
    their_faces = np.array(mesh.faces_buffer()).reshape(-1, 3).astype(np.int64)
    seams = len(seam_vertices(faces, fuv))
    if seams == 0 or len(v2) <= len(np.unique(faces)):
        return ("the anny mesh carries uv seams for the split to act on", False)

    unflipped = float(np.abs(uv2[f2] - theirs[their_faces]).max())
    if unflipped <= float(np.finfo(np.float32).eps):
        return ("the v flip is required, not decoration (unflipped %.3e)" % unflipped, False)

    ours = to_mitsuba_uv(uv2)[f2]
    worst = float(np.abs(ours - theirs[their_faces]).max())
    # float32 tolerance: the obj loader stores texcoords as float32.
    return ("%d seam vertices, flip required (%.3e), split matches mitsuba (%.3e)"
            % (seams, unflipped, worst),
            mesh.vertex_count() == len(v2) and worst <= float(np.finfo(np.float32).eps))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    ap.error("nothing to do; pass --self-test")


if __name__ == "__main__":
    sys.exit(main())
