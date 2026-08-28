"""Scale, up, forward and handedness, measured off a rig rather than assumed.

CLAUDE.md rule 6 says conventions are data: parse rotation order, up axis and units, never
assume them. Three defects in one week came from breaking that rule, and each produced a
result that looked fine.

* The fit projected components 0 and 1 while the rig is Z-up, so the solver rotated every
  body ninety degrees to reach the pixels and reported 0.022% of stature.
* `stature` measured the body's DEPTH for the same reason, so every percent-of-stature
  figure was normalised by 0.434 where 1.660 belonged.
* The camera grid placed cameras at world azimuth while its phrases claim a subject's
  front, so ninety-six frames labelled "front view" were profiles.

None of the three raised anything. This script raises, and it drives each check with a
broken rig first, because a check that passes on known-broken input certifies the defect.

WHAT IS MEASURED, all four from the geometry:

1. UP is the longest rest extent, and it must be longer than the next by a clear margin.
2. SCALE is stature along that axis, reported in metres with a household equivalent, and
   compared against the stage's own metersPerUnit when reading a USD.
3. FORWARD is the chest: the normal to the shoulder line, which is where the ribs point.
   When a rig has no shoulders the witness falls back in a declared order: clavicles, upper
   arms, upper legs, then the span between the wrists, which is the three-point headset
   case where a head and two controllers are all there is, then the gaze, then the feet.
   The witness used is reported and running out of them raises. A gaze forward is where the
   head looks rather than where the body faces, and the report says so. The gaze and the feet are reported beside the chest and must
   agree with it on an UNPOSED rig. On a posed rig they need not: a stance can plant the
   feet away from the chest.
4. HANDEDNESS has three witnesses. The signed volume of the mesh flips under any
   reflection. Every joint basis has determinant +1 for a rotation and -1 for a reflection.
   The `.R` against `.L` split is reported beside them and catches crossed limbs rather
   than mirroring, which is a different defect.

THE CONVENTION THIS RIG ACTUALLY USES, measured rather than declared:

    up          +Z          longest rest extent, 1.59x the next axis
    forward     -Y          the chest normal, azimuth 270, with gaze and feet agreeing to 0.0
    right       -X          forward cross up, and the .R joints average +0.300 on it
    handedness  right       every joint basis determinant +1.000000, signed volume +0.05120

No 180 degree correction is needed anywhere. A front view places the camera in front of the
chest and shows the face, which is what the rendered frames show at both facings tested,
270 for the rest pose and 345.9 for the hv_1 fit. If the sign were inverted the same frames
would show the back, so the render is the check.

    python check_conventions.py <rig.npz|rig.usda> [--self-test]
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

# A stature outside this band is a units error rather than an unusual body: 1.66 m is ANNY,
# 166 would be centimetres read as metres and 0.0166 would be the reverse.
STATURE_M = (1.2, 2.3)
UP_MARGIN = 1.2          # the tall axis must beat the next by this factor
FORWARD_TOLERANCE_DEG = 25.0

ANCHORS = (("credit card", 0.76), ("penny", 1.52), ("pencil", 7.0), ("AAA battery", 10.5),
           ("AA battery", 14.5), ("nickel", 21.2), ("golf ball", 42.7), ("adult wrist", 57.0),
           ("soda can", 66.0))


def household(mm: float) -> str:
    name, size = min(ANCHORS, key=lambda a: abs(mm - a[1]))
    n = mm / size
    return f"about one {name}" if n < 1.5 else f"about {n:.0f} {name}s stacked"


def read_npz(path):
    """The rig from the arrays the renderer itself loads. No USD reader involved.

    `mesh_to_usda.py` writes a `.usda` for the archive, and reading it back would put a USD
    library in the render environment, which renders with mitsuba and needs nothing else.
    The npz beside it carries the same vertices and the same bind transforms, so the check
    runs on the renderer's own input.
    """
    import json
    import pathlib

    data = np.load(path)
    names_file = pathlib.Path(path).with_suffix(".names.json")
    names = json.loads(names_file.read_text(encoding="utf-8")) if names_file.is_file() else []
    joints = (data["bone_poses"][:, :3, 3] if "bone_poses" in data
              else np.zeros((0, 3), dtype=np.float64))
    return {
        "points": np.asarray(data["verts"], dtype=np.float64),
        "joints": np.asarray(joints, dtype=np.float64),
        "faces": np.asarray(data["faces"]) if "faces" in data else None,
        "bases": np.asarray(data["bone_poses"]) if "bone_poses" in data else None,
        "names": names,
        "meters_per_unit": 1.0,
        "up_axis": "",
    }


def read_rig(path):
    """Either form. The npz is the renderer's input and the usda is the archive."""
    return read_npz(path) if str(path).lower().endswith(".npz") else read_usd(path)


def read_usd(path):
    from pxr import Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.Open(str(path))
    mesh = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    skel_prim = next(p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    skel = UsdSkel.Skeleton(skel_prim)
    names = list(skel_prim.GetAttribute("annyJointNames").Get() or [])
    binds = np.asarray(skel.GetBindTransformsAttr().Get(), dtype=np.float64)
    joints = binds[:, 3, :3]          # USD is row-vector, so translation is the last row

    # Faces and bases, WITHOUT WHICH TWO CHECKS REPORT NOT RUN ON EVERY ARCHIVE. USD carries
    # both; not reading them left the mirror and handedness tests dead on the .usda path.
    counts = UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get()
    indices = UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get()
    faces, at = [], 0
    for n in (counts or []):
        fan = indices[at:at + n]
        faces.extend([fan[0], fan[i], fan[i + 1]] for i in range(1, n - 1))
        at += n
    return {
        "points": points, "joints": joints, "names": names,
        "faces": np.asarray(faces, dtype=np.int64) if faces else None,
        "bases": np.transpose(binds, (0, 2, 1)) if len(binds) else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
    }


def measure(rig) -> dict:
    """Every convention, from the geometry, with no default anywhere."""
    points, joints, names = rig["points"], rig["joints"], rig["names"]
    index = {n: i for i, n in enumerate(names)}
    extent = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(extent)[::-1]
    up_axis, next_axis = int(order[0]), int(order[1])

    out = {
        "extent": extent, "up_axis": up_axis,
        "up_margin": float(extent[up_axis] / max(extent[next_axis], 1e-12)),
        "stature_m": float(extent[up_axis] * rig["meters_per_unit"]),
    }
    up = np.zeros(3)
    up[up_axis] = 1.0
    out["up"] = up

    def joint(name):
        if name not in index:
            raise KeyError(f"the rig has no joint named {name}")
        return joints[index[name]]

    # FORWARD IS THE CHEST. A body faces where its ribs point: the normal to the line across
    # the shoulders. The gaze and the feet are secondary, and on a posed rig they are not
    # even wrong witnesses, they are different questions. A person can plant their feet and
    # turn their head, and the hv_1 fit does exactly that, with the feet 55 degrees off the
    # chest while the chest and the gaze agree to 3.
    #
    # The sign is fixed against the rest pose, where the answer is known independently:
    # cross(up, shoulder) gives 270 and cross(shoulder, up) gives 90, and the rest rig faces
    # 270 by both the gaze and the feet. pelvis.L and pelvis.R are coincident at the origin
    # in this rig, separation 0.0, so a hip witness is degenerate and is not used.
    def optional(*wanted):
        """The joints if the rig has all of them, else None. A rig is not required to be
        ANNY, and a missing joint is a reason to try the next witness rather than to stop."""
        if any(nm not in index for nm in wanted):
            return None
        return [joints[index[nm]] for nm in wanted]

    def span(pair):
        """A left-to-right span, refused when the two joints are too close to give a
        direction. `pelvis.L` and `pelvis.R` are coincident in this rig, separation 0.0,
        because the hips are the local root, and a zero cross product reads as azimuth 0
        rather than as no answer."""
        if pair is None:
            return None
        v = pair[0] - pair[1]
        return None if np.linalg.norm(v) < 1e-3 * max(extent.max(), 1e-9) else np.cross(up, v)

    # THE CHEST FIRST, THEN A DECLARED FALLBACK ORDER. A rig without `shoulder01` is not a
    # broken rig, it is a different rig, so each witness is tried in turn and the one used
    # is reported. What is refused is running out of witnesses, because a default forward
    # would be an assumption wearing a measurement's clothes.
    ladder = (
        ("chest, from shoulder01", span(optional("shoulder01.R", "shoulder01.L"))),
        ("chest, from clavicle", span(optional("clavicle.R", "clavicle.L"))),
        ("chest, from upperarm01", span(optional("upperarm01.R", "upperarm01.L"))),
        ("hips, from upperleg01", span(optional("upperleg01.R", "upperleg01.L"))),
        # THE HEADSET CASE. Three-point tracking gives a head and two controllers and
        # nothing else: no shoulders, no clavicles, no legs. The span between the hands is
        # the only body-width witness left, and crossing it with up is the same
        # construction the shoulders use. It is worse than the shoulders and better than
        # the gaze, because hands wander but a head turns independently of the chest by
        # design.
        ("chest, from the wrists, three-point tracking",
         span(optional("wrist.R", "wrist.L"))),
        # LAST, AND IT IS A DIFFERENT QUANTITY. A gaze forward is where the head looks,
        # not where the body faces, and a headset reports exactly that. When this is the
        # witness the report says so, because reading it as a chest is the error.
        ("gaze, from the eyes, WHICH IS NOT THE CHEST",
         (lambda p: None if p is None else (p[0] + p[1]) / 2 - p[2])(
             optional("eye.L", "eye.R", "head"))),
        ("feet, from the toes", (lambda p: None if p is None else
                                 (p[0] + p[1]) / 2 - (p[2] + p[3]) / 2)(
            optional("toe3-1.L", "toe3-1.R", "foot.L", "foot.R"))),
    )
    usable = [(name, v) for name, v in ladder
              if v is not None and np.linalg.norm(v - up * float(v @ up)) > 1e-9]
    if not usable:
        raise KeyError("no forward witness: the rig has no shoulders, clavicles, upper "
                       "arms, upper legs, wrists, eyes or toes among "
                       f"{len(names)} joints")
    out["forward_witness"] = usable[0][0]
    out["forward_fallbacks"] = [name for name, _ in usable[1:]]
    out["forward_witnesses_absent"] = [name for name, v in ladder if v is None]

    chest = usable[0][1]
    eyes_pair = optional("eye.L", "eye.R", "head")
    toes_pair = optional("toe3-1.L", "toe3-1.R", "foot.L", "foot.R")
    eyes = chest if eyes_pair is None else (eyes_pair[0] + eyes_pair[1]) / 2 - eyes_pair[2]
    toes = chest if toes_pair is None else ((toes_pair[0] + toes_pair[1]) / 2
                                            - (toes_pair[2] + toes_pair[3]) / 2)
    flat = [v - up * float(v @ up) for v in (chest, eyes, toes)]
    flat = [v / max(np.linalg.norm(v), 1e-12) for v in flat]
    out["forward_chest"], out["forward_eyes"], out["forward_toes"] = flat
    out["forward"] = flat[0]

    def between(a, b):
        return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))

    out["gaze_off_chest_deg"] = between(flat[0], flat[1])
    out["feet_off_chest_deg"] = between(flat[0], flat[2])
    # A rest rig has all three aligned. That is what makes it the rest rig, and the check
    # for it is here rather than on a posed body, where a turn is a pose and not a defect.
    out["forward_disagreement_deg"] = max(out["gaze_off_chest_deg"], out["feet_off_chest_deg"])

    right = np.cross(out["forward"], up)
    out["right"] = right
    r_side = [joints[index[n]] for n in names if n.endswith(".R")]
    l_side = [joints[index[n]] for n in names if n.endswith(".L")]
    out["right_dot"] = float(np.mean([p @ right for p in r_side]))
    out["left_dot"] = float(np.mean([p @ right for p in l_side]))
    out["sides_split"] = out["right_dot"] > 0 > out["left_dot"]

    # THE SIDE TEST CANNOT CATCH A MIRROR ON ITS OWN, WHICH THE MIRRORED CONTROL SHOWED.
    # `right` is built from the chest, the chest mirrors with the body, so .R lands positive
    # again and the test passes. Signed volume does not mirror away: summing the tetrahedra
    # of a closed mesh gives a positive number for outward winding and flips sign under any
    # reflection, whatever the joints are named.
    # THE JOINT BASES ARE A THIRD WITNESS, AND THEY COST NOTHING. A rotation has
    # determinant +1 and a reflection has -1, so a mirrored export shows it in every joint
    # basis whether or not the mesh came along. Measured: the rest rig and the hv_1 fit both
    # read +1.000000 at the root and across all 104, and mirroring one axis takes all 104 to
    # -1. The hips are the local root here, pelvis.L and pelvis.R both sitting at the origin
    # with separation 0.0, which is also why a hip-width witness for forward is degenerate.
    bases = rig.get("bases")
    if bases is not None and len(bases):
        dets = np.linalg.det(np.asarray(bases)[:, :3, :3])
        out["root_det"] = float(dets[0])
        out["negative_bases"] = int((dets < 0).sum())
    else:
        out["root_det"], out["negative_bases"] = None, None

    faces = rig.get("faces")
    if faces is not None and len(faces):
        tri = points[np.asarray(faces, dtype=np.int64)]
        out["signed_volume"] = float(
            np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    else:
        out["signed_volume"] = None
    out["handed"] = (out["sides_split"]
                     and (out["signed_volume"] is None or out["signed_volume"] > 0)
                     and (out["negative_bases"] in (None, 0)))
    return out


def check(rig) -> tuple[list, dict]:
    m = measure(rig)
    problems = []
    axis = "XYZ"[m["up_axis"]]

    if m["up_margin"] < UP_MARGIN:
        problems.append(f"up axis is not clear: extents {np.round(m['extent'], 3).tolist()} "
                        f"put {axis} only {m['up_margin']:.2f}x above the next")
    declared = rig.get("up_axis")
    if declared and declared.upper() != axis:
        problems.append(f"the stage declares up {declared} and the geometry measures {axis}")

    if not STATURE_M[0] <= m["stature_m"] <= STATURE_M[1]:
        problems.append(f"stature {m['stature_m']:.3f} m is outside {STATURE_M}, "
                        f"which is a units error rather than a body")

    if not rig.get("posed") and m["forward_disagreement_deg"] > FORWARD_TOLERANCE_DEG:
        problems.append(
            f"the witnesses disagree on a rig declared unposed: gaze is "
            f"{m['gaze_off_chest_deg']:.1f} deg off the chest and the feet are "
            f"{m['feet_off_chest_deg']:.1f} deg off it. On a POSED rig this is a stance "
            f"rather than a defect; pass posed=True and read the chest.")

    if m["negative_bases"]:
        problems.append(f"{m['negative_bases']} joint bases have a negative determinant, "
                        f"root {m['root_det']:+.6f}: a rotation is +1 and a reflection is -1")
    elif m["negative_bases"] is None:
        problems.append("joint bases NOT RUN: the rig carries no bone transforms")
    if m["signed_volume"] is not None and m["signed_volume"] <= 0:
        problems.append(f"the mesh is mirrored: signed volume {m['signed_volume']:+.5f}, "
                        f"and a closed mesh with outward winding is positive")
    elif m["signed_volume"] is None:
        problems.append("signed volume NOT RUN: no faces in the rig, so a mirror would pass")
    if not m["sides_split"]:
        problems.append(f"the named sides do not split: .R joints average "
                        f"{m['right_dot']:+.3f} and .L average {m['left_dot']:+.3f} on the "
                        f"right axis, so limbs cross the body or a side is mislabelled")
    return problems, m


def report(name, problems, m):
    axis = "XYZ"[m["up_axis"]]
    azimuth = math.degrees(math.atan2(m["forward"][1], m["forward"][0])) % 360
    mm = m["stature_m"] * 1000
    print(f"  {name}")
    print(f"    up         {axis}, {m['up_margin']:.2f}x the next axis")
    print(f"    scale      stature {m['stature_m']:.3f} m, {mm:.0f} mm, "
          f"{household(mm / 25)} at 1/25 scale")
    print(f"    forward    azimuth {azimuth:.1f} deg from {m['forward_witness']}, "
          f"gaze {m['gaze_off_chest_deg']:.1f} deg off, feet "
          f"{m['feet_off_chest_deg']:.1f} deg off")
    if m["forward_witnesses_absent"]:
        print(f"               absent: {', '.join(m['forward_witnesses_absent'])}")
    volume = "no faces" if m["signed_volume"] is None else f"{m['signed_volume']:+.5f}"
    det = "no bases" if m["root_det"] is None else f"{m['root_det']:+.6f}"
    print(f"    handed     signed volume {volume}, root det {det}, "
          f".R {m['right_dot']:+.3f}, .L {m['left_dot']:+.3f}")
    for p in problems:
        print(f"    BAD  {p}")


def broken(rig, kind):
    """A rig with one convention deliberately wrong. Each must be rejected."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rig.items()}
    out["names"] = list(rig["names"])
    if kind == "y up":
        swap = [0, 2, 1]
        out["points"], out["joints"] = rig["points"][:, swap], rig["joints"][:, swap]
        out["up_axis"] = "Z"
    elif kind == "centimetres":
        out["points"], out["joints"] = rig["points"] * 100, rig["joints"] * 100
    elif kind == "mirrored":
        out["points"], out["joints"] = rig["points"].copy(), rig["joints"].copy()
        out["points"][:, 0] *= -1
        out["joints"][:, 0] *= -1
        if rig.get("bases") is not None:
            bases = np.asarray(rig["bases"]).copy()
            bases[:, :3, :3] = np.diag([-1.0, 1.0, 1.0]) @ bases[:, :3, :3]
            out["bases"] = bases
    elif kind == "eyes moved behind the head":
        out["joints"] = rig["joints"].copy()
        index = {n: i for i, n in enumerate(rig["names"])}
        head = rig["joints"][index["head"]]
        for eye in ("eye.L", "eye.R"):
            out["joints"][index[eye]] = head - (rig["joints"][index[eye]] - head)
    elif kind.startswith("no "):
        # A rig missing a witness must fall back and say which one it used, not stop.
        if kind == "no body, a headset and two controllers":
            keep_names = {"head", "eye.L", "eye.R", "wrist.L", "wrist.R"}
            keep = [i for i, n in enumerate(rig["names"]) if n in keep_names]
            out["names"] = [rig["names"][i] for i in keep]
            out["joints"] = rig["joints"][keep]
            if rig.get("bases") is not None:
                out["bases"] = np.asarray(rig["bases"])[keep]
            return out
        drop = {"no shoulders": ("shoulder01.L", "shoulder01.R"),
                "no shoulders or clavicles": ("shoulder01.L", "shoulder01.R",
                                              "clavicle.L", "clavicle.R"),
                "no forward witness at all": tuple(
                    n for n in rig["names"]
                    if n.split(".")[0] in {"shoulder01", "clavicle", "upperarm01",
                                           "upperleg01", "wrist", "eye", "head",
                                           "toe3-1", "foot"})}[kind]
        keep = [i for i, n in enumerate(rig["names"]) if n not in drop]
        out["names"] = [rig["names"][i] for i in keep]
        out["joints"] = rig["joints"][keep]
        if rig.get("bases") is not None:
            out["bases"] = np.asarray(rig["bases"])[keep]
    elif kind == "a sphere, with no clear tall axis":
        out["points"] = rig["points"] / (rig["points"].max(axis=0) - rig["points"].min(axis=0))
    else:
        raise ValueError(kind)
    return out


def self_test(rig) -> int:
    fails = []
    print("negative controls, one broken convention each")
    for kind in ("y up", "centimetres", "mirrored", "eyes moved behind the head",
                 "a sphere, with no clear tall axis", "no forward witness at all"):
        try:
            problems, _ = check(broken(rig, kind))
        except Exception as error:  # noqa: BLE001
            print(f"  ok  {kind}: rejected ({type(error).__name__})")
            continue
        if problems:
            print(f"  ok  {kind}: rejected ({problems[0][:66]})")
        else:
            fails.append(kind)
            print(f"  BAD {kind}: accepted, so this gate certifies the defect")

    print("fallbacks, which must be taken and named rather than refused")
    for kind, wanted in (("no shoulders", "clavicle"),
                         ("no shoulders or clavicles", "upperarm01"),
                         ("no body, a headset and two controllers", "wrists")):
        try:
            problems, m = check(broken(rig, kind))
        except KeyError as error:
            fails.append(kind)
            print(f"  BAD {kind}: raised instead of falling back ({str(error)[:50]})")
            continue
        used = m["forward_witness"]
        if wanted in used and not problems:
            print(f"  ok  {kind}: fell back to {used}")
        else:
            fails.append(kind)
            print(f"  BAD {kind}: used {used!r} with {len(problems)} problem(s)")

    print("positive control")
    problems, m = check(rig)
    if problems:
        fails.append("the real rig")
        report("the rig as given", problems, m)
    else:
        print("  ok  the rig as given: accepted")
    print(f"\n{len(fails)} failed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rig", help="a rest-skel .npz, or a .usda written by mesh_to_usda.py")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rig = read_rig(args.rig)
    problems, m = check(rig)
    print("measured, not assumed")
    report(args.rig, problems, m)
    rc = 1 if problems else 0
    if args.self_test:
        print()
        rc |= self_test(rig)
    else:
        print(f"\n{len(problems)} problems")
    return rc


if __name__ == "__main__":
    sys.exit(main())
