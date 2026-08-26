"""Do the pose and depth conditioning agree with the frame they were made for?

A corpus row is a render plus the signals a generator is conditioned on: a depth map and a
set of keypoints. Those three are only worth anything if they describe the same picture, and
nothing here checked that. The loop asked EditScore instead, which answers a perceptual
question about two images and returned a floor score of 0.0 on every pair, because the
question it was asked was whether one render is another pose's photograph.

This asks the question the corpus actually needs, and it is arithmetic rather than a model:

1. SILHOUETTE. Rasterise the mesh through the frame's own camera and compare the depth
   mask against the render's alpha. They are the same body through the same camera, so the
   intersection over union is near one, and a wrong camera or a wrong subject drops it.
2. KEYPOINTS. Project the 104 joints through that camera and count how many land inside the
   silhouette. A joint outside is measured in pixels from the body rather than counted, so
   the report says how far wrong rather than only that something is.
3. DEPTH RANGE. The near and far the depth was normalised against are written beside it,
   because a ControlNet conditioned on a per-frame normalisation with no recorded range
   learns the normalisation and not the body.

WHAT THIS CANNOT SEE. A mirrored subject, when the subject is symmetric: the silhouette is
the same and the joints still land inside. `check_conventions.py` catches that with signed
mesh volume and the joint basis determinants, and this file does not repeat the attempt.

Every camera value is read from the frame's sidecar rather than recomputed, so this checks
the render that exists and not a render it would make now.

    python check_controls.py <frame.png> --rig <rest-skel.npz> [--self-test]
    python check_controls.py <directory> --rig <rest-skel.npz>
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
from PIL import Image

import render_corpus
import render_view

# A joint may sit a pixel or two outside a thin limb because the mask is a rasterisation of
# triangles and the render is a path trace with a filter. Beyond this it is a real
# disagreement rather than an edge.
EDGE_TOLERANCE_PX = 3.0
IOU_FLOOR = 0.95
UP = np.array([0.0, 0.0, 1.0])


def load_rig(path):
    data = np.load(path)
    names_file = pathlib.Path(path).with_suffix(".names.json")
    names = json.loads(names_file.read_text(encoding="utf-8")) if names_file.is_file() else []
    return (np.asarray(data["verts"], dtype=np.float64),
            np.asarray(data["faces"], dtype=np.int64),
            np.asarray(data["bone_poses"][:, :3, 3], dtype=np.float64),
            names)


def controls_for(frame_png, verts, faces, joints):
    """The depth mask and the projected joints for one rendered frame."""
    side = json.loads(pathlib.Path(frame_png).with_suffix(".json").read_text(encoding="utf-8"))
    image = Image.open(frame_png).convert("RGBA")
    width, height = image.size
    alpha = np.asarray(image)[:, :, 3] > 127

    centre = np.asarray(side["normalisation"]["centre"], dtype=np.float64)
    scale = float(side["normalisation"]["scale"])
    eye = np.asarray(side["eye"], dtype=np.float64)
    view = render_corpus.look_at(eye, np.zeros(3), UP)

    verts_px, verts_depth = render_corpus.project(
        (verts - centre) * scale, view, side["fov_deg"], width, height)
    zbuf, mask = render_corpus.rasterise_depth(verts_px, verts_depth, faces, width, height)
    joints_px, _ = render_corpus.project(
        (joints - centre) * scale, view, side["fov_deg"], width, height)
    return {"alpha": alpha, "mask": mask, "zbuf": zbuf, "joints_px": joints_px,
            "size": (width, height), "sidecar": side}


def distance_to_mask(mask, points):
    """Pixels from each point to the nearest covered pixel. Zero when the point is inside."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.full(len(points), np.inf)
    covered = np.stack([xs, ys], axis=1).astype(np.float64)
    out = np.empty(len(points))
    for i, p in enumerate(points):
        out[i] = np.sqrt(((covered - p) ** 2).sum(axis=1).min())
    return out


def measure(frame_png, verts, faces, joints):
    c = controls_for(frame_png, verts, faces, joints)
    alpha, mask = c["alpha"], c["mask"]
    union = np.logical_or(alpha, mask).sum()
    iou = float(np.logical_and(alpha, mask).sum() / union) if union else 0.0

    width, height = c["size"]
    px = c["joints_px"]
    on_frame = ((px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height))
    inside = np.zeros(len(px), dtype=bool)
    idx = np.clip(px.astype(int), [0, 0], [width - 1, height - 1])
    inside[on_frame] = mask[idx[on_frame, 1], idx[on_frame, 0]]

    outside = np.nonzero(~inside)[0]
    distances = distance_to_mask(mask, px[outside]) if len(outside) else np.array([])
    far = outside[distances > EDGE_TOLERANCE_PX] if len(outside) else np.array([], dtype=int)

    near, farz = (float(c["zbuf"][mask].min()), float(c["zbuf"][mask].max())) if mask.any() else (0.0, 0.0)
    return {
        "iou": iou, "inside": int(inside.sum()), "joints": len(px),
        "off_frame": int((~on_frame).sum()),
        "worst_px": float(distances.max()) if len(distances) else 0.0,
        "far_indices": far.tolist(),
        "depth_near": near, "depth_far": farz,
    }


def check(frame_png, verts, faces, joints, names):
    m = measure(frame_png, verts, faces, joints)
    problems = []
    if m["iou"] < IOU_FLOOR:
        problems.append(f"silhouette IoU {m['iou']:.3f} is under {IOU_FLOOR}: the depth "
                        f"control and the render are not the same body through the same camera")
    if m["far_indices"]:
        named = [names[i] if i < len(names) else str(i) for i in m["far_indices"][:4]]
        problems.append(f"{len(m['far_indices'])} joints sit over {EDGE_TOLERANCE_PX:.0f} px "
                        f"outside the silhouette, worst {m['worst_px']:.1f} px: {named}")
    if m["depth_far"] <= m["depth_near"]:
        problems.append("the depth range is empty, so its normalisation is not invertible")
    return problems, m


def error_score(m) -> dict:
    """The disagreement as an ERROR, where zero is perfect and larger is worse.

    A quality score is the wrong shape for this loop. EditScore returns one, and its rubric
    scores whether an edit succeeded, so a perfect correspondence reads 0.0 and a 45 degree
    camera change reads 2.19: higher is not better and the number does not order the cases.
    An error orders them, and it is already in the units this file measures.

    Two components rather than one weighted number, because they fail differently and a
    weighting would be a choice hidden inside a scalar. Silhouette error catches a wrong
    camera or a wrong subject. Keypoint error catches a joint that left the body while the
    outline stayed put, which is rule 4's failure and the one a silhouette cannot see.
    """
    outside = len(m["far_indices"])
    return {
        "silhouette_error": round(1.0 - m["iou"], 5),
        "keypoints_outside": outside,
        "worst_joint_px": round(m["worst_px"], 2),
        "keypoint_error_fraction": round(outside / max(m["joints"], 1), 5),
    }


def report(name, problems, m):
    print(f"  {name}")
    print(f"    silhouette IoU {m['iou']:.4f}")
    print(f"    keypoints      {m['inside']} of {m['joints']} inside, "
          f"{m['off_frame']} off frame, worst {m['worst_px']:.1f} px out")
    print(f"    depth range    {m['depth_near']:.4f} to {m['depth_far']:.4f} camera units")
    for p in problems:
        print(f"    BAD  {p}")


def rotate(points, degrees):
    """Turn the subject about the up axis, which a wrong facing does to a whole set."""
    a = np.radians(degrees)
    m = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return points @ m.T


def self_test(frame_png, verts, faces, joints, names) -> int:
    """Each control breaks one correspondence, and each must be caught."""
    fails = []
    print("negative controls")
    # A MIRROR IS NOT IN SCOPE HERE AND SAYING SO IS THE POINT. The rest pose is left-right
    # symmetric, so mirroring it produces nearly the same silhouette and the joints still
    # land inside: this check accepted it. Mirroring is caught by check_conventions.py,
    # which reads signed mesh volume and the joint basis determinants, and neither of those
    # is a silhouette. A control that cannot fail belongs in the other file, not here.
    cases = {
        "the mesh scaled by 1.2": (verts * 1.2, faces, joints * 1.2),
        "the joints shifted 40 px worth": (verts, faces, joints + np.array([0.06, 0.0, 0.0])),
        "the mesh rotated 15 degrees about up": (rotate(verts, 15.0), faces, rotate(joints, 15.0)),
    }
    for label, (v, f, j) in cases.items():
        problems, _ = check(frame_png, v, f, j, names)
        if problems:
            print(f"  ok  {label}: rejected ({problems[0][:62]})")
        else:
            fails.append(label)
            print(f"  BAD {label}: accepted")

    print("positive control")
    problems, m = check(frame_png, verts, faces, joints, names)
    if problems:
        fails.append("the frame as rendered")
        report("the frame as rendered", problems, m)
    else:
        print(f"  ok  the frame as rendered: accepted, IoU {m['iou']:.4f}, "
              f"{m['inside']} of {m['joints']} joints inside")
    print(f"\n{len(fails)} failed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="a rendered frame, or a directory of them")
    ap.add_argument("--rig", required=True, help="the rest-skel .npz the frames were rendered from")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--error", action="store_true",
                    help="print the error score for each frame rather than only failures")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    verts, faces, joints, names = load_rig(args.rig)
    target = pathlib.Path(args.target)
    frames = ([target] if target.is_file()
              else [pathlib.Path(p) for p in sorted(glob.glob(str(target / "pose_*.png")))])
    if not frames:
        raise SystemExit(f"no frames at {target}")
    if args.limit:
        frames = frames[: args.limit]

    if args.self_test:
        return self_test(frames[0], verts, faces, joints, names)

    worst_iou, total_out, bad = 1.0, 0, 0
    for frame in frames:
        problems, m = check(frame, verts, faces, joints, names)
        worst_iou = min(worst_iou, m["iou"])
        total_out += len(m["far_indices"])
        if args.error:
            e = error_score(m)
            print(f"  {frame.name}: silhouette_error {e['silhouette_error']:.5f}  "
                  f"outside {e['keypoints_outside']}  worst {e['worst_joint_px']:.2f} px")
        if problems:
            bad += 1
            report(frame.name, problems, m)
    print(f"{len(frames)} frame(s), worst IoU {worst_iou:.4f}, "
          f"{total_out} joint(s) outside the silhouette, {bad} frame(s) with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
