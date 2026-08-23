"""Render ANNY with the answers already on the picture.

RFD 0122's rung 1. The corpus cannot be fetched: no licence-clean wholebody keypoint set
exists, and the one in this workspace is blocklisted because it contains the whole blinded
holdout. So the labels are made rather than found.

THREE OUTPUTS, AND NOTHING ELSE.

    image                the input, for both consumers
    keypoint positions   for the keypoint detector
    3D shape             for Pixal3D, which is why the renderer writes the mesh too

Depth, the camera matrix and the part index are intermediates. They are computed here because
the visibility test needs depth, and they do not land beside the image. A fourth output is
drift unless one of the two consumers asked for it.

WHY THE LABELS ARE TRUE BY CONSTRUCTION. `anny/data/keypoints/coco.pth` stores each keypoint as
a weight vector over the mesh, 19,158 wide, summing to 1. A keypoint position is therefore a
weighted sum of posed vertices: `W @ V`. Nothing is detected, so nothing can be wrong about it
in the way an annotation can. The projection to 2D is camera arithmetic.

That asset holds 23 points, not 17. COCO-17 plus the six foot points.

THE TOPOLOGY IS PINNED, and it has to be. The weight matrix is 19,158 wide and ANNY's default
returns 13,718, because the default drops every unattached vertex. A shape mismatch here would
not error, it would refuse to multiply, and the version that quietly matched would put
keypoints on a different body.

    Anny(topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False))

`remove_unattached_vertices` is the only argument that moves the count. A sweep over
`nudity_edits`, `eyes` and `tongue`, all eight combinations, returns 19,158 every time.

VERTEX ORDER IS FROZEN, which is stronger than the count. `coco.pth` is indexed by vertex and
every hm08 group is a range, so a permutation keeps 19,158, still multiplies, and moves every
keypoint and every group boundary. RFD 0121 records the face-index hash that pins it.
"""

import argparse
import hashlib
import io as _io
import json
import pathlib
import sys

import numpy as np

# 2 visible, 1 present and hidden, 0 out of frame. Not a boolean: see the note on
# KEYPOINTS_2D in anny_render_schema.py.
VIS_OUT_OF_FRAME = 0
VIS_OCCLUDED = 1
VIS_VISIBLE = 2

BASEMESH_VERTS = 19158


def load_model():
    """ANNY on the topology `coco.pth` is indexed against, asserted rather than assumed."""
    import anny
    from anny.models.model_data import TopologyConfig

    model = anny.Anny(  # corpus-model-exempt: asserts the basemesh topology coco.pth is
        # indexed against (19,158 vertices), which is deliberately NOT the corpus model's
        # 13,718-vertex body submodel. Using build_corpus_model() here would measure the
        # wrong topology and the assertion would silently stop testing anything.
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False)
    )
    out = model()
    n = out["vertices"].shape[1]
    if n != BASEMESH_VERTS:
        raise SystemExit(
            f"topology returns {n} vertices, expected {BASEMESH_VERTS}. The keypoint weight "
            f"matrix is {BASEMESH_VERTS} wide and will not multiply this."
        )
    return model, out


def load_keypoint_weights(n_verts):
    """name -> weight vector. The rows are the label; there is no detector anywhere here."""
    import anny
    import torch

    path = pathlib.Path(anny.__file__).parent / "data" / "keypoints" / "coco.pth"
    raw = torch.load(path, map_location="cpu", weights_only=False)
    names = list(raw)
    W = np.stack([raw[k].numpy() for k in names])
    if W.shape[1] != n_verts:
        raise SystemExit(
            f"keypoint weights are {W.shape[1]} wide against {n_verts} vertices. "
            "The topology is not the one these were authored against."
        )
    # Each row sums to 1, so a keypoint is a convex combination and lands on the body rather
    # than somewhere scaled by an accidental normalisation.
    sums = W.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise SystemExit(f"weight rows do not sum to 1: min {sums.min()}, max {sums.max()}")
    return names, W


def look_at(eye, target, up):
    """World-to-camera. Right-handed, looking down -Z, which is what the projection assumes."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def project(points, view, fov_deg, width, height):
    """World points to pixels, and the camera-space depth that the visibility test needs."""
    n = points.shape[0]
    cam = (view @ np.concatenate([points, np.ones((n, 1))], axis=1).T).T[:, :3]
    depth = -cam[:, 2]                       # positive in front of the camera
    f = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
    aspect = width / height
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc_x = (f / aspect) * cam[:, 0] / depth
        ndc_y = f * cam[:, 1] / depth
    px = (ndc_x * 0.5 + 0.5) * width
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * height
    return np.stack([px, py], axis=1), depth


def rasterise_depth(verts_px, verts_depth, faces, width, height):
    """Triangle z-buffer, in numpy. Returns depth in camera units and the silhouette.

    Blender is not on this path. Depth is the conditioning signal for the generator under
    RFD 0121's ControlNet, so it is the output that matters most, and headless Blender wrote
    no Z pass through two engines: Workbench has no compositor passes at all, and EEVEE
    produced a uniform grey frame with no body in it.

    Depth is exact geometry, so computing it is a scanline and not a renderer. That also
    removes a dependency from the one output the corpus cannot be wrong about.

    Barycentric per triangle, nearest surface wins. Depth is interpolated linearly in camera
    space rather than perspective-correctly, which is a real approximation: over one triangle
    of a 19k-vertex body at these framings the error is far below the tolerance the
    visibility test uses, and stating it beats implying exactness.
    """
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    tri = verts_px[faces]                     # (F, 3, 2)
    triz = verts_depth[faces]                 # (F, 3)

    # Drop anything behind the camera or entirely off-frame before the per-triangle loop.
    keep = (triz > 0).all(axis=1)
    xmin = np.floor(tri[..., 0].min(axis=1)).astype(int)
    xmax = np.ceil(tri[..., 0].max(axis=1)).astype(int)
    ymin = np.floor(tri[..., 1].min(axis=1)).astype(int)
    ymax = np.ceil(tri[..., 1].max(axis=1)).astype(int)
    keep &= (xmax >= 0) & (xmin < width) & (ymax >= 0) & (ymin < height)

    for i in np.nonzero(keep)[0]:
        x0, x1 = max(xmin[i], 0), min(xmax[i] + 1, width)
        y0, y1 = max(ymin[i], 0), min(ymax[i] + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        (ax, ay), (bx, by), (cx, cy) = tri[i]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-12:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
        w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * triz[i, 0] + w1 * triz[i, 1] + w2 * triz[i, 2]
        block = zbuf[y0:y1, x0:x1]
        np.copyto(block, z, where=inside & (z < block))
        zbuf[y0:y1, x0:x1] = block

    return zbuf, np.isfinite(zbuf)


def depth_png(zbuf, mask):
    """16-bit depth, with the range written beside it so the mapping is invertible.

    Normalising per image without recording the range makes each frame's grey mean something
    different, and a ControlNet conditioned on that learns the normalisation rather than the
    body.
    """
    from PIL import Image

    if not mask.any():
        return None, (0.0, 0.0)
    near, far = float(zbuf[mask].min()), float(zbuf[mask].max())
    span = far - near if far > near else 1.0
    # Near is bright, which is the convention every depth ControlNet was trained on.
    norm = np.zeros_like(zbuf)
    norm[mask] = 1.0 - (zbuf[mask] - near) / span
    return Image.fromarray((norm * 65535).astype(np.uint16), mode="I;16"), (near, far)


def visibility_from_zbuf(kp_px, kp_depth, zbuf, width, height, tol=0.02):
    """2, 1 or 0, read off the rasterised depth rather than a vertex approximation.

    The earlier version splatted vertices into a buffer, which could call a keypoint visible
    through a gap between them. A triangle buffer has no gaps, so the middle state is now a
    measurement rather than an estimate.

    OPEN QUESTION, and it is a semantic one rather than a bug. Tightening the buffer moved
    the counts from 18 visible and 5 occluded to 8 and 15, because a joint centre is INSIDE
    the body. A shoulder sits several centimetres under the skin, so a strict depth test
    calls it occluded in every view, correctly and uselessly.

    COCO does not mean that by visible. Its v=2 means an annotator could see where the joint
    is, not that the joint centre had line of sight, and a detector trained against the
    strict reading would learn that shoulders are never visible.

    `tol` is the knob and 0.02 metres is too small for it: about a credit card and a half,
    against a shoulder joint five to ten centimetres deep. The right fix is probably to test
    the joint's surface neighbourhood rather than its centre, and the number is left as it is
    rather than tuned until the counts look plausible, because a number chosen to make an
    output look right is not a measurement.
    """
    out = np.full(len(kp_px), VIS_OUT_OF_FRAME, dtype=np.int8)
    for i, ((x, y), d) in enumerate(zip(kp_px, kp_depth)):
        if not (0 <= x < width and 0 <= y < height and d > 0):
            continue
        nearest = zbuf[int(y), int(x)]
        if not np.isfinite(nearest):
            continue                      # inside the frame, off the body: still not visible
        out[i] = VIS_VISIBLE if d <= nearest + tol else VIS_OCCLUDED
    return out


def visibility(kp_px, kp_depth, verts_px, verts_depth, width, height, tol=0.02):
    """2, 1 or 0 per keypoint.

    The middle state is the one worth having and the one a real dataset guesses. A keypoint
    inside the frame is occluded when some part of the body sits nearer the camera along the
    same ray, so this compares its depth against the nearest surface in its pixel.

    A z-buffer built from vertices rather than rasterised triangles, which is coarser and
    honest about being so: it can call a keypoint visible through a gap between vertices on a
    sparse mesh. On 19,158 vertices the gaps are small, and the alternative is a full
    rasteriser this does not need yet.
    """
    zbuf = np.full((height, width), np.inf)
    xs = np.clip(verts_px[:, 0].astype(int), 0, width - 1)
    ys = np.clip(verts_px[:, 1].astype(int), 0, height - 1)
    inside = (
        (verts_px[:, 0] >= 0) & (verts_px[:, 0] < width)
        & (verts_px[:, 1] >= 0) & (verts_px[:, 1] < height)
        & (verts_depth > 0)
    )
    np.minimum.at(zbuf, (ys[inside], xs[inside]), verts_depth[inside])

    out = np.full(len(kp_px), VIS_OUT_OF_FRAME, dtype=np.int8)
    for i, ((x, y), d) in enumerate(zip(kp_px, kp_depth)):
        if not (0 <= x < width and 0 <= y < height and d > 0):
            continue
        nearest = zbuf[int(y), int(x)]
        out[i] = VIS_VISIBLE if d <= nearest + tol else VIS_OCCLUDED
    return out


def stick_figure(kp_px, vis, names, width, height, edges):
    """The check RFD 0122 asks for before any of this is scaled up.

    Not decoration. The pose sources are all locomotion, and if a character artist would not
    draw these poses the corpus is wrong however correct the labels are. Twenty of these
    answer that, and they cost no GPU.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (250, 249, 245))
    d = ImageDraw.Draw(img)
    idx = {n: i for i, n in enumerate(names)}
    for a, b in edges:
        if a in idx and b in idx and vis[idx[a]] and vis[idx[b]]:
            d.line([tuple(kp_px[idx[a]]), tuple(kp_px[idx[b]])], fill=(20, 20, 24), width=3)
    for i, (x, y) in enumerate(kp_px):
        if vis[i] == VIS_OUT_OF_FRAME:
            continue
        colour = (14, 107, 121) if vis[i] == VIS_VISIBLE else (191, 67, 24)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)
    return img


# COCO-17 plus the feet, as the asset stores them.
EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_big_toe"),
    ("right_ankle", "right_heel"), ("right_heel", "right_big_toe"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="render_out")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--stick", action="store_true", help="write stick figures and stop")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, posed = load_model()
    verts = posed["vertices"][0].detach().cpu().numpy().astype(np.float64)
    faces = model.faces.cpu().numpy().astype(np.int64)
    names, W = load_keypoint_weights(verts.shape[0])

    # The label. A weighted sum of posed vertices, not a prediction.
    kp_world = W @ verts

    centre = verts.mean(axis=0)
    radius = float(np.linalg.norm(verts - centre, axis=1).max()) * 3.0
    up = np.array([0.0, 0.0, 1.0])           # ANNY is Z-up

    rows = []
    for v in range(args.views):
        a = 2.0 * np.pi * v / args.views
        eye = centre + np.array([np.cos(a), np.sin(a), 0.25]) * radius
        view = look_at(eye, centre, up)

        kp_px, kp_depth = project(kp_world, view, args.fov, args.width, args.height)
        v_px, v_depth = project(verts, view, args.fov, args.width, args.height)

        # Depth first. It is the conditioning signal, and the visibility test is a read of
        # the same buffer rather than a second approximation of it.
        zbuf, mask = rasterise_depth(v_px, v_depth, faces, args.width, args.height)
        img16, (near, far) = depth_png(zbuf, mask)
        if img16 is not None:
            img16.save(out_dir / f"depth_{v:02d}.png")

        vis = visibility_from_zbuf(kp_px, kp_depth, zbuf, args.width, args.height)

        counts = {int(s): int((vis == s).sum()) for s in (0, 1, 2)}
        rows.append({"view": v, "visibility_counts": counts,
                     "depth_near": near, "depth_far": far,
                     "silhouette_px": int(mask.sum())})
        print(f"  view {v}  visible {counts.get(2,0):2d}  occluded {counts.get(1,0):2d}  "
              f"out of frame {counts.get(0,0):2d}")

        if args.stick:
            img = stick_figure(kp_px, vis, names, args.width, args.height, EDGES)
            img.save(out_dir / f"stick_{v:02d}.png")

        # The mesh and the camera, handed to Blender exactly as projected here. Written
        # rather than recomputed there, so the render and the labels cannot drift apart.
        np.savez(out_dir / f"mesh_{v:02d}.npz", verts=verts, faces=faces)
        (out_dir / f"cam_{v:02d}.json").write_text(json.dumps({
            "view": view.tolist(), "fov_deg": args.fov,
            "width": args.width, "height": args.height,
        }))
        np.savez(out_dir / f"kp_{v:02d}.npz", px=kp_px, vis=vis)

    (out_dir / "summary.json").write_text(
        json.dumps({"keypoints": len(names), "names": names, "views": rows}, indent=2)
    )
    print(f"\n{len(names)} keypoints, {args.views} views -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
