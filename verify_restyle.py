"""Did the restyle move the body? Silhouette agreement against the render's own alpha.

WHAT THIS DOES NOT MEASURE, said first because the number looks like more than it is. Rule 4
wants joint drift, and joint drift needs a keypoint detector -- pose-consensus's referee. This
checks only that the figure is still where it was. A restyle can pass here and still have
bent an arm inside the silhouette.

RETRACTED: a fixed brightness threshold. The first version of this check thresholded at a
constant and scored the CycleGAN ukiyo-e restyle at IoU 0.098, which reads as a destroyed
pose. The pose was untouched; ukiyo-e lifts the background from black to warm grey, so the
threshold called the whole canvas foreground. It measured `is this pixel bright` rather than
`is this the body` -- the convenient proxy, exactly where the RFD says not to reach for one.
Background is now estimated per image from the border, where the render has no body.

The control is not optional. A check that cannot fail cannot pass: the reference is shifted
and must score badly, or this says nothing about a body that moved.
"""
import argparse
import glob
import os
import pathlib

import numpy as np
from PIL import Image

SIZE = 256
DELTA = 25.0          # distance from background, in 0..255 RGB
CONTROL_SHIFT = 20    # px


def _fill_from_border(free):
    """Flood the background inward from the border; whatever is unreached is enclosed.

    THE THIRD WAY THIS CHECK HAS BEEN WRONG. A fixed threshold failed on ukiyo-e, which lifts
    the background; a background-relative threshold then failed on line art, where the figure's
    INTERIOR is the same white as the paper and only the strokes register. Scoring the outline
    of a drawing against a filled silhouette gave 0.393 for a sketch whose pose was correct.
    Background is what connects to the border, so the fill is the definition rather than a
    patch: enclosed white is body, surrounding white is paper.
    """
    h, w = free.shape
    seen = np.zeros_like(free)
    stack = [(0, x) for x in range(w) if free[0, x]] + [(h - 1, x) for x in range(w) if free[h - 1, x]]
    stack += [(y, 0) for y in range(h) if free[y, 0]] + [(y, w - 1) for y in range(h) if free[y, w - 1]]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return ~seen


def body_mask(path, size=SIZE):
    a = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)).astype(np.float32)
    edge = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3),
                           a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    bg = np.median(edge, 0)
    near_bg = np.linalg.norm(a - bg, axis=2) <= DELTA
    return _fill_from_border(near_bg), bg


def reference(lottie, view, size=SIZE):
    """Ground truth from the render's own depth, read out of the Lottie that carries it.

    This used to open an OpenEXR and take its `A` channel, which needed Mitsuba imported purely
    to decode a file. The renderer writes one Lottie now, with each view's depth embedded as a
    data-URI PNG whose RGBA8 holds the float32 bit pattern, so the dependency goes away and the
    numbers are the same ones.

    THE MATTE IS DERIVED RATHER THAN STORED. Depth is metres from the camera and the renderer
    writes 0 where no ray hit, so `depth > 0` is the silhouette exactly -- no threshold to pick
    and nothing to disagree with. That is why the Lottie carries no matte asset.
    """
    import base64
    import io
    import json

    doc = json.loads(pathlib.Path(lottie).read_text(encoding="utf-8"))
    tags = doc["meta"]["views"]
    if view not in tags:
        raise SystemExit(f"view {view!r} is not in {lottie}: has {tags}")
    uri = next(a for a in doc["assets"] if a["id"] == f"depth_{tags.index(view)}")["p"]
    px = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
                    .convert("RGBA")).astype(np.uint32)
    depth = (px[..., 0] | (px[..., 1] << 8) | (px[..., 2] << 16)
             | (px[..., 3] << 24)).astype(np.uint32).view(np.float32)
    alpha = ((depth > 0).astype(np.uint8) * 255)
    return np.asarray(Image.fromarray(alpha).resize((size, size), Image.NEAREST)) > 127


def iou(a, b):
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lottie", required=True,
                    help="the renderer's multiview Lottie, which carries the depth")
    ap.add_argument("--view", default="three-quarter",
                    help="which embedded viewpoint is ground truth")
    ap.add_argument("--dir", required=True, help="directory of restyled png to score")
    ap.add_argument("--floor", type=float, default=0.80)
    args = ap.parse_args()

    ref = reference(args.lottie, args.view)
    shifted = np.zeros_like(ref)
    shifted[:, CONTROL_SHIFT:] = ref[:, :-CONTROL_SHIFT]
    control = iou(shifted, ref)

    print(f"{'panel':22} {'IoU':>7}   background")
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.png"))):
        if "contact_sheet" in f:
            continue
        m, bg = body_mask(f)
        v = iou(m, ref)
        rows.append((os.path.basename(f), v))
        print(f"{os.path.basename(f):22} {v:7.3f}   ({bg[0]:.0f},{bg[1]:.0f},{bg[2]:.0f})")

    print(f"\ncontrol: reference shifted {CONTROL_SHIFT} px -> {control:.3f}")
    if control >= args.floor:
        print("  FAIL the control scores as well as a real match, so this check is decoration")
        return 1
    print(f"  ok   the control fails, so a passing score means the body did not move")

    bad = [n for n, v in rows if v < args.floor]
    if bad:
        print(f"\n{len(bad)} below the {args.floor} floor: {', '.join(bad)}")
        return 1
    print(f"\nall {len(rows)} restyles keep the silhouette above {args.floor}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
