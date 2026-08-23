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

import numpy as np
from PIL import Image

SIZE = 256
DELTA = 25.0          # distance from background, in 0..255 RGB
CONTROL_SHIFT = 20    # px


def body_mask(path, size=SIZE):
    a = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)).astype(np.float32)
    edge = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3),
                           a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    bg = np.median(edge, 0)
    return np.linalg.norm(a - bg, axis=2) > DELTA, bg


def reference(exr, size=SIZE):
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    bmp = mi.Bitmap(exr)
    names = [c.name for c in bmp.struct_()]
    alpha = (np.array(bmp)[..., names.index("A")] > 0.5).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(alpha).resize((size, size), Image.NEAREST)) > 127


def iou(a, b):
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exr", required=True, help="the render whose alpha is ground truth")
    ap.add_argument("--dir", required=True, help="directory of restyled png to score")
    ap.add_argument("--floor", type=float, default=0.80)
    args = ap.parse_args()

    ref = reference(args.exr)
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
