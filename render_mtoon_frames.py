# SPDX-License-Identifier: Apache-2.0 OR MIT
"""An MToon tone sweep as RGBA frames: left solved to dE 12, right a flat x0.6.

    python render_mtoon_frames.py --out frames --frames 60 --width 3840 --height 2160
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mtoon  # noqa: E402
import mtoon_forward as forward  # noqa: E402
import mtoon_sweep as sweep  # noqa: E402

SHAPE = pathlib.Path(__file__).resolve().parent / "mtoon-reference" / "testshape.obj"
LIGHT = np.array([1.0, 0.2, 0.6])


def frame_for(tone_base, shade_colour, width, height, spp=1, **kw):
    img = forward.render(tone_base, shade_colour, width=width, height=height, spp=spp, **kw)
    out = np.zeros(img.shape[:2] + (4,), dtype=np.float32)
    out[..., :3] = img[..., :3]
    out[..., 3] = img[..., 3]
    return out


def tone_frames(count):
    """The Monk ladder at the multiplier holding contrast at dE 12."""
    rows = []
    for i in range(count):
        tone = sweep.MST[int(i * len(sweep.MST) / count)]
        base = sweep.linear_of(tone)
        mult = sweep.solve_multiplier(base, 12.0)
        rows.append((tone, base, [c * mult for c in base]))
    return rows


def to_srgb8(rgba):
    lin = np.clip(rgba[..., :3], 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    out = np.zeros(rgba.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.round(srgb * 255).astype(np.uint8)
    out[..., 3] = np.round(np.clip(rgba[..., 3], 0, 1) * 255).astype(np.uint8)
    return out


def render(out_dir, frames, width, height, spp=1, verbose=True):
    half = width // 2
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    written = []
    for i, (tone, base, shade_colour) in enumerate(tone_frames(frames)):
        left = frame_for(base, shade_colour, half, height, spp,
                         shading_toony_factor=0.9)
        right = frame_for(base, [c * 0.6 for c in base], half, height, spp,
                          shading_toony_factor=0.9)
        frame = np.concatenate([left, right], axis=1)
        path = out_dir / ("frame_%04d.raw" % i)
        to_srgb8(frame).tofile(path)
        written.append(path)
    if verbose:
        total = time.perf_counter() - t0
        print("  %d frames of %dx%d rgba  %.3f s  (%.3f s/frame)"
              % (len(written), width, height, total, total / max(1, len(written))))
    return written


def self_test():
    """Six controls; four reject a frame not showing the material."""
    r = []
    base = sweep.linear_of("825c43")
    dark = [c * sweep.solve_multiplier(base, 12.0) for c in base]
    frame = frame_for(base, dark, 128, 128, shading_toony_factor=1.0)
    hit = frame[..., 3] > 0.5
    r.append(("the render hits the shape", 0.05 < hit.mean() < 0.95))
    r.append(("alpha is coverage", np.array_equal(frame[..., 3] > 0.5, hit)))
    r.append(("nothing is shaded outside the mask", np.all(frame[~hit, :3] == 0)))

    inside = frame[hit][:, :3]
    r.append(("both plateaus reach the frame",
              mtoon.delta_e(list(inside.max(axis=0)), list(inside.min(axis=0))) > 5.0))

    flat = frame_for(base, [c * 0.6 for c in base], 128, 128, shading_toony_factor=1.0)
    r.append(("the flat multiplier renders a different frame",
              not np.allclose(frame, flat)))

    eight = to_srgb8(frame)
    r.append(("the 8-bit conversion keeps the mask exactly",
              np.array_equal(eight[..., 3] > 127, hit)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="frames")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    if args.self_test:
        return self_test()
    render(args.out, args.frames, args.width, args.height, args.spp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
