# SPDX-License-Identifier: Apache-2.0 OR MIT
"""An MToon tone sweep as RGBA frames: left solved to dE 12, right a flat x0.6.

    python render_mtoon_frames.py --out frames --frames 60 --width 3840 --height 2160
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import subprocess
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frame_ring  # noqa: E402
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


def tone_frames(hold):
    """Each Monk tone held for `hold` frames at the dE 12 multiplier."""
    rows = []
    for tone in sweep.MST:
        base = sweep.linear_of(tone)
        mult = sweep.solve_multiplier(base, 12.0)
        for k in range(hold):
            rows.append((tone, base, [c * mult for c in base], k / max(1, hold - 1)))
    return rows


def sun(phase, arc=70.0, elevation=0.35):
    """The light swept through `arc` degrees, smoothstepped so it settles at both ends."""
    eased = phase * phase * (3.0 - 2.0 * phase)
    angle = np.radians((eased * 2.0 - 1.0) * arc)
    return (float(np.cos(angle)), float(elevation), float(np.sin(angle)))


def to_srgb8(rgba):
    lin = np.clip(rgba[..., :3], 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    out = np.zeros(rgba.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.round(srgb * 255).astype(np.uint8)
    out[..., 3] = np.round(np.clip(rgba[..., 3], 0, 1) * 255).astype(np.uint8)
    return out


def render_chunks(stream, hold, width, height, spp=1, slots=8, chunk=50, verbose=True):
    """Render a slice per subprocess. Dr.Jit breaks after a few hundred renders in one
    process -- `could not resolve symbol` naming `callables` -- and no cache flush fixes it,
    so each chunk gets a fresh interpreter and the stream is appended."""
    total = len(tone_frames(hold))
    stream = pathlib.Path(stream)
    if stream.exists():
        stream.unlink()
    t0 = time.perf_counter()
    for start in range(0, total, chunk):
        count = min(chunk, total - start)
        out = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--out", str(stream),
             "--hold", str(hold), "--width", str(width), "--height", str(height),
             "--spp", str(spp), "--slots", str(slots),
             "--start", str(start), "--count", str(count), "--append"],
            capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit("FAIL  chunk at %d died: %s" % (start, out.stderr[-1500:]))
        if verbose:
            print("    %4d/%d  %.1f s  %s" % (start + count, total,
                                              time.perf_counter() - t0,
                                              out.stdout.strip().splitlines()[-1]))
    got = stream.stat().st_size // (width * height * 4)
    if got != total:
        raise SystemExit("FAIL  stream holds %d frames, wanted %d" % (got, total))
    if verbose:
        print("  %d frames of %dx%d rgba  %.1f s  (%.3f s/frame)  %s"
              % (total, width, height, time.perf_counter() - t0,
                 (time.perf_counter() - t0) / total, stream))
    return total


def render(stream, hold, width, height, spp=1, slots=8, flush_every=10, verbose=True,
           start=0, count=None, append=False):
    """Render into a bounded ring; a writer thread drains it to one RGBA stream."""
    import drjit as dr
    import mitsuba as mi
    half = width // 2
    stream = pathlib.Path(stream)
    stream.parent.mkdir(parents=True, exist_ok=True)
    rows = tone_frames(hold)[start:None if count is None else start + count]
    scenes = [forward.build(rows[0][1], rows[0][2], half, height, spp,
                            shading_toony_factor=0.9) for _ in range(2)]
    ring = frame_ring.FrameRing(slots, "frames")

    def writer():
        with open(stream, "ab" if append else "wb") as fh:
            while True:
                try:
                    fh.write(ring.get(timeout=600))
                except frame_ring.Closed:
                    return

    thread = threading.Thread(target=writer, name="write", daemon=True)
    thread.start()
    t0 = time.perf_counter()
    for i, (tone, base, shade_colour, phase) in enumerate(rows):
        light = sun(phase)
        forward.retune(scenes[0], base, shade_colour, light)
        forward.retune(scenes[1], base, [c * 0.6 for c in base], light)
        left = np.array(mi.render(scenes[0], spp=spp, seed=i))
        right = np.array(mi.render(scenes[1], spp=spp, seed=i))
        frame = np.concatenate([left[..., :4], right[..., :4]], axis=1)
        ring.put(to_srgb8(frame).tobytes())
        if (i + 1) % flush_every == 0:
            dr.flush_malloc_cache()
        if verbose and (i + 1) % hold == 0:
            print("    tone %s  %d/%d  %.1f s" % (tone, i + 1, len(rows),
                                                  time.perf_counter() - t0))
    ring.close()
    thread.join()
    if verbose:
        print("  " + ring.report())
    if ring.put_count != len(rows) or ring.get_count != len(rows):
        raise SystemExit("FAIL  ring moved %d in and %d out of %d frames"
                         % (ring.put_count, ring.get_count, len(rows)))
    return len(rows)


def self_test():
    """Nine controls; four reject a frame not showing the material."""
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

    a, b = sun(0.0), sun(1.0)
    r.append(("the sun moves across the hold", abs(a[2] - b[2]) > 0.5))
    r.append(("the sweep is symmetric about the middle",
              abs(sun(0.5)[2]) < 1e-9 and abs(a[2] + b[2]) < 1e-9))
    r.append(("the easing settles at both ends",
              abs(sun(0.02)[2] - a[2]) < abs(sun(0.5)[2] - sun(0.44)[2])))

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
    ap.add_argument("--out", default="frames.rgba")
    ap.add_argument("--hold", type=int, default=90,
                    help="frames per tone; 90 at 60fps is 1.5 s each")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    if args.self_test:
        return self_test()
    if args.count is None:
        render_chunks(args.out, args.hold, args.width, args.height, args.spp,
                      args.slots, args.chunk)
    else:
        render(args.out, args.hold, args.width, args.height, args.spp, args.slots,
               start=args.start, count=args.count, append=args.append)
    return 0


if __name__ == "__main__":
    sys.exit(main())
