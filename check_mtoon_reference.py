# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gate: our MToon 1.0 agrees with three-vrm, pixel for pixel, up to a factor of 1/pi.

Needs node, three, @pixiv/three-vrm and playwright. mtoon-reference/README.md installs them
and records what this measured.

    python check_mtoon_reference.py --self-test
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mtoon  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent / "mtoon-reference"
QUANT = 1.0 / 255.0
BASE = (0.72, 0.52, 0.32)
SHADE = (0.30, 0.21, 0.13)


def render(**q):
    args = "&".join("%s=%s" % (k, v) for k, v in q.items())
    out = subprocess.run(["node", "run.js", args], capture_output=True, text=True,
                         cwd=HERE)
    if out.returncode != 0:
        raise SystemExit("three-vrm render failed:\n" + out.stderr[:2000])
    return json.loads(out.stdout)


def ours(res, light, **kw):
    """Our model at the same normals the render used."""
    idx = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    u, v = np.meshgrid(idx, idx)              # readPixels is bottom-up, v ascends with row
    rr = u * u + v * v
    inside = rr <= 1.0
    n = np.stack([u, v, np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))], axis=-1)
    L = np.asarray(light, float)
    L = L / np.linalg.norm(L)
    dot_nl = n @ L
    dot_nv = n[..., 2]
    out = np.zeros((res, res, 3))
    for j in range(res):
        for i in range(res):
            if not inside[j, i]:
                continue
            col, _ = mtoon.evaluate(float(dot_nl[j, i]), float(dot_nv[j, i]),
                                    BASE, SHADE, **kw)
            out[j, i] = col
    return out, inside


def compare(res=64, toony=0.9, shift=0.0, verbose=True):
    d = render(res=res, toony=toony, shift=shift,
               base_r=BASE[0], base_g=BASE[1], base_b=BASE[2],
               shade_r=SHADE[0], shade_g=SHADE[1], shade_b=SHADE[2])
    px = np.array(d["pixels"], dtype=np.float64).reshape(res, res, 4)[..., :3] / 255.0
    mine, inside = ours(res, d["light"],
                        shading_toony_factor=toony, shading_shift_factor=shift)
    edge = np.zeros_like(inside)
    edge[1:-1, 1:-1] = inside[1:-1, 1:-1] & inside[:-2, 1:-1] & inside[2:, 1:-1] \
        & inside[1:-1, :-2] & inside[1:-1, 2:]
    a, b = px[edge], mine[edge]
    if verbose:
        print("  three r%s, %d px compared (%d in the disc)" % (d["three"], edge.sum(),
                                                                inside.sum()))
        print("  theirs max %s   ours max %s" % (np.round(a.max(axis=0), 4),
                                                 np.round(b.max(axis=0), 4)))
        print("  theirs min %s   ours min %s" % (np.round(a.min(axis=0), 4),
                                                 np.round(b.min(axis=0), 4)))
        nz = b > 1e-6
        if nz.any():
            ratio = a[nz] / b[nz]
            print("  ratio theirs/ours  median %.4f  p05 %.4f  p95 %.4f"
                  % (np.median(ratio), np.percentile(ratio, 5), np.percentile(ratio, 95)))
        print("  max |diff| %.4f   mean |diff| %.4f   (1/255 = %.4f)"
              % (np.abs(a - b).max(), np.abs(a - b).mean(), 1 / 255))
    return a, b


def self_test():
    """Five controls; three reject a port drifted from the reference."""
    import math
    r, rows = [], []
    for toony in (0.9, 0.5, 0.0, 1.0):
        a, b = compare(toony=toony, verbose=False)
        scaled = np.abs(a - b / math.pi).max(axis=1)
        rows.append((toony, len(scaled), int((scaled > 4 * QUANT).sum()),
                     float(np.percentile(scaled, 99)), float(scaled.max())))
        print("  toony %.1f  px %d  over 4/255 %d  p99 %.4f  max %.4f" % rows[-1])

    soft = [x for x in rows if x[0] != 1.0]
    r.append(("a soft ramp agrees below the readback quantisation",
              all(n == 0 for _, _, n, _, _ in soft)))
    r.append(("a hard ramp disagrees only on the terminator",
              all(p99 < 4 * QUANT for t, _, _, p99, _ in rows if t == 1.0)
              and all(n <= 4 for t, _, n, _, _ in rows if t == 1.0)))
    a, b = compare(toony=0.9, verbose=False)
    nz = b > 1e-6
    scale = float(np.median(a[nz] / b[nz]))
    r.append(("the scale between the two is 1/pi", abs(scale - 1 / math.pi) < 0.005))
    r.append(("unscaled, the two do NOT agree, so the factor is real",
              np.abs(a - b).max() > 40 * QUANT))
    r.append(("the reference actually rendered a body", len(a) > 1000))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    sys.exit(self_test() if a.self_test else compare() and 0)
