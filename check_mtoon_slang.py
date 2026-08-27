# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gate: the Slang-compiled shading agrees with mtoon.py, which agrees with three-vrm.

Build:
    slangc mtoon.slang -target cpp -entry shadeMain -o mtoon_slang_gen.cpp
    clang++ -O2 -std=c++17 -shared -o mtoon_shade.dll mtoon_shade.cpp

    python check_mtoon_slang.py --self-test [--bench]
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mtoon  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
LIB = HERE / ("mtoon_shade.dll" if sys.platform == "win32" else "mtoon_shade.so")

_lib = None


def library():
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise SystemExit("FAIL  %s is not built; the docstring has the two commands" % LIB)
        _lib = ctypes.CDLL(str(LIB))
        _lib.mtoon_shade.restype = None
        _lib.mtoon_shade.argtypes = [ctypes.c_uint32] + [
            np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")] * 4
        _lib.mtoon_params_size.restype = ctypes.c_uint32
    return _lib


def pack(base, shade, light_color=(1.0, 1.0, 1.0), **kw):
    """Four float3 then six floats. The stride is read from the library, not assumed."""
    size = library().mtoon_params_size()
    stride = (size - 6 * 4) // 4 // 4
    if stride not in (3, 4):
        raise SystemExit("FAIL  unexpected float3 stride %d from a %d byte struct"
                         % (stride, size))
    p = dict(mtoon.DEFAULTS, **kw)
    buf = np.zeros(size // 4, dtype=np.float32)
    for i, vec in enumerate((base, shade, p["parametric_rim_color_factor"], light_color)):
        buf[i * stride:i * stride + 3] = vec
    tail = 4 * stride
    for j, k in enumerate(("shading_shift_factor", "shading_toony_factor",
                           "shading_shift_texture", "parametric_rim_lift_factor",
                           "parametric_rim_fresnel_power_factor", "rim_lighting_mix_factor")):
        buf[tail + j] = p[k]
    return buf


def shade(dot_nl, dot_nv, base, shade_color, light_color=(1.0, 1.0, 1.0), **kw):
    nl = np.ascontiguousarray(dot_nl, dtype=np.float32)
    nv = np.ascontiguousarray(dot_nv, dtype=np.float32)
    out = np.zeros(nl.size * 3, dtype=np.float32)
    library().mtoon_shade(nl.size, nl, nv, pack(base, shade_color, light_color, **kw), out)
    return out.reshape(-1, 3)


def reference(dot_nl, dot_nv, base, shade_color, light_color=(1.0, 1.0, 1.0), **kw):
    return np.array([mtoon.evaluate(float(a), float(b), base, shade_color, light_color, **kw)[0]
                     for a, b in zip(np.ravel(dot_nl), np.ravel(dot_nv))])


def bench(n=1 << 22):
    rng = np.random.default_rng(20260826)
    nl = rng.uniform(-1, 1, n).astype(np.float32)
    nv = rng.uniform(0, 1, n).astype(np.float32)
    base, sh = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)

    t0 = time.perf_counter()
    shade(nl, nv, base, sh)
    slang = time.perf_counter() - t0

    m = 20000
    t0 = time.perf_counter()
    reference(nl[:m], nv[:m], base, sh)
    py = (time.perf_counter() - t0) / m * n

    print("  slang   %8.3f s for %d shading calls  (%.1f ns each)"
          % (slang, n, slang / n * 1e9))
    print("  python  %8.3f s extrapolated from %d  (%.1f ns each)" % (py, m, py / n * 1e9))
    print("  speedup %8.0fx" % (py / slang))
    return slang, py


def self_test():
    """Eight controls. Four must reject a kernel that has drifted from the model."""
    r = []
    base, sh = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)
    nl = np.linspace(-1, 1, 401)
    nv = np.linspace(0, 1, 401)

    size = library().mtoon_params_size()
    r.append(("the params struct is the size a float3 layout gives",
              size in (4 * 3 * 4 + 24, 4 * 4 * 4 + 24)))

    for toony, shift in ((0.9, 0.0), (0.5, 0.3), (0.0, -0.3), (1.0, 0.0)):
        got = shade(nl, nv, base, sh, shading_toony_factor=toony, shading_shift_factor=shift)
        want = reference(nl, nv, base, sh, shading_toony_factor=toony,
                         shading_shift_factor=shift)
        worst = float(np.abs(got - want).max())
        r.append(("toony %.1f shift %+.1f agrees to float32 (%.2e)" % (toony, shift, worst),
                  worst <= 4 * float(np.finfo(np.float32).eps)))

    rim = shade(nl, nv, base, sh, parametric_rim_color_factor=(1, 1, 1),
                parametric_rim_fresnel_power_factor=3.0, rim_lighting_mix_factor=0.5)
    rim_want = reference(nl, nv, base, sh, parametric_rim_color_factor=(1, 1, 1),
                         parametric_rim_fresnel_power_factor=3.0, rim_lighting_mix_factor=0.5)
    r.append(("the rim term agrees too", np.abs(rim - rim_want).max() <= 1e-6))

    wrong = reference(nl, nv, base, sh, shading_toony_factor=0.1)
    r.append(("a different parameter gives a different answer, so agreement is not vacuous",
              np.abs(shade(nl, nv, base, sh, shading_toony_factor=0.9) - wrong).max() > 0.01))

    tail = shade(np.zeros(65, np.float32), np.zeros(65, np.float32), base, sh)
    r.append(("a size past one thread group is fully written", not np.allclose(tail[64], 0)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.bench:
        bench()
        return 0
    if args.self_test:
        return self_test()
    ap.error("pass --self-test or --bench")


if __name__ == "__main__":
    sys.exit(main())
