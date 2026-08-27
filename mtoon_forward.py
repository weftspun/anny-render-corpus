# SPDX-License-Identifier: Apache-2.0 OR MIT
"""MToon 1.0 as a wide forward Mitsuba integrator, in Dr.Jit ops so llvm_ad_rgb widens it.

    python mtoon_forward.py --self-test [--bench]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mtoon  # noqa: E402

SHAPE = pathlib.Path(__file__).resolve().parent / "mtoon-reference" / "testshape.obj"
_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    import drjit as dr
    import mitsuba as mi

    class _Forward(mi.SamplingIntegrator):
        def __init__(self, props):
            mi.SamplingIntegrator.__init__(self, props)
            g = lambda k, d=0.0: float(props.get(k, d))  # noqa: E731
            v = np.array([g("light_x", 1.0), g("light_y", 0.2), g("light_z", 0.6)])
            self.light = tuple(float(x) for x in v / np.linalg.norm(v))
            self.base = [g("base_" + c) for c in "rgb"]
            self.shade = [g("shade_" + c) for c in "rgb"]
            self.rim = [g("rim_" + c) for c in "rgb"]
            self.shadows = g("shadows", 1.0) > 0.5
            self.p = {k: g(k, mtoon.DEFAULTS[k]) for k in mtoon.PROP_KEYS}

        def sample(self, scene, sampler, ray, medium=None, active=True):
            si = scene.ray_intersect(ray, active)
            hit = si.is_valid() & active
            n = si.sh_frame.n
            to_light = mi.Vector3f(*self.light)
            dot_nl = dr.dot(n, to_light)
            dot_nv = dr.dot(n, -ray.d)

            if self.shadows:
                lit = hit & (dot_nl > 0.0)
                occluded = scene.ray_test(si.spawn_ray(to_light), lit)
                dot_nl = dr.select(occluded, mi.Float(-1.0), dot_nl)

            toony = min(max(self.p["shading_toony_factor"], 0.0), 1.0)
            lo, hi = -1.0 + toony, 1.0 - toony
            s = dot_nl + self.p["shading_shift_factor"] + self.p["shading_shift_texture"]
            if abs(hi - lo) <= 1e-6:
                t = dr.select(s >= lo, mi.Float(1.0), mi.Float(0.0))
            else:
                t = dr.clip((s - lo) / (hi - lo), 0.0, 1.0)

            base = mi.Color3f(*self.base)
            shade = mi.Color3f(*self.shade)
            col = shade + (base - shade) * t

            r = dr.clip(1.0 - dot_nv + self.p["parametric_rim_lift_factor"], 0.0, 1.0)
            r = dr.power(r, max(self.p["parametric_rim_fresnel_power_factor"], 1e-6))
            col = col + mi.Color3f(*self.rim) * r
            return dr.select(hit, col, mi.Color3f(0.0)), hit, []

        def to_string(self):
            return "MToonForward[]"

    mi.register_integrator("mtoon_forward", lambda props: _Forward(props))
    _REGISTERED = True


def integrator_dict(base, shade, light=(1.0, 0.2, 0.6), shadows=True, **kw):
    p = dict(mtoon.DEFAULTS, **kw)
    d = {"type": "mtoon_forward", "shadows": 1.0 if shadows else 0.0}
    for i, name in enumerate("rgb"):
        d["base_" + name] = float(base[i])
        d["shade_" + name] = float(shade[i])
        d["rim_" + name] = float(p["parametric_rim_color_factor"][i])
    for i, name in enumerate("xyz"):
        d["light_" + name] = float(light[i])
    for k in mtoon.PROP_KEYS:
        d[k] = float(p[k])
    return d


def build(base, shade, width=256, height=256, spp=1, shape=SHAPE, **kw):
    import mitsuba as mi
    register()
    return mi.load_dict({
        "type": "scene",
        "integrator": integrator_dict(base, shade, **kw),
        "sensor": {
            "type": "perspective", "fov": 40,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 3, 0], target=[0, 0, 0], up=[0, 0, 1]),
            "film": {"type": "hdrfilm", "width": width, "height": height,
                     "rfilter": {"type": "box"}, "pixel_format": "rgba"},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "obj": {"type": "obj", "filename": str(shape)},
    })


def retune(scene, base, shade, light=None, **kw):
    """Change the material in place; the integrator is a Python object."""
    it = scene.integrator()
    it.base = [float(c) for c in base]
    it.shade = [float(c) for c in shade]
    if light is not None:
        v = np.array(light, dtype=float)
        it.light = tuple(float(x) for x in v / np.linalg.norm(v))
    for k, v in kw.items():
        if k in it.p:
            it.p[k] = float(v)
    return scene


def render(base, shade, width=256, height=256, spp=1, shape=SHAPE, **kw):
    import mitsuba as mi
    scene = build(base, shade, width, height, spp, shape, **kw)
    return np.array(mi.render(scene, spp=spp))


def self_test():
    """Eight controls; five reject a render that lost the model or the shadows."""
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    r = []
    base, shade = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)

    img = render(base, shade, 192, 192, shading_toony_factor=1.0)
    rgb, alpha = img[..., :3], img[..., 3]
    hit = alpha > 0.5
    r.append(("the shape renders", 0.05 < hit.mean() < 0.95))
    r.append(("alpha is coverage", np.all(rgb[~hit] == 0)))

    body = rgb[hit]
    r.append(("the lit plateau is the base colour",
              mtoon.delta_e(list(body.max(axis=0)), list(base)) < 1.5))
    r.append(("the shade plateau reaches film",
              mtoon.delta_e(list(body.min(axis=0)), list(shade)) < 1.5))

    lit_only = render(base, shade, 192, 192, shading_toony_factor=1.0, shadows=False)
    shaded = (rgb[hit] @ np.ones(3)) < (body.max() * 0.9)
    lit_frac = ((lit_only[..., :3][hit] @ np.ones(3)) < (body.max() * 0.9))
    r.append(("shadows change the picture, so the ray is doing something",
              not np.allclose(img, lit_only)))
    r.append(("shadows put MORE of the body in shade",
              shaded.mean() > lit_frac.mean()))

    flat = render(base, base, 192, 192, shading_toony_factor=1.0)
    fb = flat[..., :3][flat[..., 3] > 0.5]
    r.append(("base == shade renders flat",
              mtoon.delta_e(list(fb.max(axis=0)), list(fb.min(axis=0))) < 1.0))

    other = render((0.2, 0.2, 0.9), (0.05, 0.05, 0.3), 192, 192)
    r.append(("a different material renders differently", not np.allclose(img, other)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def bench(width=3840, height=2160):
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    base, shade = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)
    render(base, shade, 256, 256)
    t0 = time.perf_counter()
    render(base, shade, width, height)
    dt = time.perf_counter() - t0
    px = width * height
    print("  forward %dx%d with shadows: %.3f s  (%.1f ns/px)" % (width, height, dt,
                                                                  dt / px * 1e9))
    print("  60 frames: %.1f s" % (60 * dt))
    return dt


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
