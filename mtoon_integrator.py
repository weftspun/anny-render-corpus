# SPDX-License-Identifier: Apache-2.0 OR MIT
"""A Mitsuba integrator for MToon 1.0, because MToon is not a BSDF.

MToon paints its shade colour where dot(N, L) < 0, and a physically based integrator
contributes nothing there, so a BSDF renders lit-to-black and the shade plateau never
appears. This shades every hit against the key light directly, so both plateaus reach film.

    python mtoon_integrator.py --self-test
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import mtoon

_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    import mitsuba as mi

    class _MToonIntegrator(mi.SamplingIntegrator):
        def __init__(self, props):
            mi.SamplingIntegrator.__init__(self, props)
            g = lambda k, d=0.0: float(props.get(k, d))  # noqa: E731
            self.light = np.array([g("light_x", -1.0), g("light_y", -0.2), g("light_z", -0.6)])
            self.light /= np.linalg.norm(self.light)
            self.light_color = np.array([g("light_r", 1.0), g("light_g", 1.0),
                                         g("light_b", 1.0)])
            self.base = np.array([g("base_" + c) for c in "rgb"])
            self.shade = np.array([g("shade_" + c) for c in "rgb"])
            self.shadows = bool(g("shadows", 1.0))
            self.kw = dict(mtoon.DEFAULTS)
            self.kw["parametric_rim_color_factor"] = tuple(g("rim_" + c) for c in "rgb")
            for k in mtoon.PROP_KEYS:
                self.kw[k] = g(k, mtoon.DEFAULTS[k])

        def sample(self, scene, sampler, ray, medium=None, active=True):
            si = scene.ray_intersect(ray)
            valid = si.is_valid()
            if not valid:
                return mi.Spectrum(0.0), valid, []
            n = np.array(si.sh_frame.n).reshape(-1)[:3]
            to_light = -self.light
            dot_nl = float(np.dot(n, to_light))
            dot_nv = float(np.dot(n, -np.array(ray.d).reshape(-1)[:3]))
            if self.shadows and dot_nl > 0.0:
                shadow = si.spawn_ray(mi.Vector3f(*to_light))
                if scene.ray_test(shadow):
                    dot_nl = -1.0
            col, _ = mtoon.evaluate(dot_nl, dot_nv, self.base, self.shade,
                                    self.light_color, **self.kw)
            return mi.Spectrum(list(col)), valid, []

        def to_string(self):
            return "MToonIntegrator[]"

    mi.register_integrator("mtoon", lambda props: _MToonIntegrator(props))
    _REGISTERED = True


def integrator_dict(base, shade, light=(-1.0, -0.2, -0.6), light_color=(1.0, 1.0, 1.0),
                    shadows=True, **kw):
    p = dict(mtoon.DEFAULTS, **kw)
    d = {"type": "mtoon", "shadows": 1.0 if shadows else 0.0}
    for i, name in enumerate("rgb"):
        d["base_" + name] = float(base[i])
        d["shade_" + name] = float(shade[i])
        d["rim_" + name] = float(p["parametric_rim_color_factor"][i])
        d["light_" + name] = float(light_color[i])
    for i, name in enumerate("xyz"):
        d["light_" + name] = float(light[i])
    for k in mtoon.PROP_KEYS:
        d[k] = float(p[k])
    return d


def render(base, shade, res=96, **kw):
    import mitsuba as mi
    register()
    scene = mi.load_dict({
        "type": "scene",
        "integrator": integrator_dict(base, shade, **kw),
        "sensor": {
            "type": "perspective", "fov": 40,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": res, "height": res,
                     "rfilter": {"type": "box"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent", "sample_count": 4},
        },
        "ball": {"type": "sphere", "radius": 1.0,
                 "bsdf": {"type": "diffuse"}},
    })
    return np.array(mi.render(scene, spp=4))


def body_pixels(image):
    flat = image.reshape(-1, 3)
    return flat[flat.sum(axis=1) > 1e-9]


def plateaus(image, decimals=3):
    """Lightest and darkest colour covering a tenth of the body or more."""
    body = body_pixels(image)
    if len(body) < 20:
        return None, None
    rows, counts = np.unique(np.round(body, decimals), axis=0, return_counts=True)
    keep = rows[counts >= max(3, len(body) // 10)]
    if len(keep) == 0:
        return None, None
    luma = keep @ np.array([0.2126, 0.7152, 0.0722])
    return keep[int(np.argmax(luma))], keep[int(np.argmin(luma))]


def self_test():
    """Eight controls. Five must reject a render the BSDF route would have passed."""
    r = []
    base, shade = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)

    img = render(base, shade, shading_toony_factor=1.0, shading_shift_factor=0.0)
    hi, lo = plateaus(img)
    r.append(("the sphere renders", hi is not None))
    r.append(("the shade plateau reaches film", lo is not None and lo.sum() > 1e-3))
    r.append(("the lit plateau is the base colour",
              hi is not None and mtoon.delta_e(list(hi), list(base)) < 1.0))
    r.append(("the shaded plateau is the shade colour",
              lo is not None and mtoon.delta_e(list(lo), list(shade)) < 1.0))
    r.append(("rendered contrast equals the material contrast",
              abs(mtoon.delta_e(list(hi), list(lo))
                  - mtoon.delta_e(list(base), list(shade))) < 1.0))

    flat = render(base, base, shading_toony_factor=1.0)
    fhi, flo = plateaus(flat)
    r.append(("base == shade renders no contrast",
              fhi is not None and mtoon.delta_e(list(fhi), list(flo)) < 1.0))

    hard = render(base, shade, shading_toony_factor=1.0)
    soft = render(base, shade, shading_toony_factor=0.0)
    r.append(("a hard ramp has fewer distinct values than a soft one",
              len(np.unique(np.round(body_pixels(hard), 3), axis=0))
              < len(np.unique(np.round(body_pixels(soft), 3), axis=0))))

    other = render((0.2, 0.2, 0.9), (0.05, 0.05, 0.3), shading_toony_factor=1.0)
    r.append(("a different material renders differently", not np.allclose(img, other)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    if args.self_test:
        return self_test()
    ap.error("pass --self-test")


if __name__ == "__main__":
    sys.exit(main())
