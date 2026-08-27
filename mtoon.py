# SPDX-License-Identifier: Apache-2.0 OR MIT
"""MToon 3.3 as a Mitsuba BSDF, ported from V-Sekai's Godot-MToon-Shader (MIT).

    python mtoon.py --self-test
    python mtoon.py --sweep out_dir
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

EPS_COL = 0.00001

DEFAULTS = {
    "shade_shift": 0.0,
    "shade_toony": 0.9,
    "shading_grade_rate": 1.0,
    "shading_grade": 1.0,
    "light_attenuation": 1.0,
    "rim_color": (0.0, 0.0, 0.0),
    "rim_lift": 0.0,
    "rim_fresnel_power": 1.0,
    "rim_lighting_mix": 0.0,
    "light_color_attenuation": 0.0,
}


def light_intensity(dot_nl, shade_shift, shade_toony,
                    light_attenuation=1.0, shading_grade_rate=1.0, shading_grade=1.0):
    """The toon ramp, from calculateLighting()."""
    grade = 1.0 - shading_grade_rate * (1.0 - shading_grade)
    i = dot_nl * 0.5 + 0.5
    i = i * light_attenuation
    i = i * grade
    i = i * 2.0 - 1.0
    hi = shade_toony * shade_shift + (1.0 - shade_toony) * 1.0
    lo = shade_shift
    return float(np.clip((i - lo) / max(EPS_COL, hi - lo), 0.0, 1.0))


def shade_mix(dot_nl, lit, shade, **kw):
    p = dict(DEFAULTS, **kw)
    t = light_intensity(dot_nl, p["shade_shift"], p["shade_toony"],
                        p["light_attenuation"], p["shading_grade_rate"], p["shading_grade"])
    lit, shade = np.asarray(lit, float), np.asarray(shade, float)
    return shade + (lit - shade) * t, t


def rim_term(dot_nv, **kw):
    p = dict(DEFAULTS, **kw)
    f = np.clip(1.0 - dot_nv + p["rim_lift"], 0.0, 1.0) ** max(p["rim_fresnel_power"], 0.001)
    return np.asarray(p["rim_color"], float) * f * p["rim_lighting_mix"]


def light_color_atten(light_color, amount):
    c = np.asarray(light_color, float) / math.pi
    return c + (max(EPS_COL, c.max()) - c) * amount


class MToonBSDF:
    """MToon is col * lightColor / PI with no cosine; it is already inside the ramp."""

    def __init__(self, lit, shade, **kw):
        self.lit, self.shade, self.params = lit, shade, kw

    def eval(self, dot_nl, dot_nv):
        col, _ = shade_mix(dot_nl, self.lit, self.shade, **self.params)
        return col + rim_term(dot_nv, **self.params)


_REGISTERED = False

PROP_KEYS = ("shade_shift", "shade_toony", "shading_grade_rate", "shading_grade",
             "light_attenuation", "rim_lift", "rim_fresnel_power", "rim_lighting_mix",
             "light_color_attenuation")


def scene_dict(lit, shade, **kw):
    """Parameters as a Mitsuba dict entry. Never a closure: register_bsdf is a no-op after
    the first call, so a captured model renders the first material for every later one."""
    p = dict(DEFAULTS, **kw)
    d = {"type": "mtoon"}
    for i, name in enumerate("rgb"):
        d["lit_" + name] = float(lit[i])
        d["shade_" + name] = float(shade[i])
        d["rim_" + name] = float(p["rim_color"][i])
    for k in PROP_KEYS:
        d[k] = float(p[k])
    return d


def register():
    """Register once; Mitsuba ignores repeats, so this does too."""
    global _REGISTERED
    if _REGISTERED:
        return
    import mitsuba as mi

    class _Plugin(mi.BSDF):
        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
            self.m_components = [self.m_flags]
            g = lambda k, d=0.0: float(props.get(k, d))  # noqa: E731
            self.lit = np.array([g("lit_" + c) for c in "rgb"])
            self.shade = np.array([g("shade_" + c) for c in "rgb"])
            self.kw = dict(DEFAULTS)
            self.kw["rim_color"] = tuple(g("rim_" + c) for c in "rgb")
            for k in PROP_KEYS:
                self.kw[k] = g(k, DEFAULTS[k])

        def _value(self, si, wo):
            nl = float(np.array(mi.Frame3f.cos_theta(wo)).reshape(-1)[0])
            nv = float(np.array(mi.Frame3f.cos_theta(si.wi)).reshape(-1)[0])
            col, _ = shade_mix(nl, self.lit, self.shade, **self.kw)
            return list((col + rim_term(nv, **self.kw)) / math.pi)

        def sample(self, ctx, si, sample1, sample2, active=True):
            bs = mi.BSDFSample3f()
            bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
            bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
            bs.eta, bs.sampled_type = 1.0, +mi.BSDFFlags.DiffuseReflection
            bs.sampled_component = 0
            return bs, mi.Spectrum(self._value(si, bs.wo))

        def eval(self, ctx, si, wo, active=True):
            return mi.Spectrum(self._value(si, wo))

        def pdf(self, ctx, si, wo, active=True):
            return mi.warp.square_to_cosine_hemisphere_pdf(wo)

        def eval_pdf(self, ctx, si, wo, active=True):
            return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

        def to_string(self):
            return "MToon[]"

    mi.register_bsdf("mtoon", lambda props: _Plugin(props))
    _REGISTERED = True


def sweep(lit, shade, samples=64, **kw):
    """The ramp over the hemisphere."""
    nl = np.linspace(-1.0, 1.0, samples)
    return nl, np.array([shade_mix(x, lit, shade, **kw)[1] for x in nl])


def _lab(rgb):
    m = ((.4124, .3576, .1805), (.2126, .7152, .0722), (.0193, .1192, .9505))
    x, y, z = (sum(m[i][j] * rgb[j] for j in range(3)) for i in range(3))
    w = (.95047, 1.0, 1.08883)
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29  # noqa: E731
    fx, fy, fz = f(x / w[0]), f(y / w[1]), f(z / w[2])
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e(a, b):
    return math.dist(_lab(a), _lab(b))


def rendered_contrast(lit, shade, **kw):
    """dE between the fully lit and fully shaded plateaus."""
    hi, _ = shade_mix(1.0, lit, shade, **kw)
    lo, _ = shade_mix(-1.0, lit, shade, **kw)
    return delta_e(list(hi), list(lo))


def self_test():
    """Nine controls. Five must reject a port that drifted from the shader."""
    r = []
    lit, shade = (0.8, 0.7, 0.6), (0.4, 0.35, 0.3)

    r.append(("full light saturates the ramp", light_intensity(1.0, 0.0, 0.9) == 1.0))
    r.append(("full shadow bottoms the ramp", light_intensity(-1.0, 0.0, 0.9) == 0.0))
    r.append(("the ramp is monotone in dot_nl",
              all(x <= y + 1e-12 for x, y in zip(*(lambda a: (a, a[1:]))(
                  [light_intensity(v, 0.0, 0.9) for v in np.linspace(-1, 1, 50)])))))
    hard = [light_intensity(v, 0.0, 1.0) for v in (-0.01, 0.01)]
    r.append(("shade_toony 1 gives a hard step", hard == [0.0, 1.0]))
    soft = [light_intensity(v, 0.0, 0.0) for v in (0.01, 0.5)]
    r.append(("shade_toony 0 ramps instead of stepping",
              soft[0] < 0.1 and abs(soft[1] - 0.5) < 1e-9 and hard[1] == 1.0))
    r.append(("shade_shift moves the terminator",
              light_intensity(0.0, 0.5, 0.9) < light_intensity(0.0, -0.5, 0.9)))
    r.append(("the mix reaches lit and shade exactly",
              np.allclose(shade_mix(1.0, lit, shade)[0], lit)
              and np.allclose(shade_mix(-1.0, lit, shade)[0], shade)))
    r.append(("rim is zero when rim_lighting_mix is zero",
              np.allclose(rim_term(0.5, rim_color=(1, 1, 1), rim_lighting_mix=0.0), 0.0)))
    r.append(("rim rises toward the silhouette",
              rim_term(0.1, rim_color=(1, 1, 1), rim_lighting_mix=1.0)[0]
              > rim_term(0.9, rim_color=(1, 1, 1), rim_lighting_mix=1.0)[0]))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ramp", action="store_true", help="print the ramp for a few toony values")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.ramp:
        for toony in (0.0, 0.5, 0.9, 1.0):
            nl, t = sweep((1, 1, 1), (0, 0, 0), samples=11, shade_toony=toony)
            print("  toony %.1f  " % toony + " ".join("%.2f" % v for v in t))
        return 0
    ap.error("pass --self-test or --ramp")


if __name__ == "__main__":
    sys.exit(main())
