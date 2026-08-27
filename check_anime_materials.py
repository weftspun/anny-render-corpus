# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gate: the anime material layer's tone scale holds contrast equal across every tone.

Re-derives the shade multiplier the layer deliberately does not store, and fails when a tone
cannot reach the target rather than clamping it.

    python check_anime_materials.py [layer.usda] [--self-test]
"""
from __future__ import annotations

import argparse
import math
import sys

from pxr import Usd

FLAT = 0.6          # the naive multiplier this gate exists to reject
TOLERANCE = 0.05    # dE, how close a solved tone must land on the target


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lab(rgb_linear):
    m = ((.4124, .3576, .1805), (.2126, .7152, .0722), (.0193, .1192, .9505))
    x, y, z = (sum(m[i][j] * rgb_linear[j] for j in range(3)) for i in range(3))
    white = (.95047, 1.0, 1.08883)

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = f(x / white[0]), f(y / white[1]), f(z / white[2])
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e(a, b):
    return math.dist(lab(a), lab(b))


def linear_of(hex_string):
    h = hex_string.lstrip("#")
    return [srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]


def solve_multiplier(lit, target):
    """The multiplier landing the shade colour `target` dE from the lit colour."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if delta_e(lit, [c * mid for c in lit]) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def read(path):
    stage = Usd.Stage.Open(path)
    if not stage:
        raise SystemExit("FAIL  %s does not open" % path)
    scale = stage.GetPrimAtPath("/AnimeMaterials/ToneScale")
    if not scale:
        raise SystemExit("FAIL  %s has no /AnimeMaterials/ToneScale" % path)
    target = scale.GetAttribute("targetDeltaE").Get()
    tones = []
    for prim in scale.GetChildren():
        tone = prim.GetAttribute("monkTone").Get()
        hexs = prim.GetAttribute("srgbHex").Get()
        if tone is None or not hexs:
            raise SystemExit("FAIL  %s carries no monkTone or srgbHex" % prim.GetPath())
        tones.append((int(tone), str(hexs)))
    return target, sorted(tones)


def check(path, verbose=True):
    target, tones = read(path)
    problems = []
    if len(tones) != 10:
        problems.append("the scale has %d rows, not the 10 Monk levels" % len(tones))
    if [t for t, _ in tones] != list(range(1, len(tones) + 1)):
        problems.append("tones are not 1..%d without a gap" % len(tones))

    flat, solved = [], []
    for tone, hexs in tones:
        lit = linear_of(hexs)
        flat_de = delta_e(lit, [c * FLAT for c in lit])
        mult = solve_multiplier(lit, target)
        got = delta_e(lit, [c * mult for c in lit])
        flat.append(flat_de)
        solved.append(got)
        if abs(got - target) > TOLERANCE:
            problems.append("MST %d reaches only dE %.2f against a target of %.2f"
                            % (tone, got, target))
        if verbose:
            print("  MST %2d  #%s  flat x%.1f -> dE %5.2f   solved x%.4f -> dE %5.2f"
                  % (tone, hexs.lstrip("#"), FLAT, flat_de, mult, got))

    if verbose and flat:
        print()
        print("  flat multiplier spread   %.2f to %.2f dE  (%.1fx)"
              % (min(flat), max(flat), max(flat) / min(flat)))
        print("  solved spread            %.2f to %.2f dE  (%.1fx)"
              % (min(solved), max(solved), max(solved) / min(solved)))
    # The whole point: the flat multiplier must be shown to be unequal, or this gate is
    # asserting a property nothing was ever at risk of violating.
    if flat and max(flat) / min(flat) < 2.0:
        problems.append("the flat multiplier is not measurably unequal here, so this gate "
                        "certifies nothing")
    return problems


def self_test():
    """Five controls. Three must reject a scale a careless generator would have produced."""
    import tempfile
    import pathlib
    results = []
    d = pathlib.Path(tempfile.mkdtemp())

    def layer(name, body, target=12.0):
        p = d / name
        p.write_text('#usda 1.0\n(defaultPrim = "AnimeMaterials")\n'
                     'def Scope "AnimeMaterials" {\n def Scope "ToneScale" {\n'
                     '  float targetDeltaE = %r\n%s\n }\n}\n' % (target, body),
                     encoding="utf-8")
        return str(p)

    every = "".join('  def "MST%02d"\n  {\n   int monkTone = %d\n   string srgbHex = "#%s"\n  }\n'
                    % (i, i, h) for i, h in enumerate(
                        ["f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96",
                         "a07e56", "825c43", "604134", "3a312a", "292420"], 1))
    results.append(("the full scale passes", not check(layer("ok.usda", every), False)))

    short = "".join('  def "MST%02d"\n  {\n   int monkTone = %d\n   string srgbHex = "#%s"\n  }\n'
                    % (i, i, h) for i, h in enumerate(["f6ede4", "f3e7db", "f7ead0"], 1))
    results.append(("a scale missing tones is rejected",
                    bool(check(layer("short.usda", short), False))))

    # Light tones only: the flat multiplier looks fine because the dark end is absent, which
    # is exactly how a corpus certifies an equity it never tested.
    light = "".join('  def "MST%02d"\n  {\n   int monkTone = %d\n   string srgbHex = "#%s"\n  }\n'
                    % (i, i, h) for i, h in enumerate(
                        ["f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96",
                         "f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96"], 1))
    results.append(("a light-only scale is rejected as certifying nothing",
                    bool(check(layer("light.usda", light), False))))

    results.append(("an unreachable target is reported, not clamped",
                    bool(check(layer("hard.usda", every, target=60.0), False))))

    results.append(("the solver hits its target on a known tone",
                    abs(delta_e(linear_of("#292420"),
                                [c * solve_multiplier(linear_of("#292420"), 12.0)
                                 for c in linear_of("#292420")]) - 12.0) < TOLERANCE))

    bad = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(results) - bad, len(results)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layer", nargs="?", default="anime_materials.usda")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    problems = check(args.layer)
    print()
    for p in problems:
        print("FAIL  %s" % p)
    if problems:
        return 1
    print("the tone scale holds contrast equal across all 10 tones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
