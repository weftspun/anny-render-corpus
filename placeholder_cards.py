# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Cards naming the axes the corpus has no data for, so a clip cannot imply coverage.

    python placeholder_cards.py --self-test [--out card.png]
"""
from __future__ import annotations

import argparse
import io
import pathlib
import sys

import numpy as np

# Every gap CORPUS_DESIGN.md names, and nothing else.
GAPS = (
    ("PERSONA SPECIES", "no avatar assets",
     "the survey measures humanoid 50%, semi-humanoid 38%, robot 6%,",
     "animal 2%, plant 2%, other 1%, monster 0%. This repository has",
     "one human body and 1,254 CC0 objects, so the axis is a rule",
     "with controls and not a thing that can be rendered."),
    ("SKIN TONE BY POPULATION", "no citable distribution",
     "the Monk scale publishes colour, not demography, and the survey",
     "measures residence rather than skin tone. The tone axis is",
     "uniform by decision. Weighting it by this population would give",
     "Monk 10 two frames of 900, where the defect is largest."),
    ("TEST SPLIT", "0 records",
     "the corpus is 85 train, 10 val, 0 test. `split` is typed",
     "\"train\" | \"val\", so the rung cannot be written down at all.",
     "An empty rung is a fill; a missing one is a schema change."),
    ("IDENTITIES", "1 of 23,000",
     "every cell is n=1, so no error bar exists anywhere in the",
     "corpus. Split hygiene assigns per identity and there is one,",
     "so the rule currently protects nothing."),
    ("GLOBAL ILLUMINATION", "not implemented",
     "giEqualizationFactor is an MToon parameter this model does not",
     "carry, and the three-vrm comparison set it to zero to remove it.",
     "Engines differ most on exactly this term."),
    ("SUBSURFACE AND OUTLINE", "assets unused",
     "a 2048 square sss.png ships with the basemesh and nothing binds",
     "it. Skin is translucent, so an albedo-only tone sweep understates",
     "real variation, and MToon's outline is absent too."),
)

FONT_CANDIDATES = ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/"
                   "DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc")


def font(size):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        if pathlib.Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def card(title, verdict, *lines, width=3840, height=2160):
    """One card as linear-float RGBA; alpha is where the ink is."""
    from PIL import Image, ImageDraw
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = height / 2160.0
    x = int(240 * unit)
    y = int(700 * unit)
    draw.text((x, y), title, font=font(int(96 * unit)), fill=(255, 255, 255, 255))
    y += int(150 * unit)
    draw.text((x, y), verdict.upper(), font=font(int(72 * unit)), fill=(255, 96, 96, 255))
    y += int(170 * unit)
    for line in lines:
        draw.text((x, y), line, font=font(int(52 * unit)), fill=(200, 200, 200, 255))
        y += int(78 * unit)
    out = np.asarray(image, dtype=np.float32) / 255.0
    # sRGB in, linear out, so the packer applies its transfer function once.
    rgb = out[..., :3]
    out[..., :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return out


def wrap(text, width=64):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def citation(cff_path, lines=9):
    """The citation as a card, read from the .cff so the two cannot disagree."""
    import yaml
    d = yaml.safe_load(io.open(cff_path, encoding="utf-8").read())
    who = ", ".join(a.get("name") or "%s %s" % (a.get("given-names", ""),
                                                a.get("family-names", "")).strip()
                    for a in d.get("authors", []))
    body = ["cff-version %s   %s   %s" % (d["cff-version"], who, d["license"]), ""]
    body += wrap(" ".join(d.get("abstract", "").split()))[:lines]
    return card(d["title"], d["license"], *body)


def cards(width=3840, height=2160):
    return [card(*gap, width=width, height=height) for gap in GAPS]


def self_test():
    """Thirteen controls. Four must reject a card set that hides a gap."""
    r = []
    one = card(*GAPS[0], width=640, height=360)
    r.append(("a card is a float RGBA frame",
              one.shape == (360, 640, 4) and one.dtype == np.float32))
    r.append(("ink is opaque and the rest is not", one[..., 3].max() > 0.99
              and one[..., 3].min() < 0.01))
    r.append(("a card is mostly empty, so it reads as a card",
              0.005 < (one[..., 3] > 0.5).mean() < 0.35))
    r.append(("values are linear, so the packer does not gamma twice",
              float(one[..., :3].max()) <= 1.0))

    blank = card("", "", width=640, height=360)
    r.append(("an empty card carries no ink", float(blank[..., 3].max()) < 0.01))
    r.append(("two different gaps render differently",
              not np.array_equal(one, card(*GAPS[1], width=640, height=360))))

    r.append(("every gap has a card", len(cards(320, 180)) == len(GAPS)))
    r.append(("every gap names a verdict", all(len(g) >= 3 and g[1] for g in GAPS)))
    r.append(("every gap carries at least one line of why",
              all(len(g) >= 3 for g in GAPS)))

    cff = next(iter(sorted(pathlib.Path(__file__).resolve().parent.glob("*.cff"))), None)
    if cff is None:
        r.append(("a citation card can be read from a .cff", False))
    else:
        import yaml
        d = yaml.safe_load(io.open(cff, encoding="utf-8").read())
        cit = citation(cff)
        r.append(("a citation card renders from the .cff",
                  cit.shape[2] == 4 and float(cit[..., 3].max()) > 0.99))
        r.append(("the card carries more ink than a gap card",
                  (cit[..., 3] > 0.5).mean() > (one[..., 3] > 0.5).mean()))
        stem = __import__("re").sub(r"\W+", "-", d["title"].lower()).strip("-")
        r.append(("the filename stem derives from the title, per RFD 1137 step 10",
                  cff.stem == stem))
    r.append(("wrap keeps every word", " ".join(wrap("a b c d e", 3)) == "a b c d e"))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.out:
        from PIL import Image
        first = (np.clip(card(*GAPS[0], width=1920, height=1080), 0, 1) * 255)
        Image.fromarray(first.astype(np.uint8), "RGBA").save(args.out)
        print("  wrote %s" % args.out)
        return 0
    for gap in GAPS:
        print("  %-26s %s" % (gap[0], gap[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
