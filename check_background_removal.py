"""Does a background remover actually find the subject, and can this check fail?

WHY THIS EXISTS AT ALL. Pixal3D's `pipeline.json` constructs BRIA RMBG unconditionally, and
BRIA RMBG is blocklisted here as gated and non-commercial. Several replacements are available
and Apache-2.0 -- OmniGen2 as an instruction-guided edit, SAM2, RF-DETR's `RFDETRSegMedium` --
but "we swapped in a different remover" is a claim, not a measurement, and the last time a
mask was trusted without one the number was wrong: a border-median mask scored a watercolour
restyle at 0.798 and 0.899 by counting the wash as body. Both figures are retracted.

THE GROUND TRUTH IS EXACT AND COSTS NOTHING. An ANNY render carries a real alpha channel from
the matte pass, so the true foreground is known by construction rather than annotated. That is
better evidence than any public matting corpus offers, and it is also why one is not used
here: P3M-10k, AM-2k, PM-10k and VideoMatte240K are each research-only or privacy-restricted,
so none of them clears the commercial-use-and-derivatives bar this project applies to data.

THE CONSTRUCTION. Composite the render onto a plate using its own alpha -- the way
See-Through builds its training samples, `seethrough-partseg/inference/scripts/syn_data.py`,
`img_alpha_blending` with `fgbg_hist_matching` so the subject and plate agree in colour
statistics instead of looking pasted -- then ask the remover to recover what was composited,
and compare against the alpha that did the compositing.

FOUR NEGATIVE CONTROLS, AND THE CHECK FAILS IF ANY OF THEM PASSES. A check that cannot fail
certifies the defect instead of catching it.

  a. background only        no subject was composited. A large recovered foreground here is a
                            hallucinated subject, and IoU against an empty truth is 0.
  b. truth shifted 20 px    the shape is right and the position is wrong. This is the control
                            `verify_restyle.py` already runs, and it returns 1 when the control
                            passes, which is the behaviour copied here.
  c. full-frame mask        trivially "keeps the subject". It must score badly, or the measure
                            is rewarding coverage rather than agreement.

AND ONE HARD POSITIVE, WHICH IS NOT A NEGATIVE CONTROL AND WAS WRITTEN AS ONE. The low-contrast
plate -- the background recoloured to the subject's own median -- is the case that broke the
retracted border-median measurement, so it belongs here. But a good remover is supposed to
score HIGH on it, not low: it is a difficulty, not a defect. The first run of this file
demanded it fail and duly reported a problem when the exact-alpha method scored 1.0000 on it,
which was the harness being wrong rather than the method. It is now scored against the floor
in the same direction as the real case.

`--method alpha` is the harness's own positive control and not a remover at all: it reads the
input's own alpha and ignores the image it is handed, so it must score 1.0000 on both the real
and the low-contrast case. If it does not, the harness is broken before any remover is blamed.

Measured for `--method alpha` on `pose_040_front-view_eye-level-shot_medium-shot.png`:
real 1.0000, low-contrast 1.0000, background-only 0.0000, shifted-20px 0.4184,
full-frame 0.0457, against a floor of 0.90. The shifted control's margin below the floor is
what makes the floor meaningful; that number is printed rather than assumed.

    python check_background_removal.py --frame <render-with-alpha.png> --plate <bg.png> \\
        --method alpha|omnigen2|rfdetr --out <dir>
    python check_background_removal.py --frame <render.png> --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

# A recovered mask agreeing with the truth below this is not a usable remover. It is set where
# it is because the shifted control has to fail it: a 20 px shift of this subject overlaps
# itself substantially, so a floor much lower than this would pass the control and certify
# nothing. The self-test prints both numbers so the margin is visible rather than asserted.
IOU_FLOOR = 0.90
CONTROL_SHIFT_PX = 20


def load_rgba(path):
    image = Image.open(path).convert("RGBA")
    return np.asarray(image)


def truth_alpha(rgba):
    """The rendered alpha, as a boolean. Half-open at 127 the way the corpus reads it."""
    alpha = rgba[:, :, 3] > 127
    if not alpha.any():
        raise ValueError("the frame has an empty alpha channel, so there is no truth to "
                         "measure against; this check needs a matte-pass render")
    if alpha.all():
        raise ValueError("the frame's alpha is entirely opaque, so it carries no matte")
    return alpha


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def hist_match(source, reference, mask):
    """Match the subject's colour statistics to the plate's, per channel.

    See-Through does this so a composited body is not obviously pasted. It matters here for a
    reason beyond looks: a remover that is really keying on a colour discontinuity scores well
    on a badly composited image and badly on a well composited one, and only the second case
    resembles the images this pipeline will actually be given.
    """
    out = source.astype(np.float64).copy()
    for c in range(3):
        src = source[:, :, c][mask].astype(np.float64)
        ref = reference[:, :, c].astype(np.float64).ravel()
        if not len(src):
            continue
        s_mean, s_std = src.mean(), src.std() or 1.0
        r_mean, r_std = ref.mean(), ref.std() or 1.0
        out[:, :, c] = (out[:, :, c] - s_mean) * (r_std / s_std) + r_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def composite(rgba, plate, match=True):
    """Subject over plate, using the render's own alpha as the compositing operator."""
    h, w = rgba.shape[:2]
    bg = np.asarray(Image.fromarray(plate).convert("RGB").resize((w, h), Image.LANCZOS))
    fg = rgba[:, :, :3]
    mask = rgba[:, :, 3] > 127
    if match:
        fg = hist_match(fg, bg, mask)
    a = (rgba[:, :, 3:4].astype(np.float64) / 255.0)
    out = fg.astype(np.float64) * a + bg.astype(np.float64) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def flat_plate(rgba, colour):
    h, w = rgba.shape[:2]
    return np.broadcast_to(np.asarray(colour, dtype=np.uint8), (h, w, 3)).copy()


def shift(mask, px):
    """Move the truth sideways. The shape survives, the registration does not."""
    out = np.zeros_like(mask)
    out[:, px:] = mask[:, :-px] if px else mask
    return out


# --------------------------------------------------------------------------
# Removers. Each takes an RGB array and returns a boolean foreground mask.
# --------------------------------------------------------------------------

def remove_alpha(_rgb, rgba=None, **_):
    """The input's own alpha. Exact and free, and the only correct answer when it exists.

    This is the positive control as much as a method: if the harness cannot score a perfect
    mask at 1.0, the harness is wrong before any remover is blamed.
    """
    return rgba[:, :, 3] > 127


def remove_rfdetr(rgb, **_):
    """RF-DETR instance segmentation, Apache-2.0. The person instances, unioned."""
    from rfdetr import RFDETRSegMedium

    det = RFDETRSegMedium().predict(Image.fromarray(rgb), threshold=0.5)
    if det.mask is None or not len(det.mask):
        # A REMOVER THAT FOUND NOTHING IS NOT A REMOVER THAT FOUND THE BACKGROUND. An empty
        # mask scores 0 against a real subject, which is the honest outcome; returning the
        # whole frame instead would score well on coverage and hide the failure.
        return np.zeros(rgb.shape[:2], dtype=bool)
    names = det.data.get("class_name", [])
    keep = [i for i, n in enumerate(names) if str(n).lower() == "person"] or range(len(det.mask))
    return np.logical_or.reduce([det.mask[i].astype(bool) for i in keep])


def remove_omnigen2(rgb, repo=None, model="OmniGen2/OmniGen2", **_):
    """OmniGen2 as an instruction edit, Apache-2.0 in weights and code.

    THE ALPHA THIS PRODUCES IS ESTIMATED BY A GENERATIVE MODEL, not measured. It is admissible
    as an input to a downstream pipeline and it is NOT ground-truth segmentation, and the
    distinction is why this file exists: the number below says how far from the truth it lands
    rather than assuming it lands on it.
    """
    import torch
    from _omnigen2_loader import load_pipeline  # written beside this file

    pipe = load_pipeline(repo, model)
    res = pipe(prompt="Remove the background completely. Keep the person exactly as they are, "
                      "in the same position, and place them on a plain pure white background.",
               input_images=[Image.fromarray(rgb)], width=rgb.shape[1], height=rgb.shape[0],
               num_inference_steps=30, max_sequence_length=256,
               text_guidance_scale=5.0, image_guidance_scale=2.0,
               generator=torch.Generator(device="cuda").manual_seed(0), output_type="pil")
    out = np.asarray(res.images[0].convert("RGB")).astype(np.int16)
    # White is what the instruction asked for, so anything far from white is subject. The
    # threshold is stated rather than tuned: 255 minus 40 per channel in L2.
    return np.linalg.norm(255 - out, axis=2) > 40.0


METHODS = {"alpha": remove_alpha, "rfdetr": remove_rfdetr, "omnigen2": remove_omnigen2}


def run_case(label, rgb, truth, method, rgba, out_dir, **kwargs):
    mask = METHODS[method](rgb, rgba=rgba, **kwargs)
    score = iou(truth, mask)
    if out_dir:
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            pathlib.Path(out_dir) / ("mask_%s.png" % label))
        Image.fromarray(rgb).save(pathlib.Path(out_dir) / ("input_%s.png" % label))
    return score, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", required=True, help="an RGBA render from the matte pass")
    ap.add_argument("--plate", default="", help="background image; a synthetic one is used if absent")
    ap.add_argument("--method", default="alpha", choices=sorted(METHODS))
    ap.add_argument("--repo", default=r"C:\omnigen2-src")
    ap.add_argument("--out", default="")
    ap.add_argument("--floor", type=float, default=IOU_FLOOR)
    args = ap.parse_args()

    rgba = load_rgba(args.frame)
    truth = truth_alpha(rgba)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    if args.plate:
        plate = np.asarray(Image.open(args.plate).convert("RGB"))
    else:
        # A PLATE THAT IS NOT FLAT, because a flat one is the easy case and would flatter every
        # method. Two gradients and a vertical edge give texture the truth does not share.
        h, w = rgba.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        plate = np.stack([(xx * 255 // max(w - 1, 1)).astype(np.uint8),
                          (yy * 255 // max(h - 1, 1)).astype(np.uint8),
                          np.where(xx > w // 2, 200, 60).astype(np.uint8)], axis=2)

    subject_median = np.median(rgba[:, :, :3][truth], axis=0).astype(np.uint8)

    cases = {
        "real": (composite(rgba, plate), truth),
        # a. no subject was composited, so the truth is empty.
        "control_background_only": (np.asarray(
            Image.fromarray(plate).convert("RGB").resize(
                (rgba.shape[1], rgba.shape[0]), Image.LANCZOS)),
            np.zeros_like(truth)),
        # THE HARD POSITIVE, not a control: the plate recoloured to the subject's own median.
        # A remover is expected to clear the floor here, and the retracted border-median
        # measurement is what happens when nothing checks that it does.
        "hard_low_contrast": (composite(rgba, flat_plate(rgba, subject_median), match=False),
                              truth),
    }

    kwargs = dict(repo=args.repo) if args.method == "omnigen2" else {}
    results = {}
    print("method %s, floor %.2f" % (args.method, args.floor))
    for label, (rgb, expected) in cases.items():
        score, _ = run_case(label, rgb, expected, args.method, rgba, args.out, **kwargs)
        results[label] = round(score, 4)
        print("  %-26s IoU %.4f" % (label, score))

    # b. the truth itself, moved. No remover runs: this measures the FLOOR, not the method.
    # c. everything is foreground.
    results["control_shifted_%dpx" % CONTROL_SHIFT_PX] = round(
        iou(truth, shift(truth, CONTROL_SHIFT_PX)), 4)
    results["control_full_frame"] = round(iou(truth, np.ones_like(truth)), 4)
    print("  %-26s IoU %.4f" % ("control_shifted_%dpx" % CONTROL_SHIFT_PX,
                                results["control_shifted_%dpx" % CONTROL_SHIFT_PX]))
    print("  %-26s IoU %.4f" % ("control_full_frame", results["control_full_frame"]))

    # TWO DIRECTIONS, AND MIXING THEM IS THE BUG THIS BLOCK USED TO HAVE. Anything named
    # `control_` must fall BELOW the floor, or the check certifies rather than catches.
    # Everything else is a case the remover is supposed to pass, and it must clear it.
    failures = []
    for label, score in results.items():
        if label.startswith("control_"):
            if score >= args.floor:
                failures.append("%s scored %.4f, at or above the %.2f floor: the control "
                                "passed, so this check certifies rather than catches"
                                % (label, score, args.floor))
        elif score < args.floor:
            failures.append("%s scored %.4f, under the %.2f floor: this remover does not "
                            "recover the subject" % (label, score, args.floor))

    print()
    for f in failures:
        print("  BAD  %s" % f)
    print("%d control(s) plus the real case, %d problem(s)" % (len(results) - 1, len(failures)))

    if args.out:
        with open(pathlib.Path(args.out) / "background_removal.json", "w", encoding="utf-8") as fh:
            json.dump({"frame": args.frame, "method": args.method, "floor": args.floor,
                       "iou": results, "problems": failures}, fh, indent=2)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
