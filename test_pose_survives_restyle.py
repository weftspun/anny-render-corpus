"""Does OmniGen2 hold the pose through a restyle? T05, measured rather than argued.

THE QUESTION. RFD 107a rule 4 says a restyle that moves a limb makes the label a lie, and the
whole corpus rests on the label surviving. OmniGen2 has no depth control -- its `__call__`
takes prompt, negative_prompt, input_images and two guidance scales, and nothing else -- so
geometry preservation rests on image guidance alone. The plan flags the usable window as
possibly EMPTY, and an empty window is a signal to reconsider rather than a tuning problem.

WHAT IS MEASURED. Joint drift in pixels: run the keypoint detector on the source render and
on each restyle, pair the detections, and report how far each point moved. That is the
physical quantity. A silhouette score would be the convenient proxy -- it can be high while
an elbow folds the wrong way, which is exactly the failure rule 4 exists to catch.

THE SOURCE VIEWS ARE THE WHOLE HAMMERSLEY SEQUENCE, not the members that looked best.
`check_view_selection.py` exists because that mistake was made here once already: view 0
comes out looking straight down, and choosing index 3 instead because it looked more like a
person is a pitch decision wearing the sequence's clothes.

WHAT A FAILURE LOOKS LIKE, STATED FIRST. If the detector finds no person in a restyle, that is
not zero drift -- it is a restyle that destroyed the subject, and it is counted separately.
Reporting only the mean over frames that happened to detect would flatter exactly the frames
that failed worst.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Rule 2's two OmniGen2 appearances, unchanged from omnigen2_edit.py so the two files cannot
# drift into testing different prompts than the corpus uses.
PROMPTS = {
    "photographic": ("Turn this into a photograph of a real person in the same pose, natural "
                     "skin and fabric, studio lighting, plain dark background. Do not move "
                     "the body."),
    "colour-sketch": ("Turn this into a coloured pencil sketch of the same figure in the same "
                      "pose, visible strokes, paper texture, plain background. Do not move "
                      "the body."),
}
NEGATIVE = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn "
            "face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused "
            "fingers, messy drawing, broken legs censor, censored, censor_bar")


def px_to_household(p, image_px, subject_m=1.7):
    """Pixels to millimetres on the subject, with an anchor. A drift of '12 px' says nothing
    about whether a label is still true; 'about eight stacked pennies' does."""
    mm = p / image_px * subject_m * 1000.0
    anchors = [(1.52, "stacked pennies"), (7.0, "pencil widths"), (14.5, "AA batteries"),
               (21.2, "nickels"), (42.7, "golf balls"), (66.0, "soda cans")]
    best = min(anchors, key=lambda a: abs(mm / a[0] - 3.0))
    return mm, "about %.1f %s" % (mm / best[0], best[1])


def detect(model, path, threshold=0.3):
    """(K,2) keypoints for the highest-scoring person, or None if nothing was found."""
    from PIL import Image
    res = model.predict(Image.open(path).convert("RGB"), threshold=threshold)
    res = res[0] if isinstance(res, list) else res
    xy = getattr(res, "xy", None)
    if xy is None or len(xy) == 0:
        return None
    return np.asarray(xy[0], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=r"C:\anny_test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--text-guidance", type=float, default=5.0)
    ap.add_argument("--image-guidance", type=float, default=2.0)
    ap.add_argument("--views", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(a.renders, "hv_*.png")))[:a.views]
    if not sources:
        sys.exit("FAIL  no hv_*.png renders in %s" % a.renders)
    idx = [int(os.path.basename(s).split("_")[1].split(".")[0]) for s in sources]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import check_view_selection as V
    for p in V.check(idx, a.views):
        print("  ..   view selection: %s" % p)

    import torch
    from PIL import Image
    from rfdetr import RFDETRKeypointPreview
    det = RFDETRKeypointPreview(resolution=576, num_windows=1)

    from diffusers import BitsAndBytesConfig as DBnb          # noqa: F401  (kept for parity)
    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    repo = "OmniGen2/OmniGen2"
    tr = OmniGen2Transformer2DModel.from_pretrained(repo, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16)
    ml = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, subfolder="mllm",
                                                            torch_dtype=torch.bfloat16)
    pipe = OmniGen2Pipeline.from_pretrained(repo, transformer=tr, mllm=ml,
                                            torch_dtype=torch.bfloat16, trust_remote_code=True)
    pipe.to("cuda")

    rows, no_detect = [], 0
    for src in sources:
        base = detect(det, src)
        if base is None:
            print("  ..   %s: no person in the SOURCE render, skipped" % os.path.basename(src))
            continue
        img = Image.open(src).convert("RGB")
        for name, prompt in PROMPTS.items():
            out = pipe(prompt=prompt, input_images=[img], width=img.width, height=img.height,
                       num_inference_steps=a.steps, max_sequence_length=1024,
                       text_guidance_scale=a.text_guidance,
                       image_guidance_scale=a.image_guidance,
                       negative_prompt=NEGATIVE, num_images_per_prompt=1,
                       generator=torch.Generator(device="cuda").manual_seed(0),
                       output_type="pil").images[0]
            dst = os.path.join(a.out, "%s_%s.png" % (os.path.basename(src)[:-4], name))
            out.save(dst)
            got = detect(det, dst)
            if got is None:
                no_detect += 1
                print("  !!   %-28s NO PERSON DETECTED after restyle" % os.path.basename(dst))
                rows.append({"src": src, "style": name, "detected": False})
                continue
            n = min(len(base), len(got))
            d = np.linalg.norm(base[:n] - got[:n], axis=1)
            mm, house = px_to_household(float(d.mean()), img.width)
            rows.append({"src": src, "style": name, "detected": True,
                         "mean_px": float(d.mean()), "max_px": float(d.max()), "mm": mm})
            print("  ..   %-28s mean %6.1f px  max %6.1f px  (%s)"
                  % (os.path.basename(dst), d.mean(), d.max(), house))

    ok = [r for r in rows if r.get("detected")]
    print("\n%d restyles, %d with a detectable person, %d without"
          % (len(rows), len(ok), no_detect))
    if ok:
        m = np.mean([r["mean_px"] for r in ok])
        w = max(r["max_px"] for r in ok)
        mm, house = px_to_household(float(m), 512)
        print("mean joint drift %.1f px (%s), worst single joint %.1f px" % (m, house, w))
    json.dump(rows, open(os.path.join(a.out, "drift.json"), "w"), indent=1)
    # A frame with no detection is a failure, not a missing sample.
    return 1 if no_detect else 0


if __name__ == "__main__":
    sys.exit(main())
