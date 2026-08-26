"""Does OmniGen2 take more than one input image, and does a depth and pose pair change it?

THE CLAIM UNDER TEST, WHICH CAME FROM A DOCSTRING RATHER THAN A RUN.
`test_pose_survives_restyle.py` says OmniGen2 "has no depth control", and that was repeated
here as though geometry could only be held by image guidance. The pipeline call in
`omnigen2_edit.py` passes `input_images=[src]`, a list, so the narrow question is whether a
longer list is accepted and the wider one is whether it changes the output.

Three conditions, everything else held: same prompt, same seed, same guidance, same steps.

  A  input_images = [render]                  the current arm
  B  input_images = [render, depth]           depth as a second in-context image
  C  input_images = [render, depth, pose]     depth and the pose skeleton

An exception in B or C answers the narrow question one way. A run that completes answers it
the other, and then the error score says whether the geometry survived better, measured
against the render's own alpha rather than judged.

Silhouette agreement here is `verify_restyle.py`'s measure, and its retraction is inherited:
the background is estimated from the border rather than thresholded at a constant, because a
fixed threshold scored a ukiyo-e restyle at 0.098 by calling a warm grey background
foreground.

    python omnigen2_array_probe.py --repo <omnigen2 checkout> --frame <render.png> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

PROMPT = ("Restyle this character as a watercolour painting. Keep the pose, the proportions "
          "and the camera exactly as they are.")
NEGATIVE = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn "
            "face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused "
            "fingers, messy drawing, broken legs censor, censored, censor_bar")


def body_mask(image):
    """The figure, with the background estimated from the border rather than assumed."""
    a = np.asarray(image.convert("RGB"), dtype=np.float64)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    return np.linalg.norm(a - bg, axis=2) > 24.0


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", default="OmniGen2/OmniGen2")
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--text-guidance", type=float, default=5.0)
    ap.add_argument("--image-guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frame = pathlib.Path(args.frame)
    depth = frame.with_name(frame.stem + ".depth.png")
    pose = frame.with_name(frame.stem + ".pose.png")
    for p in (frame, depth, pose):
        if not p.is_file():
            sys.exit(f"FAIL  {p} is missing; run make_controls.py first")
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, args.repo)

    # model_index.json names its custom classes by BARE MODULE NAME, and diffusers resolves
    # those with a plain importlib call while validating components, so the weights' own
    # directories go on the path. `omnigen2_edit.py` carries the same workaround and the
    # reason: vendoring a copy would drift from the checkpoint it belongs to. This probe
    # failed without it, on ModuleNotFoundError: No module named 'transformer_omnigen2'.
    from huggingface_hub import hf_hub_download

    resolved = None
    for rel in ("transformer/transformer_omnigen2.py",
                "scheduler/scheduling_flow_match_euler_discrete.py"):
        local = hf_hub_download(args.model, rel)
        sys.path.insert(0, os.path.dirname(local))
        parts = os.path.normpath(local).split(os.sep)
        if "snapshots" in parts:
            resolved = parts[parts.index("snapshots") + 1]

    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import BitsAndBytesConfig as TransformersBnb
    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    # NF4 FOR BOTH HALVES, AND THIS RUN IS DEVICE EVIDENCE RATHER THAN CORPUS DATA.
    # CLAUDE.md's condition 5 keeps quantised generators out of a corpus. Nothing here is
    # kept as data: the question is whether the pipeline accepts a longer image list and
    # what that does to the silhouette, and the answer to both is the same at any precision.
    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=DiffusersBnb(**nf4))
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, subfolder="mllm", torch_dtype=torch.bfloat16,
        quantization_config=TransformersBnb(**nf4))
    # trust_remote_code, and what is trusted: the scheduler shipped in the weights repo,
    # which `omnigen2_edit.py` records as 228 lines importing only math, numpy, torch and
    # diffusers. The probe failed without it: diffusers refuses custom code in
    # scheduling_flow_match_euler_discrete.py rather than executing it silently.
    pipe = OmniGen2Pipeline.from_pretrained(
        args.model, transformer=transformer, mllm=mllm, torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    pipe.vae.to("cuda")

    src = Image.open(frame).convert("RGB")
    dep = Image.open(depth).convert("RGB")
    pos = Image.open(pose).convert("RGB")
    reference = body_mask(src)

    conditions = {
        "A_render_only": [src],
        "B_render_depth": [src, dep],
        "C_render_depth_pose": [src, dep, pos],
    }
    record = {"frame": str(frame), "revision": resolved,
              "steps": args.steps, "seed": args.seed,
              "text_guidance": args.text_guidance, "image_guidance": args.image_guidance,
              "prompt": PROMPT, "conditions": {}}

    for name, images in conditions.items():
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            res = pipe(prompt=PROMPT, input_images=images, width=1024, height=1024,
                       num_inference_steps=args.steps, max_sequence_length=1024,
                       text_guidance_scale=args.text_guidance,
                       image_guidance_scale=args.image_guidance,
                       negative_prompt=NEGATIVE, num_images_per_prompt=1,
                       generator=torch.Generator(device="cuda").manual_seed(args.seed),
                       output_type="pil")
        except Exception as error:  # noqa: BLE001
            record["conditions"][name] = {"images": len(images), "accepted": False,
                                          "error": f"{type(error).__name__}: {error}"[:300]}
            print(f"CONDITION {name}: {len(images)} image(s) REFUSED "
                  f"{type(error).__name__}: {str(error)[:120]}", flush=True)
            continue

        out = pathlib.Path(args.out) / f"{name}.png"
        res.images[0].save(out)
        agreement = iou(reference, body_mask(res.images[0]))
        record["conditions"][name] = {
            "images": len(images), "accepted": True, "file": out.name,
            "silhouette_iou": round(agreement, 4),
            "silhouette_error": round(1.0 - agreement, 4),
            "seconds": round(time.time() - started, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        }
        print(f"CONDITION {name}: {len(images)} image(s) accepted, "
              f"silhouette_error {1 - agreement:.4f}, "
              f"{record['conditions'][name]['seconds']}s, "
              f"{record['conditions'][name]['peak_vram_gib']} GiB", flush=True)

    with open(pathlib.Path(args.out) / "array_probe.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
