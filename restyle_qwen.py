"""RFD 107a rule 2: the two Qwen-Image-Edit appearances, quantised to 4 bits.

WHY 4-BIT AND NOT bf16. Qwen-Image-Edit-2511 is 57.7 GB of weights against a 24 GB card, so
bf16 does not fit and no amount of offloading makes it fast. NF4 puts the 20B transformer at
roughly a quarter of that.

WHY THIS RUN IS A RETEST RATHER THAN A FIRST TRY. `interactor-pixal3d`'s README records bnb
4-bit Qwen-Image producing camera-correct but noise-corrupted images across every software
combination tried -- diffusers 0.35/0.36, model 2508/2511, cfg and Lightning -- with torch
2.4.1 the one constant, and bnb warning of a misaligned inner dimension (3420 % 64 != 0). It
says to retest on torch >= 2.6 or with 8-bit before relying on it. The environment this runs
in is that retest, which is why the manifest bounds torch rather than pinning it.

PROVENANCE, because a generated corpus needs it. The model id and the resolved commit are
written next to every output, so the corpus can say what made it. Nothing here is training
data yet -- this is one frame, to see whether the path works at all.
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

REPO = "Qwen/Qwen-Image-Edit-2511"

# Rule 2's two appearances from this model. Kept as data so the prompts are part of the
# record rather than typed at a shell.
PROMPTS = {
    "photographic": (
        "Turn this into a photograph of a real person in the same pose, natural skin and "
        "fabric, studio lighting, plain dark background. Do not move the body."
    ),
    "colour-sketch": (
        "Turn this into a coloured pencil sketch of the same figure in the same pose, visible "
        "strokes, paper texture, plain background. Do not move the body."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bits", choices=["4", "8"], default="4")
    # An empty string, not None. diffusers only enables classifier-free guidance when a
    # negative prompt is present, so the first run passed true_cfg_scale=4.0 and silently ran
    # with guidance off -- it says so in the log, and a scale that does nothing is worse than
    # no scale at all.
    ap.add_argument("--negative", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import BitsAndBytesConfig as TransformersBnb
    from transformers import Qwen2_5_VLForConditionalGeneration

    kw = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16) if args.bits == "4" else dict(load_in_8bit=True)

    t0 = time.time()
    transformer = QwenImageTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", quantization_config=DiffusersBnb(**kw),
        torch_dtype=torch.bfloat16)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPO, subfolder="text_encoder", quantization_config=TransformersBnb(**kw),
        torch_dtype=torch.bfloat16)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f}s, {args.bits}-bit")

    src = Image.open(args.image).convert("RGB")
    record = {"model": REPO, "bits": int(args.bits), "steps": args.steps,
              "true_cfg_scale": args.cfg, "seed": args.seed, "negative_prompt": args.negative,
              "torch": torch.__version__, "outputs": {}}
    try:                                    # the commit is the half of provenance that pins it
        from huggingface_hub import HfApi
        record["revision"] = HfApi().model_info(REPO).sha
    except Exception as exc:
        record["revision"] = f"unresolved: {type(exc).__name__}"

    for name, prompt in PROMPTS.items():
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        out = pipe(image=[src], prompt=prompt, num_inference_steps=args.steps,
                   true_cfg_scale=args.cfg, negative_prompt=args.negative,
                   generator=torch.Generator("cuda").manual_seed(args.seed)).images[0]
        dst = os.path.join(args.out, f"qwen_{name}_{args.bits}bit{args.tag}.png")
        out.save(dst)
        peak = torch.cuda.max_memory_allocated() / 2**30
        record["outputs"][name] = {"file": os.path.basename(dst), "prompt": prompt,
                                   "seconds": round(time.time() - t1, 1),
                                   "peak_vram_gib": round(peak, 2)}
        print(f"  ok   {dst}  {time.time()-t1:.0f}s  peak {peak:.1f} GiB")

    with open(os.path.join(args.out, f"qwen_provenance_{args.bits}bit{args.tag}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    print("provenance written")


if __name__ == "__main__":
    main()
