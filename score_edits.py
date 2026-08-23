"""Does EditScore actually refuse our renders, and does it fit 8 GB at four bits?

TWO QUESTIONS, AND THE FIRST IS THE ONE NOBODY HAS ANSWERED. Uncensored bases were being lined
up for this job on the assumption that a grader would decline to look at an unclothed figure --
and our ANNY renders are unclothed bodies. Nothing measured that. Refusal tuning targets
requests to PRODUCE content, and scoring an edit is grading, so the assumption may simply be
false. This counts refusals on the frames in question instead of arguing about it.

THE SECOND IS DEVICE SIZING. The ASUS UGen300 is a Hailo-10H with 8 GB. Qwen3-VL-8B is 16.3 GB
at bf16 and about 4.5 GB at NF4, so four bits is the only version that could land there. Fitting
in 8 GB of a desktop card's VRAM is necessary and not sufficient -- the graph still has to
compile through the Dataflow Compiler, and Hailo's GenAI zoo ships no such model -- so read this
as a size measurement.

A QUANTISED VERIFIER IS PERMITTED WHERE A QUANTISED GENERATOR IS NOT. CLAUDE.md's condition 5
binds generators, because a quantised generator writes labels that are lies. A verifier emits a
number about somebody else's frame. But it can still be wrong, and a grader that passes bad
frames is worse than no grader, so --precision bf16 exists here to compare against.

WHY THE LOADER IS PATCHED. editscore's Qwen3VL backbone loads bf16 and then calls
`merge_and_unload()`, which cannot fold a LoRA into 4-bit base weights. At nf4 the adapter stays
attached instead of merged -- numerically the same forward pass, one indirection slower.
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from PIL import Image

BASE = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER = "EditScore/EditScore-Qwen3-VL-8B-Instruct"

# What each restyle was asked to do. The scorer needs the instruction, not just the pair.
INSTRUCTIONS = {
    "photographic": "Turn this into a photograph of a real person in the same pose.",
    "colour-sketch": "Turn this into a coloured pencil sketch of the same figure in the same pose.",
    "ukiyoe": "Restyle this as a ukiyo-e woodblock print, same pose.",
    "monet": "Restyle this in the style of Monet, same pose.",
    "corrupt": "Degrade this image with sensor noise, defocus and heavy compression.",
}


def instruction_for(name):
    for key, text in INSTRUCTIONS.items():
        if key in name.lower():
            return text
    return None


def patch_for_4bit():
    """Inject a quantisation config and stop the merge that 4-bit cannot do."""
    from transformers import BitsAndBytesConfig
    import editscore.mllm_tools.qwen3vl as q3

    real_from_pretrained = q3.Qwen3VLForConditionalGeneration.from_pretrained

    def quantised(model_id, **kw):
        kw.pop("torch_dtype", None)
        return real_from_pretrained(
            model_id, dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            **{k: v for k, v in kw.items() if k != "quantization_config"})

    q3.Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(quantised)
    real_peft = q3.PeftModel.from_pretrained

    def attach(model, path, **kw):
        m = real_peft(model, path, **kw)
        m.merge_and_unload = lambda *a, **k: m      # keep the adapter attached
        return m

    q3.PeftModel.from_pretrained = staticmethod(attach)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--source", required=True, help="the render every restyle was made from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    ap.add_argument("--num-pass", type=int, default=1)
    # 262144 is 512x512, which is `image_max_pixels` in EditScore's own training config. Our
    # renders are 1024x1024, four times the pixels and therefore four times the vision tokens --
    # which is where the peak goes when weights are only 6.3 GiB and the peak is 8.6.
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="downscale each image to at most this many pixels (0 = leave alone)")
    args = ap.parse_args()

    if args.precision == "nf4":
        patch_for_4bit()

    from editscore import EditScore
    t0 = time.time()
    scorer = EditScore(backbone="qwen3vl", model_name_or_path=BASE, lora_path=ADAPTER,
                       score_range=25, num_pass=args.num_pass)
    load_s = time.time() - t0
    weights = torch.cuda.memory_allocated() / 2**30
    print(f"loaded in {load_s:.0f}s | {args.precision} | weights {weights:.2f} GiB")

    def cap(im):
        if not args.max_pixels or im.width * im.height <= args.max_pixels:
            return im
        s = (args.max_pixels / (im.width * im.height)) ** 0.5
        return im.resize((int(im.width * s), int(im.height * s)), Image.BICUBIC)

    src = cap(Image.open(args.source).convert("RGB"))
    rows, refusals = [], 0
    for f in sorted(glob.glob(os.path.join(args.dir, "*.png"))):
        name = os.path.basename(f)
        if "contact_sheet" in name or name == os.path.basename(args.source):
            continue
        instruction = instruction_for(name)
        if instruction is None:
            continue
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        try:
            res = scorer.evaluate([src, cap(Image.open(f).convert("RGB"))], instruction)
            overall = res.get("overall")
            raw = json.dumps(res)[:400]
            # A refusal is a non-answer, not a low score: no number came back at all.
            refused = overall is None or not isinstance(overall, (int, float))
        except Exception as exc:
            overall, raw, refused = None, f"{type(exc).__name__}: {exc}", True
        refusals += bool(refused)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"file": name, "instruction": instruction, "overall": overall,
                     "refused": bool(refused), "seconds": round(time.time() - t1, 1),
                     "peak_vram_gib": round(peak, 2), "raw": raw})
        print(f"  {name:38} score={str(overall):>6}  refused={str(refused):5} "
              f"{time.time()-t1:5.1f}s  peak {peak:.2f} GiB")

    out = {"base": BASE, "adapter": ADAPTER, "precision": args.precision,
           "max_pixels": args.max_pixels,
           "num_pass": args.num_pass, "weights_gib": round(weights, 2),
           "scored": len(rows), "refusals": refusals, "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{refusals} refusal(s) of {len(rows)} frames -> {args.out}")


if __name__ == "__main__":
    main()
