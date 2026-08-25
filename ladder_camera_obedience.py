"""Does OmniGen2 turn the body when you ask for a different camera, and does depth help?

THE QUESTION THE WHOLE SWEEP RESTS ON. The plan is to generate 96 named views of one subject
and recover a body from them. That is worth hours only if asking for "right side view" produces
a right side view. It might not, and there is a specific reason to doubt it: the phrase table
comes from fal's Multiple-Angles LoRA, which was trained on 3000+ Gaussian-splat renders --
for Qwen-Image-Edit, a model this project blocklists. OmniGen2 was never conditioned on
`<sks> right side view eye-level shot medium shot`. For OmniGen2 those words are a label, not
a control, and whether plain-language camera instruction moves the body is unmeasured.

So: eight azimuths, and the answer is a plot rather than an impression. Requested azimuth
against recovered azimuth. A flat line means the sweep would be 96 copies of one pose, and
that finding costs forty minutes instead of a day.

THREE CONDITIONS, WHICH ALSO REPLACES AN INVALID PROBE. `omnigen2_array_probe.py` passed
`[render, depth]` under the prompt "Restyle this character as a watercolour painting", which
never refers to a second image at all. The model was handed a depth map and never told it was
there, and the conclusion drawn -- that OmniGen2 has no depth control -- was drawn from a test
that could not have shown one. Upstream's README gives the required form, "the [object] from
the second image", and `gen_posed_from_reference.py` already uses it correctly.

  A  [styled]              camera instruction only
  B  [styled, depth]       and the depth of the target view, named as the second image
  C  [styled, depth, pose] and the pose skeleton, named as the third

Condition C costs resolution rather than adding to it, and that is worth knowing before the
sweep: `omnigen2_train_dataset.py` picks the pixel budget by the NUMBER of inputs, not by
position, so three inputs caps all three at 768 square where one gets 1024.

WHAT THIS DOES NOT DO. It does not judge the images. Recovered azimuth comes from fitting
ANNY to detected keypoints, which is arithmetic on the body rather than an opinion about the
picture, and EditScore runs separately afterwards.

    python ladder_camera_obedience.py --frame <front-view.png> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "7-service" / "service-livebook" / "priv" / "python"))

NEGATIVE = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn "
            "face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused "
            "fingers, messy drawing, broken legs censor, censored, censor_bar")

# THE FASTEST SETTING THAT WAS MEASURED, NOT THE DEFAULT. bf16 at 1024 square and 30 steps is
# 131 s on this card; cfg_range (0.0, 0.6) brings it to 103 s because outside the range both
# guidance scales drop to 1.0 and the pipeline runs one transformer pass per step instead of
# three. Upstream's own RL config uses the same value. TaylorSeer is deliberately absent: it
# does not fit at this size here, and it pages rather than failing.
STEPS = 30
CFG_RANGE = (0.0, 0.6)
MAX_SEQ = 256


def condition_prompt(view_phrase, condition):
    """The camera instruction, and the images named in the order the pipeline receives them."""
    base = ("Show this same person from a different camera angle: %s. Keep the person, their "
            "clothing, their body proportions and the setting exactly the same. Change only "
            "the camera." % view_phrase)
    if condition == "A":
        return base
    if condition == "B":
        return base + (" The second image is a depth map of the view you are generating: "
                       "match the body position and the camera in that depth map exactly.")
    return base + (" The second image is a depth map of the view you are generating and the "
                   "third image is its pose skeleton: match the body position, the limb "
                   "angles and the camera in those two images exactly.")


def load_pipeline(repo, model, lora_checkpoint="", lora_rank=8):
    sys.path.insert(0, repo)
    from huggingface_hub import hf_hub_download

    resolved = None
    for rel in ("transformer/transformer_omnigen2.py",
                "scheduler/scheduling_flow_match_euler_discrete.py"):
        local = hf_hub_download(model, rel)
        sys.path.insert(0, os.path.dirname(local))
        parts = os.path.normpath(local).split(os.sep)
        if "snapshots" in parts:
            resolved = parts[parts.index("snapshots") + 1]

    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    transformer = OmniGen2Transformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=torch.bfloat16)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model, subfolder="mllm", torch_dtype=torch.bfloat16)
    if lora_checkpoint:
        # THE ADAPTER IS ADDED BEFORE THE WEIGHTS ARE LOADED, and the target modules and rank
        # must match training exactly -- `train.py:262` hardcodes these four and sets
        # `lora_alpha=lora_rank`, so a mismatch here loads shapes that do not exist and the
        # run would fail rather than quietly evaluate the base model. Failing is the point.
        from peft import LoraConfig
        from safetensors.torch import load_file

        transformer.add_adapter(LoraConfig(
            r=lora_rank, lora_alpha=lora_rank, lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        state = load_file(os.path.join(lora_checkpoint, "model.safetensors"))
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        lora_keys = [k for k in state if "lora" in k.lower()]
        if not lora_keys:
            sys.exit("FAIL  %s carries no LoRA tensors, so loading it would measure the base "
                     "model while claiming to measure the adapter" % lora_checkpoint)
        print("adapter: %d LoRA tensors loaded, %d missing, %d unexpected"
              % (len(lora_keys), len(missing), len(unexpected)), flush=True)

    pipe = OmniGen2Pipeline.from_pretrained(
        model, transformer=transformer, mllm=mllm, torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    pipe.to("cuda")
    return pipe, resolved


def card_is_free(limit_mib=2000):
    """Two bf16 OmniGen2 pipelines want 29.5 GiB on a 24 GiB card, and Windows pages instead
    of failing, so neither process errors and both run about fifteen times slower. That is not
    a hypothetical: it happened, and the only symptom was a benchmark row taking 27 minutes
    instead of 131 seconds. Refusing to start is cheaper than diagnosing it again."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        used = int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return True, -1
    return used <= limit_mib, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=r"C:\omnigen2-src")
    ap.add_argument("--model", default="OmniGen2/OmniGen2")
    ap.add_argument("--frame", required=True, help="the source view, with .depth.png beside it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="A,B,C")
    # AZIMUTH AND CONDITION ARE SEPARABLE, AND SPLITTING THEM IS WHAT MADE THIS
    # AFFORDABLE. "Does the body turn?" needs many azimuths at one condition; "does
    # conditioning help?" needs many conditions at one azimuth. Measured here: A is 105 s,
    # B is 182 s, C is 529 s, so the full 8 x 3 factorial costs about 3.6 hours to answer
    # two questions that between them need 8 + 3 runs, not 24.
    ap.add_argument("--azimuths", default="",
                    help="comma-separated degrees; every azimuth in the vocabulary if omitted")
    # THE POINT OF THE WHOLE EXERCISE. The unmodified model scored a slope of 0.04 against
    # the requested azimuth -- a body that did not turn. This loads a trained adapter and runs
    # the identical eight prompts, so the only thing that changed between the two tables is
    # the weights. `checkpoint-N/model.safetensors` holds base and adapter together, 886
    # tensors of which 304 are LoRA, because the trainer saves the whole model under FSDP.
    ap.add_argument("--lora-checkpoint", default="",
                    help="a checkpoint-N directory from train.py; the base model if omitted")
    ap.add_argument("--lora-rank", type=int, default=8,
                    help="must match training; alpha equals rank in this trainer")
    ap.add_argument("--force", action="store_true", help="start even if the card is busy")
    args = ap.parse_args()

    from weft_loop import AZIMUTHS, camera_prompt

    frame = pathlib.Path(args.frame)
    depth = frame.with_name(frame.stem + ".depth.png")
    pose = frame.with_name(frame.stem + ".pose.png")
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    need = {"B": [depth], "C": [depth, pose]}
    for c in conditions:
        for p in need.get(c, []):
            if not p.is_file():
                sys.exit("FAIL  condition %s needs %s, which is missing; run make_controls.py"
                         % (c, p.name))
    os.makedirs(args.out, exist_ok=True)

    free, used = card_is_free()
    if not free and not args.force:
        sys.exit("FAIL  the GPU already holds %d MiB. Two bf16 pipelines do not fit and "
                 "Windows pages rather than failing, so both runs crawl instead of erroring. "
                 "Wait, or pass --force if you know what else is resident." % used)

    src = Image.open(frame).convert("RGB")
    dep = Image.open(depth).convert("RGB") if depth.is_file() else None
    pos = Image.open(pose).convert("RGB") if pose.is_file() else None

    pipe, resolved = load_pipeline(args.repo, args.model, args.lora_checkpoint, args.lora_rank)
    if resolved is None:
        sys.exit("FAIL  could not resolve a commit for %s from the cache path" % args.model)
    print("loaded, weights %.2f GiB, revision %s"
          % (torch.cuda.memory_allocated() / 2 ** 30, resolved), flush=True)

    record = {"source_frame": str(frame), "revision": resolved, "precision": "bf16",
              "steps": args.steps, "size": args.size, "seed": args.seed,
              "cfg_range": list(CFG_RANGE), "max_sequence_length": MAX_SEQ,
              "elevation_deg": args.elevation, "distance_factor": args.distance,
              "views": []}
    out_json = pathlib.Path(args.out) / "ladder.json"

    images_for = {"A": [src], "B": [src, dep], "C": [src, dep, pos]}
    wanted = ([float(a) for a in args.azimuths.split(",") if a.strip()]
              if args.azimuths else [float(a) for a, _ in AZIMUTHS])
    unknown = [a for a in wanted if a not in [float(x) for x, _ in AZIMUTHS]]
    if unknown:
        # THE VOCABULARY RAISES RATHER THAN SNAPPING, and so does this. `weft_loop._exact`
        # refuses to round a camera into a neighbouring sector because a phrase that names
        # a view the render does not show is a mislabelled corpus row.
        sys.exit("FAIL  %s are not azimuths in the vocabulary; it has %s"
                 % (unknown, [a for a, _ in AZIMUTHS]))
    for azimuth in wanted:
        view_phrase = camera_prompt(azimuth, args.elevation, args.distance)
        for condition in conditions:
            images = images_for[condition]
            prompt = condition_prompt(view_phrase, condition)
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
            try:
                res = pipe(prompt=prompt, input_images=images,
                           width=args.size, height=args.size,
                           num_inference_steps=args.steps, max_sequence_length=MAX_SEQ,
                           cfg_range=CFG_RANGE, text_guidance_scale=5.0,
                           image_guidance_scale=2.0, negative_prompt=NEGATIVE,
                           num_images_per_prompt=1,
                           generator=torch.Generator(device="cuda").manual_seed(args.seed),
                           output_type="pil")
            except Exception as error:  # noqa: BLE001
                record["views"].append({"azimuth_deg": azimuth, "condition": condition,
                                        "ok": False,
                                        "error": ("%s: %s" % (type(error).__name__, error))[:300]})
                print("  BAD  az %3d %s: %s" % (int(azimuth), condition, str(error)[:120]), flush=True)
                with open(out_json, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2)
                continue

            dst = pathlib.Path(args.out) / ("az%03d_%s.png" % (int(azimuth), condition))
            res.images[0].save(dst)
            elapsed = time.time() - started
            record["views"].append({
                "azimuth_deg": azimuth, "condition": condition, "ok": True,
                "view_phrase": view_phrase, "prompt": prompt, "file": dst.name,
                "n_input_images": len(images),
                "seconds": round(elapsed, 1),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            })
            print("  ok   az %3d %s  %5.1fs  %s" % (int(azimuth), condition, elapsed, dst.name),
                  flush=True)
            # AFTER EVERY VIEW. A long run that only persists on success discards everything
            # when it is stopped, which is how three good benchmark rows were nearly lost.
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)

    ok = [v for v in record["views"] if v.get("ok")]
    print("\n%d of %d generated, wrote %s" % (len(ok), len(record["views"]), out_json))
    print("NOT ANSWERED YET: whether the body turned. Run fit_ladder_azimuth.py over this "
          "directory to recover an azimuth per image and compare it against the one asked for.")
    return 0 if len(ok) == len(record["views"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
