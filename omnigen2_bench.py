"""Which OmniGen2 settings buy time, and what do they cost?

THE SWEEP IS HUNDREDS OF NEAR-IDENTICAL CALLS, so seconds-per-view is the number that gets
multiplied and every other choice is downstream of it. Nothing in this workspace has ever
passed an optimisation flag: `omnigen2_edit.py`, `omnigen2_array_probe.py` and
`gen_posed_from_reference.py` all call the pipeline with the same reduced argument set and
never touch `cfg_range`, TeaCache, TaylorSeer, the scheduler, or offload. This measures them.

BASELINE, MEASURED ON THIS DESK RATHER THAN QUOTED. logbook-fourloops-first-runs.md records
bf16 at 1024 square and 30 steps: 17.14 GiB peak, 131 s. That is 4.37 s a step, and it is the
row every other row is read against. A number without a baseline is not a measurement.

WHY bf16 AND NOT nf4. Four bits bought memory and not speed here: 133 s against 131 s,
because dequantisation costs about what the narrower reads save on this card. It also bars the
output from a corpus under CLAUDE.md's generated-synthetic condition 5. A sweep run at nf4
would be evidence that can never become data, so the benchmark measures the precision the
sweep will actually use.

TWO COSTS HERE, A THIRD ELSEWHERE. A setting that halves the time and moves the body has not
helped, so each row writes its own frame and the frames are scored afterwards, by joint drift
against the source render and by EditScore. This file measures time and memory; it does not
judge the images, because a generator scoring its own output is not a measurement.

    python omnigen2_bench.py --repo C:\\omnigen2-src --frame <render.png> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch
from PIL import Image

PROMPT = ("Turn this into a photograph of a real person in the same pose, natural skin and "
          "fabric, studio lighting, plain dark background. Do not move the body.")
NEGATIVE = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn "
            "face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused "
            "fingers, messy drawing, broken legs censor, censored, censor_bar")

# EACH ROW CHANGES ONE THING FROM THE ROW ABOVE IT, so a difference has one candidate cause.
# `max_sequence_length` goes first because it is free: the pipeline default is 256, all three
# of our scripts hardcode 1024, and the prompt above is about 30 tokens. The rest are ordered
# cheapest-risk first, and `num_images_per_prompt` is late because its benefit is per-image
# rather than per-call, so it only helps a sweep that wants more than one sample per view.
#
# TAYLORSEER IS LAST, AND THE REASON IS A MEASUREMENT. Upstream claims up to 2x from it. On
# this card at 1024 square it went the other way: the three rows before it finished in 131 s,
# 131 s and 103 s, and TaylorSeer was killed after 8 minutes still running -- at least 4.5x
# the row it followed, not half. The card sat at 24.2 GiB of 24.5 the whole time, so the cache
# tensors it adds tipped an already-full allocation into shared system memory, where a Windows
# WDDM driver pages instead of failing. That is the failure mode to watch for: it does not
# raise, it just stops being fast, and a benchmark left unattended would have recorded it as a
# very slow row rather than as a row that did not fit.
#
# It stays in the table because "does not fit at 1024 square on 24 GiB" is the useful finding,
# and it runs last so a kill costs nothing. At a smaller size it may well deliver what upstream
# claims; that is a separate row, not this one.
ROWS = [
    dict(label="baseline_msl1024", steps=30, msl=1024, nimg=1),
    dict(label="msl256", steps=30, msl=256, nimg=1),
    dict(label="msl256_cfg0.6", steps=30, msl=256, nimg=1, cfg_range=(0.0, 0.6)),
    dict(label="msl256_cfg0.6_16steps", steps=16, msl=256, nimg=1, cfg_range=(0.0, 0.6)),
    dict(label="msl256_cfg0.6_teacache", steps=30, msl=256, nimg=1, cfg_range=(0.0, 0.6),
         teacache=0.05),
    dict(label="msl256_cfg0.6_batch2", steps=30, msl=256, nimg=2, cfg_range=(0.0, 0.6)),
    dict(label="msl256_cfg0.6_taylor", steps=30, msl=256, nimg=1, cfg_range=(0.0, 0.6),
         taylorseer=True),
]


def gib(n):
    return n / 2 ** 30


def write_record(record, out_dir):
    """Persist what has been measured so far. Called after every row, not once at the end."""
    path = pathlib.Path(out_dir) / "omnigen2_bench.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


def load(repo, model):
    """Load once. The precedent run paid 75.5 s of load before its first pixel, and a sweep
    that reloads per view pays it every time."""
    sys.path.insert(0, repo)
    from huggingface_hub import hf_hub_download

    # model_index.json names its custom classes by bare module name and diffusers resolves
    # those with a plain importlib call, so the weights' own directories go on the path.
    # `omnigen2_edit.py` carries the same workaround and the same reason.
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

    t0 = time.time()
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=torch.bfloat16)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model, subfolder="mllm", torch_dtype=torch.bfloat16)
    pipe = OmniGen2Pipeline.from_pretrained(
        model, transformer=transformer, mllm=mllm, torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    pipe.to("cuda")
    return pipe, resolved, time.time() - t0


def reset_accelerators(pipe):
    """TaylorSeer and TeaCache are attributes rather than call arguments, so a row that sets
    one leaves it set for every row after it. Clearing them here is what keeps the rows
    independent. Without it the table reads as a steady speedup that is really one setting
    never being turned off."""
    pipe.enable_taylorseer = False
    if hasattr(pipe.transformer, "enable_teacache"):
        pipe.transformer.enable_teacache = False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=r"C:\omnigen2-src")
    ap.add_argument("--model", default="OmniGen2/OmniGen2")
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="run one row by label")
    args = ap.parse_args()

    frame = pathlib.Path(args.frame)
    if not frame.is_file():
        sys.exit("FAIL  %s is missing" % frame)
    os.makedirs(args.out, exist_ok=True)

    pipe, resolved, load_s = load(args.repo, args.model)
    if resolved is None:
        sys.exit("FAIL  could not resolve a commit from the cache path; a row that cannot "
                 "name the weights that produced it is not a measurement")
    print("loaded in %.0fs | weights on device %.2f GiB"
          % (load_s, gib(torch.cuda.memory_allocated())), flush=True)

    src = Image.open(frame).convert("RGB")
    record = {"frame": str(frame), "revision": resolved, "precision": "bf16",
              "size": args.size, "seed": args.seed, "load_seconds": round(load_s, 1),
              "torch": torch.__version__, "prompt": PROMPT, "rows": []}

    rows = [r for r in ROWS if not args.only or r["label"] == args.only]
    for row in rows:
        reset_accelerators(pipe)
        if row.get("taylorseer"):
            pipe.enable_taylorseer = True
        if row.get("teacache"):
            pipe.transformer.enable_teacache = True
            pipe.transformer.teacache_rel_l1_thresh = row["teacache"]

        kwargs = dict(prompt=PROMPT, input_images=[src], width=args.size, height=args.size,
                      num_inference_steps=row["steps"], max_sequence_length=row["msl"],
                      text_guidance_scale=5.0, image_guidance_scale=2.0,
                      negative_prompt=NEGATIVE, num_images_per_prompt=row["nimg"],
                      generator=torch.Generator(device="cuda").manual_seed(args.seed),
                      output_type="pil")
        if "cfg_range" in row:
            kwargs["cfg_range"] = row["cfg_range"]

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            res = pipe(**kwargs)
        except Exception as error:  # noqa: BLE001
            # A ROW THAT FAILED IS NOT A ROW THAT WAS SLOW. Recording the error keeps it in
            # the table rather than leaving a gap that reads as "not tried".
            record["rows"].append(dict(row, ok=False,
                                       error=("%s: %s" % (type(error).__name__, error))[:300]))
            print("  BAD  %s: %s: %s" % (row["label"], type(error).__name__, str(error)[:140]),
                  flush=True)
            continue

        elapsed = time.time() - started
        peak = gib(torch.cuda.max_memory_allocated())
        files = []
        for i, image in enumerate(res.images):
            dst = pathlib.Path(args.out) / ("%s_%d.png" % (row["label"], i))
            image.save(dst)
            files.append(dst.name)
        per_image = elapsed / max(len(res.images), 1)
        record["rows"].append(dict(row, ok=True, files=files,
                                   seconds=round(elapsed, 1),
                                   seconds_per_image=round(per_image, 1),
                                   seconds_per_step=round(elapsed / row["steps"], 2),
                                   peak_vram_gib=round(peak, 2)))
        print("  ok   %-26s %6.1fs  %6.1fs/image  %5.2fs/step  peak %5.2f GiB"
              % (row["label"], elapsed, per_image, elapsed / row["steps"], peak), flush=True)
        # AFTER EVERY ROW, NOT AT THE END. The first run of this file wrote the JSON once, on
        # the way out, and then row four ran 4.5 times longer than the row before it because
        # TaylorSeer's caches tipped a 24 GiB card into shared memory. Killing it would have
        # thrown away three completed measurements to escape a fourth. A benchmark that only
        # persists if it finishes is a benchmark that punishes you for stopping it.
        write_record(record, args.out)

    out = write_record(record, args.out)

    good = [r for r in record["rows"] if r.get("ok")]
    if good:
        base = next((r for r in good if r["label"] == "baseline_msl1024"), None)
        print("\nlabel                        s/image   vs baseline   peak GiB")
        for r in good:
            ratio = ("%.2fx" % (r["seconds_per_image"] / base["seconds_per_image"])
                     if base else "--")
            print("  %-26s %6.1f   %10s   %6.2f"
                  % (r["label"], r["seconds_per_image"], ratio, r["peak_vram_gib"]))
    failed = [r for r in record["rows"] if not r.get("ok")]
    print("\n%d row(s) ran, %d failed" % (len(good), len(failed)))
    print("written %s" % out)
    print("NOT MEASURED HERE: whether the body moved. Score these frames with "
          "test_pose_survives_restyle.py and score_edits.py before adopting a row.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
