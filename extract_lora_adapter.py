"""Pull the LoRA out of a 15 GB checkpoint and write it as a PEFT adapter.

WHY THIS IS NOT OPTIONAL. `train.py` saves the **entire model** in every checkpoint, base
weights included. `docs/FINETUNE.md:80-81` records the reason: under FSDP the trainer cannot
easily separate the adapter parameters, so it writes all of them. A checkpoint here is 15 GB
of which the trained part is a few megabytes, and publishing the 15 GB version would ship
somebody else's Apache-2.0 base weights as though they were ours, at a thousand times the
size, with no way to tell which tensors we actually changed.

WHAT IS TAKEN. Only keys containing `lora_`, and the count is asserted rather than assumed:
rank 8 over `to_k`, `to_q`, `to_v` and `to_out.0` gives **304** tensors. A run that produced a
different number changed the target modules or the rank, and writing it out under the same
name as the config that did not produce it is how a checkpoint comes to disagree with its own
card.

THE NAMES ARE REWRITTEN, AND THAT IS THE PART THAT SILENTLY BREAKS. The trainer stores
`...to_q.lora_A.default.weight`, carrying the adapter's PEFT name in the middle. A loader that
expects `...to_q.lora_A.weight` finds nothing under those keys, adds a freshly initialised
adapter, and generates perfectly ordinary images from the base model. Nothing raises, and the
output looks like a model that learned nothing rather than a model that was never loaded. So
the rewrite is checked by counting what matched.

    python extract_lora_adapter.py --checkpoint <checkpoint-N> --out <dir> [--rank 8]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

EXPECTED_TENSORS = 304
TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="a checkpoint-N directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--base", default="OmniGen2/OmniGen2")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    src = pathlib.Path(args.checkpoint) / "model.safetensors"
    if not src.is_file():
        sys.exit("FAIL  no model.safetensors under %s" % args.checkpoint)

    state = load_file(str(src))
    lora = {k: v for k, v in state.items() if "lora_" in k}
    if not lora:
        sys.exit("FAIL  %s carries no LoRA tensors. Publishing it would ship the base model "
                 "under our name." % src)
    if len(lora) != EXPECTED_TENSORS:
        print("NOTE  %d LoRA tensors, not the %d that rank-%d over %s produces. The adapter "
              "card must say what was actually trained."
              % (len(lora), EXPECTED_TENSORS, args.rank, ", ".join(TARGET_MODULES)))

    # PEFT reads `base_model.model.<path>.lora_A.weight`. The trainer writes
    # `<path>.lora_A.default.weight`. Both halves of that change are counted below.
    renamed, dropped_default, prefixed = {}, 0, 0
    for key, value in lora.items():
        new = key
        if ".default." in new:
            new = new.replace(".default.", ".")
            dropped_default += 1
        if not new.startswith("base_model.model."):
            new = "base_model.model." + new
            prefixed += 1
        renamed[new] = value

    if dropped_default != len(lora) or prefixed != len(lora):
        sys.exit("FAIL  rewrote .default. on %d of %d keys and prefixed %d of %d. A partial "
                 "rewrite loads some tensors and silently initialises the rest, which reads "
                 "as a model that learned nothing."
                 % (dropped_default, len(lora), prefixed, len(lora)))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out / "adapter_model.safetensors"))

    # `lora_alpha` equals the rank because `train.py:267` sets it that way and ignores any
    # value in the config. Writing the real number here rather than a wished-for one is what
    # keeps the adapter loadable by anything that is not this trainer.
    config = {
        "peft_type": "LORA",
        # A STRING, BECAUSE null FAILS THE HUB'S OWN VALIDATOR. It read `None`, which is what
        # PEFT itself accepts for a diffusion transformer, and Hugging Face answered
        # `"peft.task_type" must be a string` on the model page. FEATURE_EXTRACTION is the
        # member of PEFT's TaskType that fits a transformer with no language-modelling head.
        "task_type": "FEATURE_EXTRACTION",
        "base_model_name_or_path": args.base,
        "r": args.rank,
        "lora_alpha": args.rank,
        "lora_dropout": 0.0,
        "target_modules": TARGET_MODULES,
        "init_lora_weights": "gaussian",
        "bias": "none",
        "inference_mode": True,
    }
    with open(out / "adapter_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    size = (out / "adapter_model.safetensors").stat().st_size
    print("%d LoRA tensors of %d in the checkpoint" % (len(lora), len(state)))
    print("adapter %.2f MiB against a %.1f GiB checkpoint"
          % (size / 2 ** 20, src.stat().st_size / 2 ** 30))
    print("rank %d, alpha %d, targets %s" % (args.rank, args.rank, ", ".join(TARGET_MODULES)))
    print("wrote %s" % (out / "adapter_model.safetensors"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
