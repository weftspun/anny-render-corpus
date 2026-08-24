"""ANNY supplies the pose, a COCO photograph supplies the appearance, the prompt binds them.

THE ALGORITHM, STATED BECAUSE AN EARLIER VERSION OF THIS TEST GOT IT WRONG. It is not a
restyle of the render. OmniGen2 takes a LIST of input images, and the corpus route uses two:

    input_images = [anny_render, coco_reference]

The render carries the geometry whose 104 joints are known by construction; the photograph
carries skin, fabric and lighting that no render has; the prompt says to keep the first and
borrow the second. A single-image restyle asks the model to invent an appearance, which is a
different and easier question than the one the corpus actually needs answered.

WHICH MAKES align_res LOAD-BEARING RATHER THAN INCIDENTAL. Upstream aligns the output
resolution to the input only when there is exactly one input image:

    if len(_input_images) == 1 and align_res:

Every call here passes two, so that path is never taken, and a keypoint label is only true of
an image that registers with the render it came from. The output size is therefore passed
explicitly rather than inherited.

TRAIN2017 ONLY, ASSERTED NOT ASSUMED. The reference photographs come from
`coco_images_train2017`, and the 523 blinded val2017 ids are checked against every one before
it is used. CLAUDE.md's corollary is exact: if train2017 feeds a generation pipeline, val2017
must not, because an image generated from a held-out photo carries that photo's content into
training, and nothing downstream can find it again.

WHAT THIS FILE DOES NOT DO. It generates; it does not judge. Whether the pose survived is
measured separately by comparing detected keypoints against the source render, because a
generator scoring its own output is not a measurement.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys

PROMPT = ("Generate a photograph of a real person in exactly the pose of the first image, "
          "with the appearance, skin, clothing and lighting of the second image. Keep the "
          "body position, limb angles and camera framing of the first image unchanged.")
NEGATIVE = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn "
            "face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused "
            "fingers, messy drawing, broken legs censor, censored, censor_bar")


def load_references(shard_dir, holdout_parquet, n):
    import pyarrow.parquet as pq
    from PIL import Image

    if not os.path.exists(holdout_parquet):
        sys.exit("FAIL  no holdout manifest at %s: the exclusion cannot be proven, and an "
                 "unproven exclusion is the failure the rule names" % holdout_parquet)
    holdout = set(pq.read_table(holdout_parquet)["image_id"].to_pylist())
    out = []
    for shard in sorted(glob.glob(os.path.join(shard_dir, "images_*.parquet"))):
        for row in pq.read_table(shard).to_pylist():
            if row["image_id"] in holdout:
                sys.exit("FAIL  CONTAMINATION: image_id %d is one of the %d blinded holdout "
                         "images. Refusing to generate from it." % (row["image_id"], len(holdout)))
            if hashlib.sha256(row["image"]).hexdigest() != row["sha256"]:
                sys.exit("FAIL  sha256 mismatch on image_id %d" % row["image_id"])
            out.append((row["image_id"], Image.open(io.BytesIO(row["image"])).convert("RGB")))
            if len(out) >= n:
                return out, len(holdout)
    return out, len(holdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=r"C:\anny_test")
    ap.add_argument("--shards", default="coco_images_train2017")
    ap.add_argument("--holdout", default=r"coco_person_commercial_val2017\images.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--text-guidance", type=float, default=5.0)
    ap.add_argument("--image-guidance", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=512)
    # OMNIGEN2'S CODE IS DECLARED NOWHERE, AND THIS ARGUMENT IS THE SYMPTOM. It is not a
    # wheel: `omnigen2_edit.py` takes the same `--repo` pointing at a manual checkout of
    # VectorSpaceLab/OmniGen2, and the only copy on this desk rides along inside the
    # EditScore wheel's `examples/OmniGen2-RL/`. So the generator that produces two of the
    # four corpus appearances depends on source that no manifest names, no lockfile pins,
    # and `repo status` cannot see -- the hazard the `uv` blocklist entry is written about,
    # arriving through a checkout instead of a pip install.
    #
    # The fix is a `<project>` in the goal manifest with a pinned revision, forked first per
    # the fork-before-you-pin rule. Until that lands this argument is load-bearing and its
    # default is a path inside another environment, which is worse than it looks.
    ap.add_argument("--repo", default=r"C:\omnigen2-src",
        help="checkout of VectorSpaceLab/OmniGen2 (see the note in the source)")
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

    refs, n_holdout = load_references(a.shards, a.holdout, len(sources))
    print("  ok    %d reference photographs, none among the %d blinded holdout ids, all "
          "sha256-verified" % (len(refs), n_holdout))

    if not os.path.isdir(os.path.join(a.repo, "omnigen2")):
        sys.exit("FAIL  no omnigen2 package under %s. Pass --repo at a checkout of "
                 "VectorSpaceLab/OmniGen2." % a.repo)
    sys.path.insert(0, os.path.abspath(a.repo))

    import torch
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    repo = "OmniGen2/OmniGen2"
    # THE SAME TWO LINES omnigen2_edit.py ALREADY CARRIES, and omitting them cost a run.
    # `model_index.json` names its custom classes by BARE MODULE NAME --
    # "transformer_omnigen2", "scheduling_flow_match_euler_discrete" -- and diffusers
    # resolves those with a plain importlib call while validating components. The files ship
    # inside the WEIGHTS repo, so their directories go on the path; vendoring a copy would
    # drift from the checkpoint they belong to.
    #
    # This is the second file to need it, which means it should be a shared helper rather
    # than a paragraph copied twice. Recorded here rather than quietly duplicated a third
    # time.
    from huggingface_hub import hf_hub_download
    for rel in ("transformer/transformer_omnigen2.py",
                "scheduler/scheduling_flow_match_euler_discrete.py"):
        sys.path.insert(0, os.path.dirname(hf_hub_download(repo, rel)))

    tr = OmniGen2Transformer2DModel.from_pretrained(repo, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16)
    ml = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, subfolder="mllm",
                                                            torch_dtype=torch.bfloat16)
    pipe = OmniGen2Pipeline.from_pretrained(repo, transformer=tr, mllm=ml,
                                            torch_dtype=torch.bfloat16, trust_remote_code=True)
    pipe.to("cuda")

    manifest = []
    for src, (ref_id, ref) in zip(sources, refs):
        pose = Image.open(src).convert("RGB").resize((a.size, a.size), Image.LANCZOS)
        appearance = ref.resize((a.size, a.size), Image.LANCZOS)
        torch.cuda.reset_peak_memory_stats()
        # WHICH OMNIGEN2 IS A FACT, NOT A DISCOVERY, AND FINDING THAT OUT COST TWO RUNS. The
        # copy vendored inside the EditScore wheel's examples/OmniGen2-RL takes `size=(w,h)`
        # and wants input_images as a list OF LISTS; upstream at 18e6f9d5 takes height/width
        # and a flat list. Same import path, different API, and nothing on this desk said
        # which was in use. Upstream is pinned in --repo's default now and the revision is
        # recorded in the manifest this writes.
        #
        # Passed explicitly because align_res only fires at len(input_images) == 1, and this
        # algorithm always passes two.
        res = pipe(prompt=PROMPT, input_images=[pose, appearance],
                   height=a.size, width=a.size,
                   num_inference_steps=a.steps, max_sequence_length=1024,
                   text_guidance_scale=a.text_guidance,
                   image_guidance_scale=a.image_guidance,
                   negative_prompt=NEGATIVE, num_images_per_prompt=1,
                   generator=torch.Generator(device="cuda").manual_seed(0),
                   output_type="pil")
        peak = torch.cuda.max_memory_allocated() / 2**30
        name = os.path.basename(src)[:-4]
        dst = os.path.join(a.out, "%s_from_%012d.png" % (name, ref_id))
        res.images[0].save(dst)
        # The pose image is saved beside it at the same size: the drift measurement compares
        # them, and comparing against a differently-sized source would measure the resize.
        pose.save(os.path.join(a.out, "%s_pose.png" % name))
        manifest.append({"pose": os.path.basename(src), "reference_image_id": ref_id,
                         "out": os.path.basename(dst), "size": a.size,
                         "steps": a.steps, "peak_gib": round(peak, 2)})
        print("  ..    %-34s ref %012d  peak %.2f GiB" % (os.path.basename(dst), ref_id, peak))

    import subprocess
    try:
        rev = subprocess.check_output(["git", "-C", a.repo, "rev-parse", "HEAD"],
                                      text=True).strip()
    except Exception:                                             # noqa: BLE001
        rev = "unknown"
    json.dump({"prompt": PROMPT, "negative": NEGATIVE, "model": repo, "code_revision": rev,
               "text_guidance": a.text_guidance, "image_guidance": a.image_guidance,
               "frames": manifest},
              open(os.path.join(a.out, "generation.json"), "w"), indent=1)
    print("  ok    %d frames -> %s" % (len(manifest), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
