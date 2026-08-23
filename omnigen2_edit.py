"""OmniGen2 image editing, at the precision the destination allows.

bf16 IS THE DEFAULT BECAUSE QUANTISED WEIGHTS DO NOT PRODUCE CORPUS DATA. That is condition 5
on generated synthetic in CLAUDE.md, and it is a DECISION rather than a measurement.

RETRACTED, AND THE RETRACTION STAYS BESIDE WHAT IT RETRACTS. This docstring read that the
condition "is measured": the same edit, same seed, same guidance, scoring 0.776 silhouette
agreement at bf16 against 0.328 at NF4, because at four bits the photographic prompt stopped
editing and started generating. Those two runs did not differ only in precision. The bf16 one
went through upstream's inference.py, which passes a quality-control negative prompt by
default; the NF4 one went through this script, which passed an empty string. Holding the
prompt fixed reverses the result -- NF4 with the negative prompt scores 0.825 and bf16 scores
0.776 -- so the prompt moved it and precision did not measurably move it.

The default is unchanged, because the condition was decided rather than derived. What changes
is the argument this file offers for it: citing a withdrawn number invites the next reader to
re-derive it, reach the opposite answer, and quietly drop the rule.

NF4 REMAINS AVAILABLE, AND EVERY OUTPUT SAYS WHAT IT IS FOR. Quantisation for device
qualification is a different activity: fitting this model into 6.72 GiB answers whether it
clears the ASUS UGen300's 8 GB, and that is evidence about memory. So --precision nf4 works,
and the provenance it writes carries corpus_eligible false. The rule is about destination.

WHAT THIS DOES NOT SHOW. Fitting in 8 GB of a desktop card's VRAM is necessary and nowhere near
sufficient for the accelerator: the model still has to compile through the Dataflow Compiler,
and Hailo's own GenAI zoo ships no diffusion model at all -- it had Stable Diffusion 1.5 in its
first release and the changelog records it being removed. So read this as a size measurement,
not as a port.

The prompts and the input are rule 2's, unchanged from the bf16 run, so the two are comparable.
"""
import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

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
    ap.add_argument("--repo", required=True, help="checkout of VectorSpaceLab/OmniGen2")
    ap.add_argument("--model", default="OmniGen2/OmniGen2")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--text-guidance", type=float, default=5.0)
    ap.add_argument("--image-guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    # UPSTREAM'S DEFAULT, NOT AN EMPTY STRING, AND THE DIFFERENCE IS NOT COSMETIC. This script
    # passed "" and upstream's inference.py passes a quality-control negative prompt. Comparing
    # a run of ours against a run of theirs therefore varied two things at once, which is how a
    # precision claim got made on a confounded pair.
    ap.add_argument("--negative", default=(
        "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, "
        "mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy "
        "drawing, broken legs censor, censored, censor_bar"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--precision", choices=["bf16", "nf4"], default="bf16",
                    help="bf16 for anything that becomes corpus; nf4 only for device sizing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.repo)

    # model_index.json names its custom classes by BARE MODULE NAME -- "transformer_omnigen2"
    # and "scheduling_flow_match_euler_discrete" -- and diffusers resolves those with a plain
    # importlib call while validating components. Passing a pre-quantised transformer takes the
    # path where that validation runs, so the import has to succeed:
    #
    #   ModuleNotFoundError: No module named 'transformer_omnigen2'
    #
    # The files ship inside the weights repo, so the fix is to put their directories on the
    # path rather than to vendor a copy that would then drift from the checkpoint they belong to.
    from huggingface_hub import hf_hub_download
    resolved = None
    for rel in ("transformer/transformer_omnigen2.py",
                "scheduler/scheduling_flow_match_euler_discrete.py"):
        local = hf_hub_download(args.model, rel)
        sys.path.insert(0, os.path.dirname(local))
        # THE REVISION COMES OUT OF THE PATH THAT WAS ACTUALLY DOWNLOADED, not out of a
        # second call to the hub API. `model_index.json` and these two files are fetched
        # here and the pipeline is loaded from the same cache, so the snapshot directory
        # names the commit that produced this run's weights. An API call answers what the
        # repository resolves to NOW, which is a different question and can differ by a
        # push. Condition 1 asks for the checkpoint that generated the data.
        parts = os.path.normpath(local).split(os.sep)
        if "snapshots" in parts:
            resolved = parts[parts.index("snapshots") + 1]

    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import BitsAndBytesConfig as TransformersBnb
    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    quantised = args.precision == "nf4"
    if quantised:
        print("  NOTE  nf4: these outputs are device-sizing evidence, NOT corpus data "
              "(CLAUDE.md, generated-synthetic condition 5)")
    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    t0 = time.time()
    # Both halves quantised. The 4B transformer is the obvious one; leaving the 3.7B MLLM in
    # bf16 would put 7.4 GiB back and defeat the point.
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16,
        **({"quantization_config": DiffusersBnb(**nf4)} if quantised else {}))
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, subfolder="mllm", torch_dtype=torch.bfloat16,
        **({"quantization_config": TransformersBnb(**nf4)} if quantised else {}))
    # trust_remote_code, and what was actually trusted: the scheduler that ships in the weights
    # repo, 228 lines, importing only math, numpy, torch and diffusers -- no subprocess, no
    # network, no eval. It is the same file upstream's own inference.py executes; this path just
    # reaches it through a different door because the components are pre-quantised.
    pipe = OmniGen2Pipeline.from_pretrained(
        args.model, transformer=transformer, mllm=mllm, torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    pipe.vae.to("cuda")
    if not quantised:
        pipe.to("cuda")          # unquantised components are not placed by the loader
    load_s = time.time() - t0

    def gib(n):
        return n / 2**30

    print(f"loaded in {load_s:.0f}s | weights on device {gib(torch.cuda.memory_allocated()):.2f} GiB")
    weights_gib = gib(torch.cuda.memory_allocated())

    src = Image.open(args.image).convert("RGB")

    # THE RECORD IS SHAPED LIKE THE RELATIONS IT BECOMES, and two fields it needed were
    # missing until now. `edit_models.revision` was absent entirely -- the record named the
    # repository and never the commit, so "generated by OmniGen2" did not resolve to a
    # checkpoint. And the POSITIVE prompt was absent while the negative one was written,
    # so the conditioning was recoverable only by trusting that PROMPTS above had not been
    # edited since. Condition 1 asks for the prompt WITH the data, and a constant in a
    # source file is the second-place-a-fact-lives failure, not a record.
    #
    # `corpus_eligible` is NOT stored. It is `precision != quantised`, so storing it makes a
    # derivable column, and the schema derives it in validate() instead.
    if resolved is None:
        # An unmet precondition is a FAIL. A run whose checkpoint cannot be named produces
        # frames that can never satisfy condition 1, and finding that out after the GPU time
        # is spent is the expensive order to find it out in.
        sys.exit("FAIL  could not resolve a commit for %s: no snapshots/ in the cache path. "
                 "Without it condition 1 cannot be met and the output is not corpus data."
                 % args.model)
    record = {"edit_models": {"repo_id": args.model, "revision": resolved},
              "edit_runs": {"precision": args.precision, "steps": args.steps,
                            "text_guidance_scale": args.text_guidance,
                            "image_guidance_scale": args.image_guidance,
                            "negative_prompt": args.negative},
              "torch": torch.__version__,
              "weights_gib": round(weights_gib, 2), "outputs": {}}

    for name, prompt in PROMPTS.items():
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        res = pipe(prompt=prompt, input_images=[src], width=1024, height=1024,
                   num_inference_steps=args.steps, max_sequence_length=1024,
                   text_guidance_scale=args.text_guidance,
                   image_guidance_scale=args.image_guidance,
                   negative_prompt=args.negative, num_images_per_prompt=1,
                   generator=torch.Generator(device="cuda").manual_seed(args.seed),
                   output_type="pil")
        dst = os.path.join(args.out, f"omnigen2_{name}_{args.precision}{args.tag}.png")
        res.images[0].save(dst)
        peak = gib(torch.cuda.max_memory_allocated())
        record["outputs"][name] = {"file": os.path.basename(dst),
                                   "prompt": prompt,          # the conditioning, not its name
                                   "seed": args.seed,
                                   "seconds": round(time.time() - t1, 1),
                                   "peak_vram_gib": round(peak, 2)}
        print(f"  ok   {dst}  {time.time()-t1:.0f}s  peak {peak:.2f} GiB")

    with open(os.path.join(args.out, f"omnigen2_provenance_{args.precision}{args.tag}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    # STILL A SIDECAR, AND SAYING SO IS THE POINT. Condition 1 asks for the record WITH the
    # data, and a JSON file beside a directory of PNGs parts company with them the first time
    # somebody copies the images. `anny_render_schema.py` now carries edit_models,
    # edit_prompts, edit_runs and edited_renders for exactly this, and the fields above map
    # onto them one for one. What is not built is the writer, because it needs a `render_id`
    # to join to and the renderer does not emit parquet yet -- that is T02, not this file.
    print("provenance written (sidecar; maps 1:1 onto the edit_* relations, not yet written "
          "as rows -- needs render_id from the parquet renderer)")


if __name__ == "__main__":
    main()
