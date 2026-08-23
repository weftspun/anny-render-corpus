"""OmniGen2 at 4 bits, which is the run the 8 GB device question actually needs.

WHY THIS FILE EXISTS SEPARATELY FROM UPSTREAM'S inference.py. Upstream runs bf16, because they
run on cards that hold it: measured here, bf16 peaks at 17.3 GiB. The Hailo-10H on the ASUS
UGen300 has 8 GB, so the only version of this model that could ever land there is quantised,
and the number that matters is the 4-bit one rather than the 4-bit estimate.

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
    for rel in ("transformer/transformer_omnigen2.py",
                "scheduler/scheduling_flow_match_euler_discrete.py"):
        sys.path.insert(0, os.path.dirname(hf_hub_download(args.model, rel)))

    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import BitsAndBytesConfig as TransformersBnb
    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    t0 = time.time()
    # Both halves quantised. The 4B transformer is the obvious one; leaving the 3.7B MLLM in
    # bf16 would put 7.4 GiB back and defeat the point.
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", quantization_config=DiffusersBnb(**nf4),
        torch_dtype=torch.bfloat16)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, subfolder="mllm", quantization_config=TransformersBnb(**nf4),
        torch_dtype=torch.bfloat16)
    # trust_remote_code, and what was actually trusted: the scheduler that ships in the weights
    # repo, 228 lines, importing only math, numpy, torch and diffusers -- no subprocess, no
    # network, no eval. It is the same file upstream's own inference.py executes; this path just
    # reaches it through a different door because the components are pre-quantised.
    pipe = OmniGen2Pipeline.from_pretrained(
        args.model, transformer=transformer, mllm=mllm, torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    pipe.vae.to("cuda")
    load_s = time.time() - t0

    def gib(n):
        return n / 2**30

    print(f"loaded in {load_s:.0f}s | weights on device {gib(torch.cuda.memory_allocated()):.2f} GiB")
    weights_gib = gib(torch.cuda.memory_allocated())

    src = Image.open(args.image).convert("RGB")
    record = {"model": args.model, "quant": "nf4-double", "steps": args.steps,
              "text_guidance_scale": args.text_guidance, "image_guidance_scale": args.image_guidance,
              "seed": args.seed, "torch": torch.__version__,
              "weights_gib": round(weights_gib, 2), "outputs": {}}

    for name, prompt in PROMPTS.items():
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        res = pipe(prompt=prompt, input_images=[src], width=1024, height=1024,
                   num_inference_steps=args.steps, max_sequence_length=1024,
                   text_guidance_scale=args.text_guidance,
                   image_guidance_scale=args.image_guidance,
                   negative_prompt="", num_images_per_prompt=1,
                   generator=torch.Generator(device="cuda").manual_seed(args.seed),
                   output_type="pil")
        dst = os.path.join(args.out, f"omnigen2_{name}_nf4.png")
        res.images[0].save(dst)
        peak = gib(torch.cuda.max_memory_allocated())
        record["outputs"][name] = {"file": os.path.basename(dst),
                                   "seconds": round(time.time() - t1, 1),
                                   "peak_vram_gib": round(peak, 2)}
        print(f"  ok   {dst}  {time.time()-t1:.0f}s  peak {peak:.2f} GiB")

    with open(os.path.join(args.out, "omnigen2_provenance_nf4.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    print("provenance written")


if __name__ == "__main__":
    main()
