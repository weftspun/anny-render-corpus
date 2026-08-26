"""Load OmniGen2 at bf16, for `check_background_removal.py --method omnigen2`.

The loading path is `omnigen2_edit.py`'s, not a second derivation of it. Two things there are
load-bearing and easy to lose in a re-write:

  a. `model_index.json` names its custom classes by BARE MODULE NAME, so the two .py files that
     ship inside the WEIGHTS repo have to be on sys.path before `from_pretrained` validates
     components, or the load dies on `ModuleNotFoundError: No module named 'transformer_omnigen2'`.
  b. the unquantised components are not placed on the device by the loader, so `pipe.to("cuda")`
     is not optional.

bf16 rather than NF4, matching `omnigen2_edit.py`'s default. Nothing here writes corpus data --
this produces a MASK that is scored against an exact alpha -- but running the remover at a
different precision from the generator it stands in for would make the two numbers incomparable.
"""
import os
import sys

import torch

_PIPE = None


def load_pipeline(repo=None, model="OmniGen2/OmniGen2"):
    """Cached: the harness calls this once per case and the weights are 17 GiB."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    if repo:
        sys.path.insert(0, repo)

    from huggingface_hub import hf_hub_download
    for rel in ("transformer/transformer_omnigen2.py",
                "scheduler/scheduling_flow_match_euler_discrete.py"):
        sys.path.insert(0, os.path.dirname(hf_hub_download(model, rel)))

    from transformers import Qwen2_5_VLForConditionalGeneration
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    transformer = OmniGen2Transformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=torch.bfloat16)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model, subfolder="mllm", torch_dtype=torch.bfloat16)
    pipe = OmniGen2Pipeline.from_pretrained(
        model, transformer=transformer, mllm=mllm, torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    pipe.vae.to("cuda")
    pipe.to("cuda")
    _PIPE = pipe
    return pipe
