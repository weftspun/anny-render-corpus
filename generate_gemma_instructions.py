"""Generate the edit instruction per (frame_a, frame_b) pair via Gemma-4-12B.

Model-devised taxonomy: instead of See-Through's fixed part vocabulary
(bodytags_v3.json), we let the VLM describe what changed in its own words. The output
becomes the row's `instruction` column. Prompt is fixed; every edit family sees the
same instruction verbatim, so any variance across rows comes from the images not from
us drifting wording.

Provenance recorded per synthetic-data condition 1 (CLAUDE.md): model checkpoint HF
ref, prompt sha256, generation params, per-edit sha256 of the input images. A later
consumer can reproduce the instruction given the model + prompt + images.

Frame selection: picks the near-front view (yaw closest to 0, pitch closest to 0) from
each render dir's sidecars so the VLM sees the face at the angle a human would.

Output: build/edit/instructions.json with {edit_name: {instruction, provenance}}.

Usage:
    pixi run --environment anny-mac python generate_gemma_instructions.py \
        [--model chibifire/gemma-4-12B-it-qat-q4_0-unquantized] \
        [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import torch


PROMPT = (
    "Look at these two renders of a 3D human head, both from the same viewing "
    "angle. Frame A is the input; Frame B is the target of an edit. In ONE "
    "sentence, describe what an observer would see change between A and B. Do "
    "not name muscle names, action units, or blendshapes -- describe the "
    "appearance change. Start your answer with 'The '. Output only the sentence."
)


def pick_front_view_sidecar(view_dir: pathlib.Path) -> pathlib.Path:
    """View closest to (yaw=0, pitch=0). Sidecar JSON carries yaw_deg/pitch_deg."""
    best, best_score = None, float("inf")
    for p in view_dir.glob("view_*.json"):
        if ".keypoints" in p.name:
            continue
        s = json.loads(p.read_text())
        score = abs(s.get("yaw_deg", 0.0)) + abs(s.get("pitch_deg", 0.0))
        if score < best_score:
            best, best_score = p, score
    if best is None:
        raise SystemExit(f"no view sidecars in {view_dir}")
    return best


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir",   type=pathlib.Path, default=pathlib.Path("build/edit"))
    ap.add_argument("--render-dir", type=pathlib.Path, default=pathlib.Path("build/edit/renders"))
    ap.add_argument("--model",      default="chibifire/gemma-4-12B-it-qat-q4_0-unquantized")
    ap.add_argument("--dtype",      default="bfloat16")
    ap.add_argument("--out",        type=pathlib.Path, default=pathlib.Path("build/edit/instructions.json"))
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())

    # Import late so pixi env checks happen before HF download attempts.
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = getattr(torch, a.dtype)
    print(f"loading {a.model} in {a.dtype}...", flush=True)
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print("model loaded", flush=True)

    prompt_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    frame_a_sidecar = pick_front_view_sidecar(a.render_dir / "frame_a")
    frame_a_png = frame_a_sidecar.with_suffix(".png")
    frame_a_sha = sha256_of(frame_a_png)

    out = {"prompt": PROMPT, "prompt_sha256": prompt_sha,
           "model": a.model, "dtype": a.dtype,
           "temperature": 0.0, "seed": 0,
           "frame_a": {"path": str(frame_a_png.relative_to(pathlib.Path.cwd())),
                       "sha256": frame_a_sha,
                       "yaw_deg": json.loads(frame_a_sidecar.read_text())["yaw_deg"],
                       "pitch_deg": json.loads(frame_a_sidecar.read_text())["pitch_deg"]},
           "edits": {}}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for edit_name in manifest["edits"]:
        b_sidecar = pick_front_view_sidecar(a.render_dir / edit_name / "frame_b")
        b_png = b_sidecar.with_suffix(".png")
        img_a = Image.open(frame_a_png).convert("RGB")
        img_b = Image.open(b_png).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_a},
            {"type": "image", "image": img_b},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                 temperature=1.0)  # do_sample=False makes temp irrelevant
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.decode(gen[0][prompt_len:], skip_special_tokens=True).strip()

        out["edits"][edit_name] = {
            "instruction": text,
            "frame_b_path": str(b_png.relative_to(pathlib.Path.cwd())),
            "frame_b_sha256": sha256_of(b_png),
        }
        print(f"  {edit_name}: {text}", flush=True)

    a.out.write_text(json.dumps(out, indent=2))
    print(f"done. instructions saved to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
