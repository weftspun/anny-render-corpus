"""Publish a corpus and a trained adapter to Hugging Face, with the card that makes them usable.

RFD 1141 gives the rule; this applies it. Code stays on GitHub, artifacts go to the hub, and
each names the other, because an artifact nobody can trace back to the code that made it is a
file rather than a result.

FOUR THINGS STOP A PUBLICATION, and they are checked before a byte is uploaded rather than
apologised for afterwards:

  quantised output          CLAUDE.md condition 5 keeps it out of a corpus
  blinded-holdout derived   val2017 and anything built from it inherit its status
  licence-dirty sources     the bar is commercial use AND derivatives
  a checkpoint as a model   14.8 GiB of somebody else's base weights under our name

The last one is measured rather than argued: 304 of 886 tensors in that checkpoint are ours,
and they extract to 19.52 MiB.

THE CARD CARRIES MEASUREMENTS, NOT CLAIMS. "Improves camera control" tells a reader nothing.
"Azimuth 90 moved from 97.6 degrees wrong to 13.3" tells them whether to bother.

    python publish_artifacts.py --namespace chibifire --dry-run
    python publish_artifacts.py --namespace chibifire
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

# The GitHub repository these artifacts come from, and the side of the hexagon it sits on.
# RFD 1141 requires both in the card: a reader who finds the artifact first has to be able to
# reach the code, and the side is what says where the code belongs.
SOURCE_REPO = "https://github.com/weftspun/anny-render-corpus"
SOURCE_SIDE = "6-datasource"
REPO_NAME = "anny-render-corpus"

# GENERATED DATA GETS ITS OWN REPOSITORY, NOT A SUBDIRECTORY. CLAUDE.md condition 2 says
# generated synthetic is "stored and manifested separately from constructed and real data,
# never merged into an undifferentiated pool". A directory satisfies the letter of that; a
# separate repository satisfies the intent, because a consumer who clones the corpus cannot
# pick up generated frames by accident when they are not there to pick up. Each card links
# the other, so nobody has to guess the second one exists.
GENERATED_REPO = "anny-render-corpus-generated"

# Refused outright. Each is a CLAUDE.md rule rather than a preference.
FORBIDDEN_SUBSTRINGS = ("val2017", "coco-ood", "rf-detr-keypoint-data", "nf4", "int4", "int8")


def hf_token(item="rkuylld4umpmaxvlvbp5q7kgii"):
    out = subprocess.run(["op", "item", "get", item, "--fields", "label=credential", "--reveal"],
                         capture_output=True, text=True, timeout=90)
    token = out.stdout.strip()
    if not token.startswith("hf_"):
        sys.exit("FAIL  1Password returned no usable token. Unlock the desktop app; `op` "
                 "cannot prompt from here and reports promptError instead. stderr: %s"
                 % out.stderr.strip()[:160])
    return token


# A drive-letter or home-directory path anywhere in a string, not only at its start.
ABSOLUTE_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/c/Users/|/Users/)[^\s\"']*", re.IGNORECASE)


def basename_paths(value):
    """Reduce any string that looks like a local path to its filename, recursively.

    The measurement JSONs record which frame they measured, and they recorded it the way the
    run received it: as a full path. `source_frame` and `frame` are the two that carry it, but
    naming those two fields would leave the next field somebody adds. Walking the structure
    catches them all, and `refuse_if_absolute` is what proves the walk was complete.
    """
    if isinstance(value, dict):
        return {k: basename_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [basename_paths(v) for v in value]
    if isinstance(value, str):
        # NOT ONLY WHOLE-STRING PATHS. The first version of this checked `startswith`, and a
        # path survived inside a sentence: "no person detected in C:\...\az045_A.png at
        # threshold 0.3" is an error message, not a path field, and it disclosed a home
        # directory just as effectively. Substitution inside the string is what catches both.
        return ABSOLUTE_RE.sub(lambda m: m.group(0).replace(chr(92), "/").rsplit("/", 1)[-1],
                               value)
    return value


def copy_json_sanitised(src, dst):
    dst.write_text(json.dumps(basename_paths(json.loads(src.read_text(encoding="utf-8"))),
                              indent=2), encoding="utf-8")


def refuse_if_absolute(root):
    """No published file may name a local filesystem.

    An absolute path in a dataset is two defects wearing one string: it discloses whose
    machine made the file, and it points somewhere the reader does not have, so the record is
    inert for everyone but us. This runs over the staged tree after the rewrites, because the
    rewrite is the thing that could silently miss a field.
    """
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() in (".png", ".safetensors"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for marker in (chr(67) + chr(58) + chr(92), "C:/", "/c/Users", "/Users/"):
            if marker in text:
                hits.append((str(p.relative_to(root)), marker))
                break
    if hits:
        for rel, marker in hits[:8]:
            print("  ABSOLUTE %-52s contains %r" % (rel, marker))
        sys.exit("FAIL  %d staged file(s) still name a local path. Publishing them would "
                 "disclose a home directory and hand the reader a record that points at a "
                 "filesystem they do not have." % len(hits))


def refuse_if_forbidden(paths):
    """A publication is easier to prevent than to withdraw."""
    bad = []
    for p in paths:
        low = str(p).lower()
        for marker in FORBIDDEN_SUBSTRINGS:
            if marker in low:
                bad.append((str(p), marker))
    if bad:
        for path, marker in bad[:6]:
            print("  REFUSED %-60s matches %r" % (path, marker))
        sys.exit("FAIL  %d file(s) match a blocked marker. val2017 and anything derived from "
                 "it are the blinded holdout; a quantised precision in a filename is "
                 "condition 5. Neither is published." % len(bad))


def stage_dataset(grid, corpus, ladders, out):
    """Assemble the upload tree, keeping constructed and generated data apart.

    CLAUDE.md condition 2 is explicit that generated data is manifested separately and never
    merged into an undifferentiated pool. The renders are constructed -- deterministic from a
    rig we hold, labels true by construction -- and the OmniGen2 views are not. They go in
    different directories so that a consumer cannot mix them by accident.
    """
    out = pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "renders").mkdir(parents=True)
    (out / "controls").mkdir(parents=True)
    (out / "records").mkdir(parents=True)
    (out / "measurements").mkdir(parents=True)

    grid = pathlib.Path(grid)
    counts = {"renders": 0, "controls": 0, "records": 0, "measurements": 0}
    for p in sorted(grid.glob("pose_*")):
        if p.suffix == ".png" and not p.name.endswith((".depth.png", ".pose.png")):
            shutil.copy2(p, out / "renders" / p.name); counts["renders"] += 1
        elif p.suffix == ".json":
            shutil.copy2(p, out / "renders" / p.name); counts["renders"] += 1
        elif p.name.endswith((".depth.png", ".pose.png")):
            shutil.copy2(p, out / "controls" / p.name); counts["controls"] += 1

    # PATHS ARE REWRITTEN RELATIVE TO THE DATASET ROOT, and this is not cosmetic. The
    # trainer needs absolute paths on the machine that runs it, so the local records carry
    # them and are correct there. Published, the same string is two defects at once: it names
    # a user's home directory, and it points at a filesystem the reader does not have, so the
    # dataset is unusable by anyone who downloads it. The local copy keeps absolutes; the
    # published copy gets `renders/<name>`.
    corpus = pathlib.Path(corpus)
    for name in ("train_formA.jsonl", "val_formA.jsonl"):
        src = corpus / name
        if src.is_file():
            rows = []
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["input_images"] = ["renders/" + os.path.basename(q)
                                     for q in r.get("input_images", [])]
                r["output_image"] = "renders/" + os.path.basename(r["output_image"])
                rows.append(json.dumps(r))
            (out / "records" / name).write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
            counts["records"] += 1

    # The two YAML files named an absolute JSONL. Rewritten to sit beside their records, so a
    # reader can point the trainer at the folder they downloaded rather than at ours.
    (out / "records" / "edit_formA.yml").write_text(
        "data:" + chr(10) + "  - " + chr(10) + "    path: 'train_formA.jsonl'" + chr(10)
        + "    type: 'edit'" + chr(10) + "    ratio: !!float 1.0" + chr(10), encoding="utf-8")
    (out / "records" / "mix_formA.yml").write_text(
        "# ratio is inert in this trainer; the mix is controlled by row counts." + chr(10)
        + "data:" + chr(10) + "  - " + chr(10) + "    path: 'edit_formA.yml'" + chr(10)
        + "    type: 'edit'" + chr(10) + "    ratio: !!float 1.0" + chr(10), encoding="utf-8")
    counts["records"] += 2

    for label, d in ladders.items():
        d = pathlib.Path(d)
        for name in ("ladder.json", "azimuth_recovery_A.json", "omnigen2_bench.json"):
            src = d / name
            if src.is_file():
                copy_json_sanitised(src, out / "measurements" / ("%s_%s" % (label, name)))
                counts["measurements"] += 1
    return out, counts


def stage_generated(ladders, out):
    """The OmniGen2 outputs, with the record that makes them re-derivable.

    Condition 1 asks for the generating model, checkpoint and conditioning WITH the data. The
    ladder writes a JSON carrying the resolved commit, the prompt for every view, the seed and
    the settings, so it travels beside the images rather than being reconstructed later from a
    source file that has since moved on.
    """
    out = pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = {}
    for label, d in ladders.items():
        d = pathlib.Path(d)
        images = sorted(d.glob("az*.png")) + sorted(d.glob("*_0.png"))
        if not images:
            continue
        sub = out / label
        sub.mkdir(parents=True, exist_ok=True)
        for q in images:
            shutil.copy2(q, sub / q.name)
        for name in ("ladder.json", "azimuth_recovery_A.json", "omnigen2_bench.json"):
            src = d / name
            if src.is_file():
                copy_json_sanitised(src, sub / name)
        counts[label] = len(images)
    return out, counts


def generated_card(counts):
    rows = chr(10).join("| `%s/` | %d generated views |" % (k, v)
                        for k, v in sorted(counts.items()))
    return """---
license: apache-2.0
tags:
  - synthetic
  - generated
  - camera-control
---

# anny-render-corpus-generated

Images **generated by OmniGen2** from the constructed renders in
[`chibifire/anny-render-corpus`](https://huggingface.co/datasets/chibifire/anny-render-corpus).

**Code:** [`weftspun/anny-render-corpus`](%s), on the **`%s`** side of the hexagon.

## Why this is a separate repository

These are *generated* synthetic, not constructed. They were sampled from a model rather than
rendered deterministically from a rig, so their labels are inferred and not true by
construction. Our working agreement requires generated data to be stored and manifested
separately and never merged into an undifferentiated pool. A separate repository enforces
that where a subdirectory would only describe it: clone the corpus and these cannot come
along.

They are **not** training data on their own, and evaluation does not use them.

## Provenance

Every directory carries the JSON its run wrote: the resolved OmniGen2 commit
`df5dca8a981d74e6c3af214c145f5c735fe72367`, the full prompt for each view, the seed, the step
count, `cfg_range` and the guidance scales. bf16 throughout, because quantised weights do not
produce corpus data here.

| directory | contents |
| --- | --- |
%s

## What they show

`ladder/` is the base model asked for eight camera azimuths: recovered azimuth tracks the
request with a slope of **0.04**. `lora/` is the same eight prompts after training, at
**0.10**, with azimuth 90 moving from 97.6 degrees wrong to 13.3.

Two views in `lora/` have no detectable person in them. They are kept rather than dropped: a
set that quietly excludes its failures reports a better result than it earned.
""" % (SOURCE_REPO, SOURCE_SIDE, rows)


def dataset_card(counts):
    return """---
license: apache-2.0
task_categories:
  - keypoint-detection
  - image-to-image
tags:
  - synthetic
  - camera-control
  - pose
size_categories:
  - n<1K
---

# anny-render-corpus

A camera-controlled render corpus from the ANNY rig, and the measurements that motivated it.

**Code:** [`weftspun/anny-render-corpus`](%s), on the **`%s`** side of the hexagon.
Everything here is produced by scripts in that repository and can be regenerated from it.

## What this is for

Asked in plain language for eight camera azimuths, OmniGen2 returns a body that does not
turn. Recovered azimuth tracks the request with a slope of **0.04**, where 1.00 is obedience
and 0.00 is a body that never moved. Six of eight views came back between -1 and -11 degrees
whatever was asked; only the back view worked.

This corpus exists to close that gap, the way fal's Multiple-Angles LoRA closed it for
Qwen-Image-Edit with 3000+ Gaussian-splat renders.

## Contents

| directory | what |
| --- | --- |
| `renders/` | %d files: 96 rendered views with their camera sidecars |
| `controls/` | %d files: exact depth maps and pose skeletons |
| `records/` | %d files: the training records, 95 of them, and their mix config |
| `measurements/` | %d files: the ladder timings and the azimuth recovery |

**These renders are constructed synthetic, not generated.** They are rendered
deterministically from a rig we hold, the labels are true by construction rather than
inferred, and the same seed reproduces them. No generated image is mixed in here.

## The measurements, so the numbers are not claims

Generation cost, bf16 at 1024 square on an RTX 3090, uncontended:

| setting | seconds |
| --- | ---: |
| 30 steps, `max_sequence_length` 1024 | 131 |
| 30 steps, `max_sequence_length` 256 | 131 |
| 30 steps, `cfg_range` (0.0, 0.6) | 103 |
| one input image | 103 |
| two input images | 192 |
| three input images | 529 |

In-context images dominate cost, not step count, because each extends the sequence through
all three CFG passes.

## Licence

Apache-2.0 for this corpus. The rig it is rendered from is
[ANNY](https://github.com/naver/anny), Apache-2.0, NAVER Corp. `CITATION.cff` names every
source.
""" % (SOURCE_REPO, SOURCE_SIDE, counts["renders"], counts["controls"],
       counts["records"], counts["measurements"])


def model_card():
    return """---
license: apache-2.0
base_model: OmniGen2/OmniGen2
library_name: peft
tags:
  - lora
  - camera-control
---

# anny-camera-lora

A camera-control LoRA for [OmniGen2](https://huggingface.co/OmniGen2/OmniGen2), trained on
[`chibifire/anny-render-corpus`](https://huggingface.co/datasets/chibifire/anny-render-corpus).

**Code:** [`weftspun/anny-render-corpus`](%s), on the **`%s`** side of the hexagon.

## What it changes, measured

Identical prompts, identical seed, only the weights differ. Azimuth is recovered by fitting
the ANNY body to detected keypoints, so it is arithmetic on the body rather than an opinion
about the picture.

| requested | base model | with this adapter |
| ---: | ---: | ---: |
| 0 | 10.4 off | **6.2 off** |
| 90 | 97.6 off | **13.3 off** |
| 180 | 11.9 off | 18.6 off |
| 45, 135 | wrong | no person detected |
| 225, 270, 315 | wrong | still wrong |

Slope of recovered against requested: **0.04 to 0.10**.

**This adapter is undertrained and this card says so.** It learned the three cardinal
directions and overfit elsewhere, which is what 95 images of one body should produce. It does
not yet deliver general camera control, and the slope is the number to watch.

## Training

200 steps, 22 minutes, one RTX 3090. Rank 8, alpha 8, attention only
(`to_k`, `to_q`, `to_v`, `to_out.0`), bf16, gradient checkpointing, 8-bit AdamW, batch 1 with
8 accumulation steps, 256 square. Loss 0.196 to 0.111.

512 square did not complete a step in twelve minutes on the same card, and raised nothing:
the driver pages into shared memory rather than failing.

## Contents

`adapter_model.safetensors` carries the **304 trained tensors, 19.52 MiB**. The trainer's own
checkpoint is 14.8 GiB because it saves the whole model under FSDP, and 866 of those tensors
are the base model unchanged.

## The base weights

Load this against [`OmniGen2/OmniGen2`](https://huggingface.co/OmniGen2/OmniGen2) at revision
`df5dca8a981d74e6c3af214c145f5c735fe72367`. That exact revision is also mirrored at
[`chibifire/omnigen2-base-df5dca8a`](https://huggingface.co/chibifire/omnigen2-base-df5dca8a),
unmodified and Apache-2.0, so a cited revision stays fetchable if upstream moves. Use upstream
when you can; the mirror is the fallback.

An adapter is deltas against a base it cannot function without, so the pair is only useful
together.
""" % (SOURCE_REPO, SOURCE_SIDE)


def citation(kind):
    return """cff-version: 1.2.0
message: "If you use this %s, please cite it as below."
type: %s
title: "%s"
authors:
  - name: "weft"
license: Apache-2.0
repository-code: "%s"
abstract: >-
  Produced by the scripts in the repository above, which sits on the %s side of the
  weftspun hexagon. Renders are constructed synthetic: deterministic from a rig held
  locally, with labels true by construction, reproducible from the same seed.
references:
  - type: software
    title: "Anny"
    authors:
      - name: "NAVER Corp."
    url: "https://github.com/naver/anny"
    notes: "Apache-2.0. The rig every render comes from."
  - type: software
    title: "OmniGen2"
    authors:
      - name: "VectorSpaceLab"
    url: "https://huggingface.co/OmniGen2/OmniGen2"
    notes: "Apache-2.0. The base model, at revision df5dca8a981d74e6c3af214c145f5c735fe72367."
  - type: software
    title: "RF-DETR"
    authors:
      - name: "Roboflow"
    url: "https://github.com/roboflow/rf-detr"
    notes: "Apache-2.0. Keypoint detection used to recover an azimuth per view."
""" % (kind, "dataset" if kind == "dataset" else "software",
       REPO_NAME if kind == "dataset" else "anny-camera-lora",
       SOURCE_REPO, SOURCE_SIDE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="chibifire")
    ap.add_argument("--grid", default=r"C:\Users\ernes\Desktop\grid96-anny-rest")
    ap.add_argument("--corpus", default=r"C:\Users\ernes\Desktop\omnigen2-camera-lora")
    ap.add_argument("--adapter", default=r"C:\Users\ernes\Desktop\anny-camera-lora-adapter")
    ap.add_argument("--stage", default=r"C:\Users\ernes\Desktop\hf-stage")
    ap.add_argument("--private", action="store_true",
                    help="public by default; RFD 1141 publishes so others can check the work")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ladders = {"ladder": r"C:\Users\ernes\Desktop\ladder-8az",
               "lora": r"C:\Users\ernes\Desktop\ladder-lora",
               "bench": r"C:\Users\ernes\Desktop\omnigen2-bench"}
    stage, counts = stage_dataset(args.grid, args.corpus, ladders, args.stage)
    gen_stage, gen_counts = stage_generated(ladders, args.stage + "-generated")
    refuse_if_forbidden([q for q in gen_stage.rglob("*") if q.is_file()])
    (gen_stage / "README.md").write_text(generated_card(gen_counts), encoding="utf-8")
    (gen_stage / "CITATION.cff").write_text(citation("generated"), encoding="utf-8")

    files = [p for p in stage.rglob("*") if p.is_file()]
    refuse_if_forbidden(files)

    refuse_if_absolute(stage)
    refuse_if_absolute(gen_stage)
    (stage / "README.md").write_text(dataset_card(counts), encoding="utf-8")
    (stage / "CITATION.cff").write_text(citation("dataset"), encoding="utf-8")

    adapter = pathlib.Path(args.adapter)
    if not (adapter / "adapter_model.safetensors").is_file():
        sys.exit("FAIL  no adapter at %s; run extract_lora_adapter.py first. A checkpoint is "
                 "not a model repository." % adapter)
    (adapter / "README.md").write_text(model_card(), encoding="utf-8")
    (adapter / "CITATION.cff").write_text(citation("model"), encoding="utf-8")

    total = sum(p.stat().st_size for p in files)
    print("dataset stage %s" % stage)
    for k, v in counts.items():
        print("  %-14s %d file(s)" % (k, v))
    print("  %.1f MiB total" % (total / 2 ** 20))
    print("adapter %s  %.2f MiB"
          % (adapter, (adapter / "adapter_model.safetensors").stat().st_size / 2 ** 20))

    gen_files = [q for q in gen_stage.rglob("*") if q.is_file()]
    print("generated stage %s  (its own repository, not a subdirectory)" % gen_stage)
    for k, v in sorted(gen_counts.items()):
        print("  %-14s %d image(s)" % (k, v))
    print("  %.1f MiB total" % (sum(q.stat().st_size for q in gen_files) / 2 ** 20))

    if args.dry_run:
        print("\ndry run: nothing uploaded. Cards and citations are written into the stage.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token())
    private = bool(args.private)
    ds = "%s/%s" % (args.namespace, REPO_NAME)
    md = "%s/anny-camera-lora" % args.namespace

    api.create_repo(ds, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=ds, repo_type="dataset",
                      delete_patterns="*",
                      commit_message="Relative paths, so the records work off this machine")
    gen = "%s/%s" % (args.namespace, GENERATED_REPO)
    api.create_repo(gen, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(gen_stage), repo_id=gen, repo_type="dataset",
                      delete_patterns="*",
                      commit_message="OmniGen2 outputs, manifested apart from the renders")

    api.create_repo(md, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=md, repo_type="model",
                      delete_patterns="*",
                      commit_message="Camera-control LoRA, 304 trained tensors")

    # READ IT BACK. A push that reported success and a repository that serves the files are
    # different claims, and only the second one matters to whoever arrives next.
    for repo_id, kind in ((ds, "dataset"), (gen, "dataset"), (md, "model")):
        info = api.repo_info(repo_id, repo_type=kind)
        names = [s.rfilename for s in info.siblings]
        print("\n%s %s: %d file(s), private=%s" % (kind, repo_id, len(names), info.private))
        for required in ("README.md", "CITATION.cff"):
            print("  %s %s" % ("ok  " if required in names else "BAD ", required))
        print("  https://huggingface.co/%s%s" % ("datasets/" if kind == "dataset" else "", repo_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
