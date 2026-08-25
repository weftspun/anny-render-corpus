"""Turn a rendered camera grid into the JSONL OmniGen2's trainer reads.

WHAT THIS IS FOR, MEASURED RATHER THAN ASSUMED. `fit_ladder_azimuth.py` put a number on
OmniGen2's camera control: asked for eight azimuths in plain language, the recovered azimuth
tracked the request with a slope of **0.04**, where 1.00 is obedience and 0.00 is a body that
never turned. Six of eight views came back between -1 and -11 degrees whatever was asked. Only
the back view worked. So the model does not have the control the sweep needs, and that is
exactly the gap this corpus exists to close -- fal's Multiple-Angles LoRA was built because
Qwen-Image-Edit did not have it either, from 3000+ Gaussian-splat renders.

THE RECORD, from `data_configs/train/example/edit/jsonls/0.jsonl`:

    {"task_type": "edit", "instruction": "...",
     "input_images": ["/abs/source.png"], "output_image": "/abs/target.png"}

`input_images` is a list, so form B is the same record with the depth and pose controls added.

TWO FORMS, AND THE ABLATION BETWEEN THEM IS MEASURING COST AS MUCH AS QUALITY.

  A  [source]                    matches fal's recipe exactly
  B  [source, depth, pose]       conditioned on the target view's own controls

`omnigen2_train_dataset.py:131` picks the pixel budget by the NUMBER of inputs rather than by
position -- `self.max_input_pixels[len(input_images_path) - 1]` -- so three inputs caps all
three at 768 square where one gets 1024. Form B buys conditioning by spending resolution. At
generation time the same asymmetry was 105 s against 529 s a view. Four inputs is the ceiling:
a fifth raises IndexError inside a retry loop that silently swaps in a random other sample,
which would corrupt the training set rather than fail it.

THE SOURCE IS ONE CANONICAL VIEW, NOT EVERY PAIR. 96 views make 9,120 ordered pairs, and a
corpus that large from one subject teaches the subject rather than the camera. One fixed source
per subject is what fal's recipe implies -- an absolute target phrase only means something
relative to a known starting view.

PATHS, NOT BYTES. The loader calls `Image.open` on the path, so the images stay as files and
the JSONL references them. Every path is written absolute because the trainer's working
directory is not this one.

    python write_omnigen2_jsonl.py --grid <dir> --out <dir> --form A
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

# The trainer's own cap. A fifth input image is silently swallowed, so it is refused here.
MAX_INPUT_IMAGES = 4


def phrase_from(sidecar_path):
    """The camera phrase this frame was rendered for, read from its sidecar rather than
    re-derived from the filename. `render_grid.py` writes the prompt it used, and the sidecar
    is the record of what the pixels actually show."""
    side = json.loads(pathlib.Path(sidecar_path).read_text(encoding="utf-8"))
    for key in ("prompt", "view_prompt", "camera_prompt", "view"):
        if key in side and isinstance(side[key], str) and side[key].strip():
            return side[key].strip()
    return None


def instruction_for(phrase, form):
    base = ("Show this same person from a different camera angle: %s. Keep the person, their "
            "clothing, their body proportions and the setting exactly the same. Change only "
            "the camera." % phrase)
    if form == "A":
        return base
    return base + (" The second image is a depth map of the view you are generating and the "
                   "third image is its pose skeleton: match the body position, the limb "
                   "angles and the camera in those two images exactly.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, help="a directory of pose_*.png with sidecars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="",
                    help="the canonical source view; the front eye-level medium shot if omitted")
    ap.add_argument("--form", default="A", choices=("A", "B"))
    ap.add_argument("--val-fraction", type=float, default=0.1)
    args = ap.parse_args()

    grid = pathlib.Path(args.grid)
    frames = [pathlib.Path(p) for p in sorted(glob.glob(str(grid / "pose_*.png")))
              if not p.endswith((".depth.png", ".pose.png"))]
    if not frames:
        sys.exit("FAIL  no pose_*.png under %s" % grid)

    if args.source:
        source = pathlib.Path(args.source)
    else:
        wanted = [f for f in frames if "front-view_eye-level-shot_medium-shot" in f.name]
        if len(wanted) != 1:
            sys.exit("FAIL  %d frames match the canonical front eye-level medium shot; name "
                     "one with --source rather than letting this pick" % len(wanted))
        source = wanted[0]
    if not source.is_file():
        sys.exit("FAIL  the source view %s does not exist" % source)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, skipped = [], []
    for frame in frames:
        if frame == source:
            # THE IDENTITY PAIR IS EXCLUDED AND SAID SO. Training the model to return its
            # input for the source view's own phrase teaches it to copy, which is the failure
            # the camera control is supposed to avoid.
            skipped.append((frame.name, "the source view itself"))
            continue
        sidecar = frame.with_suffix(".json")
        if not sidecar.is_file():
            skipped.append((frame.name, "no sidecar, so the camera phrase is unknown"))
            continue
        phrase = phrase_from(sidecar)
        if not phrase:
            skipped.append((frame.name, "the sidecar carries no camera phrase"))
            continue

        inputs = [source]
        if args.form == "B":
            depth = frame.with_name(frame.stem + ".depth.png")
            pose = frame.with_name(frame.stem + ".pose.png")
            if not depth.is_file() or not pose.is_file():
                skipped.append((frame.name, "form B needs .depth.png and .pose.png; run "
                                            "make_controls.py over this grid first"))
                continue
            inputs = [source, depth, pose]
        if len(inputs) > MAX_INPUT_IMAGES:
            sys.exit("FAIL  %d input images; the loader indexes its pixel budget by input "
                     "count and a fifth raises IndexError into a silent retry" % len(inputs))

        records.append({
            "task_type": "edit",
            "instruction": instruction_for(phrase, args.form),
            "input_images": [str(p.resolve()) for p in inputs],
            "output_image": str(frame.resolve()),
        })

    if not records:
        sys.exit("FAIL  no records; %d frame(s) were skipped and the reasons are above"
                 % len(skipped))

    # SPLIT BY TARGET VIEW, DETERMINISTICALLY. Every record shares one source image, so a
    # random split would put near-identical views on both sides. Taking every Nth target in
    # the grid's own order spreads the holdout across azimuth, elevation and distance instead
    # of clustering it in one corner of the sphere.
    stride = max(int(round(1.0 / args.val_fraction)), 2) if args.val_fraction > 0 else 0
    train, val = [], []
    for i, record in enumerate(records):
        (val if stride and i % stride == 0 else train).append(record)

    for name, rows in (("train", train), ("val", val)):
        if not rows:
            continue
        path = out / ("%s_form%s.jsonl" % (name, args.form))
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print("%-5s %4d records -> %s" % (name, len(rows), path))

    mix = out / ("mix_form%s.yml" % args.form)
    edit = out / ("edit_form%s.yml" % args.form)
    with open(edit, "w", encoding="utf-8") as fh:
        fh.write("data:\n  - \n    path: '%s'\n    type: 'edit'\n    ratio: !!float 1.0\n"
                 % (out / ("train_form%s.jsonl" % args.form)).resolve().as_posix())
    with open(mix, "w", encoding="utf-8") as fh:
        # `ratio` IS INERT UPSTREAM and this file says so rather than relying on it.
        # `_collect_annotations` reads a leaked loop variable and discards the `.select()`
        # result, so mixing is controlled by row counts, not by this number.
        fh.write("# ratio is inert in this trainer; the mix is controlled by row counts.\n"
                 "data:\n  - \n    path: '%s'\n    type: 'edit'\n    ratio: !!float 1.0\n"
                 % edit.resolve().as_posix())
    print("wrote %s and %s" % (edit, mix))

    print("source view: %s" % source.name)
    print("%d record(s), %d skipped" % (len(records), len(skipped)))
    for name, why in skipped[:6]:
        print("  skipped %-52s %s" % (name, why))
    if len(skipped) > 6:
        print("  ... and %d more" % (len(skipped) - 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
