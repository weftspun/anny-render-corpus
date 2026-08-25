"""Render one mesh through a manifest of explicit camera poses.

THIS IS NOT THE SEQUENCE AND DOES NOT PRETEND TO BE. `render_view.py` walks
`sphere_hammersley_sequence`, which covers a sphere without any view being chosen, and that
is what a measurement wants. This walks a list of poses somebody wrote down, because a
labelled pair wants a known camera beside the words for it. Asking the sequence to reproduce
a grid costs 1536 renders to land 32 cells within 5 degrees and never lands on one.

The poses arrive as a manifest rather than being defined here. The vocabulary that names them
lives in `service-livebook/priv/python/weft_loop.py`, and a second copy of a phrase table is a
copy that drifts. `weft_loop.grid_cameras` writes the manifest, this reads it, and the file
between them is also the provenance record.

    python render_grid.py <mesh.npz> <manifest.json> <out_dir> [--variant cuda_ad_rgb]

Each frame writes its own sidecar, as `render_view` already does, plus the pose's prompt. A
frame whose sidecar says `explicit direction, not the sequence` came from here.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import time

import numpy as np

import check_conventions
import render_view

TRIGGER = "<sks>"


def slug(prompt: str) -> str:
    """The prompt as a filename, keeping the words and dropping the trigger token.

    THE NUMBER ALONE IS NOT A NAME. `pose_042.png` says nothing about what it shows, so a
    person picking a frame out of ninety-six has to hold a manifest open beside them, and a
    frame that gets copied somewhere else stops meaning anything at all. The pose is known
    exactly, so the file says it: `pose_042_right-side-view_eye-level-shot_medium-shot.png`.

    The index stays in front of the name. It is what orders the sweep, and the encoder reads
    frames in filename order.
    """
    words = prompt.replace(TRIGGER, "").strip()
    parts = []
    # The three phrases are separated by their own ends: an azimuth phrase ends in "view",
    # an elevation phrase in "shot", and a distance phrase is one of three known words.
    tokens = words.split()
    current = []
    for token in tokens:
        current.append(token)
        if token in ("view", "shot"):
            parts.append("-".join(current))
            current = []
    if current:
        parts.append("-".join(current))
    return "_".join(parts)


def preflight(args) -> int:
    """Scale, up, forward and handedness, before any frame is rendered.

    A rig with the wrong up axis or a mirrored side renders ninety-six frames that look
    plausible and carry wrong labels, which happened twice here. The checks cost
    milliseconds and the render costs forty-five seconds.

    The mesh npz carries vertices and faces, which is enough for up and scale. Forward and
    handedness need the joints, so they run only when --rig names the .usda. An unrun check
    is reported rather than passed over.
    """
    data = np.load(args.mesh_npz)
    if args.rig:
        rig = check_conventions.read_rig(args.rig)
        rig["posed"] = args.posed
        problems, m = check_conventions.check(rig)
        check_conventions.report(pathlib.Path(args.rig).name, problems, m)
        facing = math.degrees(math.atan2(m["forward"][1], m["forward"][0])) % 360
        if abs((facing - args.facing_deg + 180) % 360 - 180) > 1.0:
            problems.append(f"--facing-deg {args.facing_deg} against a rig that faces "
                            f"{facing:.1f} deg, so every phrase is wrong by the difference")
    else:
        points = data["verts"]
        extent = points.max(axis=0) - points.min(axis=0)
        order = np.argsort(extent)[::-1]
        margin = float(extent[order[0]] / max(extent[order[1]], 1e-12))
        stature = float(extent[order[0]])
        problems = []
        if margin < check_conventions.UP_MARGIN:
            problems.append(f"up axis is not clear: {np.round(extent, 3).tolist()}")
        lo, hi = check_conventions.STATURE_M
        if not lo <= stature <= hi:
            problems.append(f"stature {stature:.3f} m is outside {(lo, hi)}")
        print(f"conventions: up {'XYZ'[int(order[0])]} at {margin:.2f}x, "
              f"stature {stature:.3f} m")
        print("conventions NOT RUN: forward and handedness need --rig <rest-skel.npz>")
    for problem in problems:
        print(f"  BAD  {problem}")
    if problems:
        print(f"{len(problems)} convention problem(s); nothing rendered")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh_npz")
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N poses")
    # THE TITLE HAS TO NAME THE SUBJECT OR TWO SETS COLLIDE. The first version built the
    # title from the pose count alone, so a second subject through the same grid produced
    # `anny-camera-grid-96-poses-of-one-subject.cff` a second time, and the second would
    # have overwritten the first in any directory that held both.
    ap.add_argument("--subject", default="",
                    help="what was rendered; defaults to the mesh filename stem")
    # THE PHRASE IS ABOUT THE SUBJECT'S FRONT AND THE CAMERA IS PLACED IN WORLD AZIMUTH.
    # Those are the same number only for a rig that happens to face world zero. ANNY faces
    # 270 degrees, measured two ways that agree exactly: the eyes sit ahead of the head and
    # the toes ahead of the ankles, both at 270.0. Without this offset every frame the grid
    # called a front view was a profile, and the label was wrong while the pixels were fine.
    ap.add_argument("--facing-deg", type=float, default=0.0,
                    help="world azimuth the subject faces; the grid is rotated by it")
    ap.add_argument("--rig", default="",
                    help="the subject's rest-skel .npz; enables the forward and handedness checks")
    # A POSED RIG IS DECLARED, NOT DETECTED. The witness-agreement check is about an
    # unposed rig: on a posed one a turned head or a planted foot is a stance. Everything
    # else still runs, so a mirrored or mis-scaled posed rig is still refused.
    ap.add_argument("--posed", action="store_true",
                    help="the rig carries a pose, so gaze and feet may differ from the chest")
    ap.add_argument("--skip-conventions", action="store_true",
                    help="render without checking scale, up, forward and handedness")
    args = ap.parse_args()

    if not args.skip_conventions:
        rc = preflight(args)
        if rc:
            return rc

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    poses = manifest["poses"][: args.limit or None]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    for pose in poses:
        out_png = out_dir / f"pose_{pose['index']:03d}_{slug(pose['prompt'])}.png"
        side = render_view.render(
            args.mesh_npz, out_png, 0, len(poses), args.fov, (0.0, 0.0),
            args.spp, args.threads, args.variant,
            distance=pose["distance_factor"],
            direction=((pose["azimuth_deg"] + args.facing_deg) % 360.0, pose["elevation_deg"]),
        )
        # The prompt travels with the frame. A render whose label lives only in a manifest
        # somewhere else is a render nobody can use six months from now.
        sidecar = out_png.with_suffix(".json")
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["prompt"] = pose["prompt"]
        record["pose_index"] = pose["index"]
        # Both azimuths travel: what the phrase claims about the subject, and where the
        # camera actually stood. A frame carrying only one of them cannot be checked.
        record["subject_azimuth_deg"] = pose["azimuth_deg"]
        record["subject_facing_deg"] = args.facing_deg
        sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"  {pose['index']:>3}/{len(poses)}  {pose['prompt']:<52}  "
              f"radius {side['radius']:.3f}  {side['sha256'][:12]}...")

    # THE SET GETS A .cff, WHICH IS WHAT THE DELIVERABLES RULE ASKS FOR. A directory of PNGs
    # with no title, no licence and no statement of what made them is not a deliverable, it
    # is a scratch folder somebody will delete. The per-frame provenance is already in each
    # sidecar; this names the set.
    subject = args.subject or pathlib.Path(args.mesh_npz).stem
    title = f"ANNY camera grid: {len(poses)} poses of {subject}"
    citation = [
        "cff-version: 1.2.0",
        'message: "If you use these renders, please cite them as below."',
        f'title: "{title}"',
        "abstract: >-",
        f"  {len(poses)} renders of {subject}, one ANNY mesh, through an enumerated camera grid:",
        "  8 azimuths at 45 degree steps, 4 elevations at -30, 0, 30 and 60 degrees, and 3",
        "  distances at 0.6, 1.0 and 1.8 times the derived radius. Every pose is named in the",
        "  phrase table fal's Multiple-Angles LoRA is conditioned on, and the name is in the",
        "  filename as well as the sidecar, so a frame that travels keeps its label.",
        "",
        "  Constructed synthetic, not generated: rendered deterministically from a mesh we",
        "  hold, so the labels are true by construction and the same inputs reproduce the set.",
        "  This is NOT the sphere_hammersley_sequence, which covers a sphere without a view",
        "  being chosen and is what a measurement uses. Each frame's sidecar says which",
        "  generator produced it.",
        "",
        f"  Renderer: mitsuba, variant {args.variant}, {args.spp} spp, {args.fov} degree field",
        f"  of view. Mesh: {pathlib.Path(args.mesh_npz).name}. Every frame carries a sha256.",
        "license:",
        "  - Apache-2.0",
        "  - MIT",
        "authors:",
        "  - family-names: Lee",
        '    given-names: "K. S. Ernest (iFire)"',
        '    orcid: "https://orcid.org/0000-0003-4570-1436"',
        "keywords:",
        "  - camera-grid",
        "  - constructed-synthetic",
        "  - pose-labels",
        "  - reproducibility",
        "references:",
        "  - type: software",
        '    title: "Qwen-Image-Edit-2511-Multiple-Angles-LoRA"',
        "    authors:",
        "      - family-names: Odin",
        "        given-names: Lovis",
        '      - name: "fal"',
        '    url: "https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"',
        "    license: Apache-2.0",
        "    abstract: >-",
        "      The phrase table these poses are named in. Only the text is used: the model is",
        "      blocklisted here and those weights are gated, so neither is loaded.",
        "",
    ]
    # THE CITATION FILE IS NAMED AFTER ITS OWN TITLE, NOT "CITATION". The set is delivered
    # as a clip beside it, and a deliverable whose filename and whose stated title disagree
    # makes a reader ask which one is the asset. One stem, two extensions: `<title>.cff`
    # beside `<title>.mkv`.
    stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    (out_dir / f"{stem}.cff").write_text(chr(10).join(citation), encoding="utf-8")
    print(f"citation: {stem}.cff  (the clip belongs beside it as {stem}.mkv)")

    seconds = time.monotonic() - started
    print(f"{len(poses)} frames in {seconds:.1f} s, {len(poses) / seconds:.1f} fps, "
          f"variant {args.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
