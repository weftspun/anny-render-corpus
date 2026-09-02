"""Render the Video stub: 224 frames per candidate per edit at 1 view each.

MASKSCORE.md's `## The undivided unit: 1 SpeakingFaces trial` fixes the temporal unit
at one full trial. SpeakingFaces is captured at 28 fps; a typical ~8-second command is
~224 frames. This script renders that duration at a single front-facing view per frame
so the temporal signal is dense without a 24-view multiplier that would make Video
alone dominate the corpus render budget.

Meshes are posed in-process rather than saved: 11,200 mesh npz files would cost ~7 GB
of disk for data that's a deterministic function of (edit, candidate, t). The pixi
env's `anny` import stays loaded and Mitsuba renders each frame directly from the
posed vertices, one .npz file per frame written under
build/edit/video/<edit>/<candidate>/frame_<i>.png (plus .aov.npz sidecar per RFD 1173's
Video stub schema).

The 5 candidates match the still-image spec:
  rank1 -> full target action at every t along a linear ramp from rest
  rank2 -> half-strength target ramp
  rank3 -> 30%-strength target ramp
  rank4 -> WRONG action ramp
  rank5 -> full target ramp with phenotype swap

t goes from 0 (frame 0 = rest) to 1 (frame 223 = full endpoint) linearly.

Usage:
    pixi run --environment anny-mac python render_video_stub.py \
        [--n-frames 224] [--fps 28] [--view-index 8]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

import render_view


HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = ["rank1", "rank2", "rank3", "rank4", "rank5"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir",  type=pathlib.Path, default=HERE / "build" / "edit")
    ap.add_argument("--out-dir",   type=pathlib.Path, default=HERE / "build" / "edit" / "video")
    ap.add_argument("--n-frames",  type=int, default=224,
                    help="frames per candidate; 224 = ~8s at SpeakingFaces 28fps")
    ap.add_argument("--fps",       type=int, default=28)
    ap.add_argument("--view-total", type=int, default=24,
                    help="views the sphere_hammersley pool is built against")
    ap.add_argument("--view-index", type=int, default=8,
                    help="which sphere_hammersley index to render for the video (0..view-total-1)")
    ap.add_argument("--fov",     type=float, default=40.0)
    ap.add_argument("--spp",     type=int,   default=16)
    ap.add_argument("--threads", type=int,   default=1)
    ap.add_argument("--variant", default="metal_ad_rgb")
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)

    from anny import Anny
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref",
                 facial_actions="all")
    model.eval()
    fa_labels  = model.facial_action_labels
    phen_labels = model.phenotype_labels
    faces = np.asarray(model.faces, dtype=np.int64)

    def to_action_tensor(d):
        return {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in d.items()}

    def phen_tensor(**overrides):
        return {l: torch.tensor([overrides.get(l, 0.5)], dtype=torch.float64) for l in phen_labels}

    swap_attr = manifest["phenotype_swap_attribute"]
    swap_val  = manifest["phenotype_swap_value"]
    baseline_phen = phen_tensor()
    ts = [i / (a.n_frames - 1) for i in range(a.n_frames)]

    manifest_out = {"n_frames": a.n_frames, "fps": a.fps,
                    "view_index": a.view_index, "view_total": a.view_total,
                    "edits": {}}

    tmpdir = tempfile.mkdtemp(prefix="anny_video_mesh_")
    tmp_npz = pathlib.Path(tmpdir) / "cur.npz"

    t0 = time.time()
    total_renders = 0
    for edit_name, edit_spec in manifest["edits"].items():
        target = edit_spec["target_actions"]
        wrong  = edit_spec["wrong_actions"]
        edit_out = {}
        for cand in CANDIDATES:
            cand_out = a.out_dir / edit_name / cand
            cand_out.mkdir(parents=True, exist_ok=True)

            if cand == "rank1":
                endpoint_actions, endpoint_phen = target, baseline_phen
            elif cand == "rank2":
                endpoint_actions, endpoint_phen = {k: v * 0.5 for k, v in target.items()}, baseline_phen
            elif cand == "rank3":
                endpoint_actions, endpoint_phen = {k: v * 0.3 for k, v in target.items()}, baseline_phen
            elif cand == "rank4":
                endpoint_actions, endpoint_phen = wrong, baseline_phen
            elif cand == "rank5":
                endpoint_actions, endpoint_phen = target, phen_tensor(**{swap_attr: swap_val})

            for i, t in enumerate(ts):
                fa_at_t = {k: float(v * t) for k, v in endpoint_actions.items()}
                with torch.no_grad():
                    out = model(facial_actions=to_action_tensor(fa_at_t),
                                phenotype_kwargs=endpoint_phen)
                verts = out["vertices"][0].numpy().astype(np.float64)
                np.savez_compressed(tmp_npz, verts=verts, faces=faces)
                render_view.render(
                    mesh_npz=str(tmp_npz),
                    out_png=cand_out / f"frame_{i:03d}.png",
                    index=a.view_index, views=a.view_total, fov_deg=a.fov,
                    offset=(0.0, 0.0), spp=a.spp, threads=a.threads,
                    variant=a.variant, distance=1.0, direction=None, aov=True,
                )
                total_renders += 1
            elapsed = time.time() - t0
            rate = total_renders / elapsed if elapsed > 0 else 0
            remaining = 10 * len(CANDIDATES) * a.n_frames - total_renders
            eta = remaining / rate if rate > 0 else 0
            print(f"  {edit_name}/{cand}: {a.n_frames} frames | "
                  f"elapsed {elapsed / 60:.1f}m | rate {rate:.2f}/s | eta {eta / 60:.1f}m",
                  flush=True)
            edit_out[cand] = {"n_frames": a.n_frames,
                              "endpoint_actions": endpoint_actions,
                              "endpoint_phenotype_nondefault":
                                  {swap_attr: swap_val} if cand == "rank5" else {}}
        manifest_out["edits"][edit_name] = edit_out

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"done. {total_renders} renders in {(time.time() - t0) / 60:.1f} min. "
          f"Video stub content in {a.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
