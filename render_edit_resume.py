"""Resume-safe wrapper around render_edit.render_mesh — skips complete batches.

render_edit renders in-process across all meshes. If the process dies mid-run
(a Metal driver hiccup, an OS kill, an accidental Ctrl-C), a partially written
mesh directory looks the same as a completed one. This script inspects each
output dir and re-renders only the ones missing view files or below the expected
count.

Usage:
    pixi run --environment anny-mac python render_edit_resume.py [--views 24] [--variant metal_ad_rgb]
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

import render_view
import render_edit


def is_complete(dir_path: pathlib.Path, views: int) -> bool:
    """Directory has exactly `views` PNGs + `views` JSONs + `views` AOV npzs."""
    if not dir_path.is_dir():
        return False
    n_png = len(list(dir_path.glob("view_*.png")))
    n_json = len(list(dir_path.glob("view_*.json")))
    n_aov = len(list(dir_path.glob("view_*.aov.npz")))
    # sidecar json count double-includes .keypoints.json when present
    n_sidecar = sum(1 for p in dir_path.glob("view_*.json") if ".keypoints" not in p.name)
    return n_png == views and n_sidecar == views and n_aov == views


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir",   type=pathlib.Path, default=pathlib.Path("build/edit"))
    ap.add_argument("--render-dir", type=pathlib.Path, default=pathlib.Path("build/edit/renders"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", default="metal_ad_rgb")
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())
    a.render_dir.mkdir(parents=True, exist_ok=True)

    # frame_a
    fa_out = a.render_dir / "frame_a"
    if not is_complete(fa_out, a.views):
        print(f"rendering frame_a ({a.views} views)...", flush=True)
        if fa_out.exists() and not fa_out.is_symlink():
            shutil.rmtree(fa_out)
        render_edit.render_mesh(a.edit_dir / "frame_a.npz", fa_out,
                                a.views, a.fov, a.spp, a.threads, a.variant)
    else:
        print(f"frame_a: complete, skipping", flush=True)

    for edit_name in manifest["edits"]:
        e_src = a.edit_dir / edit_name
        e_dst = a.render_dir / edit_name
        for mesh_stem in render_edit.CANDIDATE_MESH_NAMES:
            out = e_dst / mesh_stem
            if is_complete(out, a.views):
                continue
            if out.exists():
                shutil.rmtree(out)
            npz = e_src / f"{mesh_stem}.npz"
            print(f"rendering {edit_name}/{mesh_stem} ({a.views} views)...", flush=True)
            render_edit.render_mesh(npz, out, a.views, a.fov, a.spp, a.threads, a.variant)
        # (Re)establish rank1 symlink.
        rank1 = e_dst / "rank1"
        if rank1.is_symlink():
            rank1.unlink()
        elif rank1.exists():
            shutil.rmtree(rank1)
        rank1.symlink_to((e_dst / "frame_b").resolve(), target_is_directory=True)

    print(f"resume done under {a.render_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
