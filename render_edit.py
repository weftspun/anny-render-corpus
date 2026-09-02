"""Render frame_a + every (edit, candidate) mesh over sphere_hammersley_sequence.

Walks the manifest generate_edit_candidates.py wrote and renders every unique mesh
once. rank1 is identical to frame_b for the edit (same facial_actions vector, same
verts), so rank1 symlinks to frame_b's render dir per edit rather than double-rendering
64+ views of the same mesh.

Layout after run:
  build/edit/renders/frame_a/                  24 views of the rest pose
  build/edit/renders/<edit>/frame_b/           24 views of the edit's target
  build/edit/renders/<edit>/rank1              symlink -> ../<edit>/frame_b
  build/edit/renders/<edit>/rank2/, rank3/, rank4/, rank5/
                                                24 views of each candidate

Wall-clock at llvm_ad_rgb 1 thread, spp=16: about 7 seconds per view.
1 frame_a + 10 edits * 5 unique meshes per edit = 51 mesh batches.
51 * 24 views = 1,224 renders ~= 2.4 hours.

Usage:
    pixi run --environment anny-mac python render_edit.py [--views 24] [--spp 16]
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

import render_view


CANDIDATE_MESH_NAMES = ("frame_b", "rank2", "rank3", "rank4", "rank5")


def render_mesh(mesh_npz: pathlib.Path, out_dir: pathlib.Path, views: int, fov: float,
                spp: int, threads: int, variant: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(views):
        render_view.render(
            mesh_npz=str(mesh_npz), out_png=out_dir / f"view_{i:03d}.png",
            index=i, views=views, fov_deg=fov, offset=(0.0, 0.0),
            spp=spp, threads=threads, variant=variant, distance=1.0,
            direction=None, aov=True,
        )
    print(f"  [{mesh_npz.parent.name}/{mesh_npz.stem}] {views} views in "
          f"{time.time() - t0:6.1f}s", flush=True)


def link(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists() and dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir",   type=pathlib.Path, default=pathlib.Path("build/edit"))
    ap.add_argument("--render-dir", type=pathlib.Path, default=pathlib.Path("build/edit/renders"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", default="llvm_ad_rgb")
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())
    a.render_dir.mkdir(parents=True, exist_ok=True)

    # frame_a — one render batch, shared across every edit.
    print(f"Rendering frame_a ({a.views} views, spp={a.spp}, {a.variant} @ {a.threads})...", flush=True)
    render_mesh(a.edit_dir / "frame_a.npz", a.render_dir / "frame_a",
                a.views, a.fov, a.spp, a.threads, a.variant)

    edits = list(manifest["edits"].keys())
    total = 1 + len(edits) * len(CANDIDATE_MESH_NAMES)
    print(f"Rendering {len(edits)} edits x {len(CANDIDATE_MESH_NAMES)} unique meshes each "
          f"= {total - 1} more mesh batches, {(total - 1) * a.views} views to go", flush=True)

    for edit_name in edits:
        e_src = a.edit_dir / edit_name
        e_dst = a.render_dir / edit_name
        for mesh_stem in CANDIDATE_MESH_NAMES:
            npz = e_src / f"{mesh_stem}.npz"
            if not npz.is_file():
                raise SystemExit(f"missing mesh: {npz}")
            render_mesh(npz, e_dst / mesh_stem, a.views, a.fov, a.spp, a.threads, a.variant)
        # rank1 is identical to frame_b (same facial_actions, same phenotype); symlink.
        link(e_dst / "frame_b", e_dst / "rank1")
        print(f"  [{edit_name}] rank1 -> frame_b (symlink)", flush=True)

    print(f"done. renders under {a.render_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
