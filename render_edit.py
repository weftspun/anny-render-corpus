"""Render frame B + 5 edit candidates over the sphere_hammersley_sequence.

Frame A is symlinked to the existing rest renders (build/bootstrap/renders/input) --
no double-render for an unchanged mesh. Rank1 == frame_b by construction (same actions
= same verts); it is symlinked once frame_b's renders complete. Frame B and rank2/3/4/5
each get their own 64-view batch with depth + normal AOVs.

Wall-clock at llvm_ad_rgb 1 thread, spp=16: ~7 s/view -> ~35 min total for 5 mesh
batches x 64 views.

Usage:
    pixi run --environment anny-mac python render_edit.py [--views 64] [--spp 16]
"""

import argparse
import pathlib
import shutil
import sys
import time

import render_view


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
        if i == 0 or (i + 1) % 16 == 0 or i == views - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (views - i - 1) / rate if rate > 0 else 0
            print(f"  [{mesh_npz.stem}] view {i+1:3d}/{views}  "
                  f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s", flush=True)


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
    ap.add_argument("--rest-renders", type=pathlib.Path,
                    default=pathlib.Path("build/bootstrap/renders/input"),
                    help="existing frame-a renders to symlink instead of re-rendering")
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", default="llvm_ad_rgb")
    a = ap.parse_args(argv[1:])

    a.render_dir.mkdir(parents=True, exist_ok=True)

    # RENDER EVERY MESH FRESH. sphere_hammersley(i, n) depends on n: view 0 of a
    # 64-view sequence is not view 0 of a 100-view sequence -- symlinking earlier
    # renders across --views would mis-align frame A against the candidates and any
    # score would be measuring the sampler difference, not the edit.
    to_render = ["frame_a", "frame_b", "rank2", "rank3", "rank4", "rank5"]
    for name in to_render:
        npz = a.edit_dir / f"{name}.npz"
        if not npz.is_file():
            raise SystemExit(f"missing mesh: {npz}")
        print(f"Rendering {name} ({a.views} views at spp={a.spp}, {a.variant} @ {a.threads} thread)...",
              flush=True)
        render_mesh(npz, a.render_dir / name, a.views, a.fov, a.spp, a.threads, a.variant)

    link(a.render_dir / "frame_b", a.render_dir / "rank1")
    print(f"  rank1 -> frame_b (symlink; rank1 == frame_b by construction)")

    print(f"done. renders under {a.render_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
