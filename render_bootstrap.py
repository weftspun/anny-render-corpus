"""Render every bootstrap mesh over the sphere_hammersley_sequence for MaskScore Rung 1.

Runs `render_view.render(...)` in one process per mesh so Mitsuba variant init and Python
startup amortize over the 64 views instead of paying them per shot. Emits one PNG + AOV
npz + JSON sidecar per view under `build/bootstrap/<mesh>/view_<index>.<ext>`.

The input mesh (rest) and the rank1 candidate are identical by construction, so the
renders under `input/` and `rank1/` symlink to the same files rather than double-rendering
128 identical frames.

Usage:
    pixi run --environment anny-mac python render_bootstrap.py [--views 64] [--spp 16]
"""

import argparse
import pathlib
import shutil
import sys
import time

import numpy as np

import render_view


def render_mesh(mesh_npz: pathlib.Path, out_dir: pathlib.Path, views: int, fov: float,
                spp: int, threads: int, variant: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sides = []
    t0 = time.time()
    for i in range(views):
        out_png = out_dir / f"view_{i:03d}.png"
        side = render_view.render(
            mesh_npz=str(mesh_npz), out_png=out_png,
            index=i, views=views, fov_deg=fov, offset=(0.0, 0.0),
            spp=spp, threads=threads, variant=variant, distance=1.0,
            direction=None, aov=True,
        )
        sides.append(side)
        if i == 0 or (i + 1) % 8 == 0 or i == views - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (views - i - 1) / rate if rate > 0 else 0
            print(f"  [{mesh_npz.stem}] view {i+1:3d}/{views}  "
                  f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s")
    return sides


def link_dir(src: pathlib.Path, dst: pathlib.Path):
    if dst.exists() and dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", type=pathlib.Path, default=pathlib.Path("build/bootstrap"))
    ap.add_argument("--render-dir", type=pathlib.Path,
                    default=pathlib.Path("build/bootstrap/renders"))
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", default="llvm_ad_rgb")
    a = ap.parse_args(argv[1:])

    rest_npz = a.pose_dir / "rest.npz"
    rank5_npz = a.pose_dir / "rank5.npz"

    print(f"Rendering rest ({a.views} views at spp={a.spp}, {a.variant} @ {a.threads} thread)...")
    render_mesh(rest_npz, a.render_dir / "input", a.views, a.fov, a.spp, a.threads, a.variant)

    # rank1 == rest by construction. Symlink so downstream scoring can address them
    # under different candidate directories without doubling the render bill.
    link_dir(a.render_dir / "input", a.render_dir / "rank1")
    print(f"  rank1 -> input (symlink; rank1 is identity by construction)")

    print(f"Rendering rank5 ({a.views} views at spp={a.spp})...")
    render_mesh(rank5_npz, a.render_dir / "rank5", a.views, a.fov, a.spp, a.threads, a.variant)

    print(f"done. renders under {a.render_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
