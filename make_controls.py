"""Write the two conditioning images for a rendered frame: a depth map and a pose skeleton.

Both come from the frame's own sidecar camera and the rig we posed, so they are exact rather
than estimated. `check_controls.py` measures their agreement with the render in pixels; this
writes them out so a generator, or a vision model, can be handed all three.

    python make_controls.py <frame.png> --rig <rest-skel.npz>
"""
from __future__ import annotations

import argparse
import colorsys
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw

import check_controls
import render_corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame")
    ap.add_argument("--rig", required=True)
    ap.add_argument("--colours", default="", help="anny-keypoint-colours.json, if it is to hand")
    args = ap.parse_args()

    verts, faces, joints, names = check_controls.load_rig(args.rig)
    c = check_controls.controls_for(args.frame, verts, faces, joints)
    frame = pathlib.Path(args.frame)

    png, (near, far) = render_corpus.depth_png(c["zbuf"], c["mask"])
    depth_path = frame.with_name(frame.stem + ".depth.png")
    png.save(depth_path)

    # The skeleton on a dark ground, in the scheme's colours when they are to hand. A joint
    # is drawn whether or not it is occluded, which is what a pose control carries.
    parents = np.load(args.rig)["parents"]
    srgb = {}
    if args.colours and pathlib.Path(args.colours).is_file():
        table = json.loads(pathlib.Path(args.colours).read_text(encoding="utf-8"))
        srgb = {k["name"]: tuple(k["srgb"]) for k in table["keypoints"]}
    pose = Image.new("RGB", c["size"], (0, 0, 0))
    draw = ImageDraw.Draw(pose)
    px = c["joints_px"]
    for i, parent in enumerate(parents):
        colour = srgb.get(names[i] if i < len(names) else "", None)
        if colour is None:
            r, g, b = colorsys.hls_to_rgb((i * 0.618) % 1.0, 0.6, 0.9)
            colour = (int(r * 255), int(g * 255), int(b * 255))
        if parent >= 0:
            draw.line([tuple(px[parent]), tuple(px[i])], fill=colour, width=4)
    for i, (x, y) in enumerate(px):
        colour = srgb.get(names[i] if i < len(names) else "", (255, 255, 255))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
    pose_path = frame.with_name(frame.stem + ".pose.png")
    pose.save(pose_path)

    print(f"depth {depth_path.name}  near {near:.4f} far {far:.4f}")
    print(f"pose  {pose_path.name}  {len(px)} joints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
