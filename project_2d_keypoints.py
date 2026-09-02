"""Project the 23 ANNY-COCO keypoints to 2D per rendered view.

For each view sidecar under a render directory, read the camera setup (eye, fov, and the
mesh-normalisation centre/scale that render_view.py wrote), regress the 23 keypoints from
the mesh vertices via `KeypointsRegressor.coco`, apply the same normalisation the render
did, then project through the pinhole camera that Mitsuba's `look_at` describes.

Matches Mitsuba's conventions (rendered in `render_view.py`):
  target = [0, 0, 0], up = [0, 0, 1] (Z-up), fov_axis = 'y', film = 1024 x 1024.

Emits one JSON per view alongside the render: {keypoint_label: [u, v, in_frame_flag]}.
These are the training labels for rf-detr-keypoint's face-model bootstrap.

Usage:
    pixi run --environment anny-mac python project_2d_keypoints.py <mesh.npz> <renders_dir>
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from anny import Anny
from anny.keypoints import KeypointsRegressor


def project_pinhole(kp3d_world: np.ndarray, eye: np.ndarray, fov_deg: float,
                    width: int = 1024, height: int = 1024) -> np.ndarray:
    """World-space keypoints to pixel coordinates via a look-at-origin pinhole camera.

    Assumes target=(0,0,0), up=(0,0,1), fov_axis='y' — the same setup render_view.py builds.
    Returns (K, 3): (u, v, in_frame) where in_frame is 1.0 if the point is in the image and
    in front of the camera, else 0.0.
    """
    forward = -eye / np.linalg.norm(eye)              # from eye toward origin
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # World-to-camera rotation and translation. Camera looks along -Z in its own frame.
    R = np.stack([right, up, -forward], axis=0)
    t = -R @ eye
    kp_cam = kp3d_world @ R.T + t

    # Perspective divide. Camera looks down -Z, so z_cam is negative in front of camera.
    z = kp_cam[:, 2]
    in_front = z < 0
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)       # focal length in normalised units
    aspect = width / height
    x_ndc = f * kp_cam[:, 0] / (-z + 1e-9) / aspect
    y_ndc = f * kp_cam[:, 1] / (-z + 1e-9)
    u = (x_ndc + 1.0) * 0.5 * width
    v = (1.0 - (y_ndc + 1.0) * 0.5) * height          # image v grows down
    in_frame = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v, in_frame.astype(np.float64)], axis=-1)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_npz", type=pathlib.Path,
                    help="the mesh whose pose supplies the keypoints (rest.npz or rank5.npz)")
    ap.add_argument("renders_dir", type=pathlib.Path,
                    help="dir with view_XXX.json sidecars from render_view.py --aov")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    a = ap.parse_args(argv[1:])

    # REGRESS THE 23 KEYPOINTS FROM THE POSED MESH. Same rig+topology the renders were
    # built from; the mesh.npz's verts are already the posed vertices, and the regressor
    # applies a linear blend of them to reach each keypoint.
    data = np.load(a.mesh_npz)
    verts = np.asarray(data["verts"], dtype=np.float64)

    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref")
    kp = KeypointsRegressor.coco(model)
    with torch.no_grad():
        kps_3d = kp({"vertices": torch.from_numpy(verts[None])})[0].numpy()

    written = 0
    for sidecar in sorted(a.renders_dir.glob("view_*.json")):
        s = json.loads(sidecar.read_text())
        centre = np.array(s["normalisation"]["centre"], dtype=np.float64)
        scale = float(s["normalisation"]["scale"])
        eye = np.array(s["eye"], dtype=np.float64)
        fov = float(s["fov_deg"])
        # Apply the same normalisation the render did — else the keypoint sits in a
        # different world scale than the mesh Mitsuba drew.
        kps_norm = (kps_3d - centre) * scale
        pixel = project_pinhole(kps_norm, eye, fov, a.width, a.height)

        out = {label: {"u": float(px[0]), "v": float(px[1]), "in_frame": bool(px[2])}
               for label, px in zip(kp.labels, pixel)}
        out_json = sidecar.with_suffix(".keypoints.json")
        out_json.write_text(json.dumps({"view": sidecar.stem,
                                        "yaw_deg": s["yaw_deg"],
                                        "pitch_deg": s["pitch_deg"],
                                        "keypoints": out}, indent=2))
        written += 1

    n_in_frame = sum(1 for label in kp.labels
                     if any(bool(json.loads(p.read_text())["keypoints"][label]["in_frame"])
                            for p in a.renders_dir.glob("view_*.keypoints.json")))
    print(f"  wrote {written} keypoint sidecars; {n_in_frame}/{len(kp.labels)} labels "
          f"in frame in at least one view")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
