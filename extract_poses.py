"""Extract world-space joint positions from retargeted motion clips.

Feeds the `poses` / `pose_rotations` relations of the ANNY render corpus.

Why positions and not rotations: three times this session a name-level mapping
looked right and was wrong (LabRCSF name-coverage vs silhouette; the ANNY
finger chain; Godot's twist math). Source clips use a BVH-flavoured 22-bone
skeleton (Hips, Chest..Chest4, Collar/Shoulder/Elbow/Wrist, Hip/Knee/Ankle/Toe)
whose rest orientations do not match ANNY's, so copying local rotations across
reproduces the candy-wrapper failure. World-space JOINT POSITIONS are
convention-free: whatever the source rest pose is, the wrist is where the wrist
is. The fitter (AnnyInverter / LBFGS) then solves for ANNY rotations that put
its joints at those positions -- the same skin-matcher approach that reached
1.7e-4 mm on a same-rig fit.

Source clips: dataset-100style-godot-clips (CC-BY-4.0, 302 glb),
dataset-o3de-motion-matching-clips (Apache-2.0/MIT), and
dataset-vr-balance-disturbance (CC-BY-4.0) -- license lineage is carried into
`motion_clips.dataset_id` so attribution survives into the trained artifact.

Usage:
  python extract_poses.py --clips <dir-of-glb> --out positions.parquet \
      [--stride 15] [--limit 10]
"""

import argparse
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pygltflib import GLTF2

# Source bone -> the ANNY bone whose position it constrains. Semantic, and
# deliberately body-only: the source skeleton has no fingers, which sidesteps
# the chained-joint problem entirely.
BONE_MAP = {
    "Hips": "root",
    "Chest": "spine05", "Chest2": "spine04", "Chest3": "spine03", "Chest4": "spine02",
    "Neck": "neck01", "Head": "head",
    "LeftCollar": "clavicle.L", "LeftShoulder": "upperarm01.L",
    "LeftElbow": "lowerarm01.L", "LeftWrist": "wrist.L",
    "RightCollar": "clavicle.R", "RightShoulder": "upperarm01.R",
    "RightElbow": "lowerarm01.R", "RightWrist": "wrist.R",
    "LeftHip": "upperleg01.L", "LeftKnee": "lowerleg01.L", "LeftAnkle": "foot.L",
    "RightHip": "upperleg01.R", "RightKnee": "lowerleg01.R", "RightAnkle": "foot.R",
}

ACC_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
             5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(g: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = g.accessors[index]
    view = g.bufferViews[acc.bufferView]
    dtype = ACC_DTYPE[acc.componentType]
    n = NCOMP[acc.type]
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count * n
    arr = np.frombuffer(blob, dtype=dtype, count=count, offset=start)
    return arr.reshape(acc.count, n) if n > 1 else arr


def trs_matrix(t, r, s) -> np.ndarray:
    x, y, z, w = r
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * np.asarray(s)[None, :]
    m[:3, 3] = t
    return m


def sample_clip(path: str, stride: int = 15):
    """Returns (bone_names, frame_times, positions[F, B, 3]) in world space."""
    g = GLTF2().load(path)
    blob = g.binary_blob()
    if not g.animations:
        return None
    anim = g.animations[0]
    names = [n.name for n in g.nodes]

    # local TRS per node, animated where a channel exists
    base_t = {i: np.array(n.translation or [0, 0, 0], float) for i, n in enumerate(g.nodes)}
    base_r = {i: np.array(n.rotation or [0, 0, 0, 1], float) for i, n in enumerate(g.nodes)}
    base_s = {i: np.array(n.scale or [1, 1, 1], float) for i, n in enumerate(g.nodes)}

    tracks = {}
    times = None
    for ch in anim.channels:
        smp = anim.samplers[ch.sampler]
        t_in = read_accessor(g, blob, smp.input).astype(float)
        v_out = read_accessor(g, blob, smp.output).astype(float)
        tracks[(ch.target.node, ch.target.path)] = (t_in, v_out)
        if times is None or len(t_in) > len(times):
            times = t_in

    parent = {}
    for i, n in enumerate(g.nodes):
        for c in (n.children or []):
            parent[c] = i

    frames = list(range(0, len(times), max(1, stride)))
    out = np.zeros((len(frames), len(g.nodes), 3))
    for fi, f in enumerate(frames):
        t_now = times[f]
        world = {}

        def resolve(i):
            if i in world:
                return world[i]
            t, r, s = base_t[i], base_r[i], base_s[i]
            for path_name, base in (("translation", "t"), ("rotation", "r"), ("scale", "s")):
                key = (i, path_name)
                if key in tracks:
                    t_in, v_out = tracks[key]
                    # nearest-sample lookup: clips are dense (>=30fps) and we
                    # subsample anyway, so interpolation would add no signal
                    j = int(np.searchsorted(t_in, t_now).clip(0, len(t_in) - 1))
                    if base == "t":
                        t = v_out[j]
                    elif base == "r":
                        r = v_out[j]
                    else:
                        s = v_out[j]
            local = trs_matrix(t, r, s)
            world[i] = (resolve(parent[i]) @ local) if i in parent else local
            return world[i]

        for i in range(len(g.nodes)):
            out[fi, i] = resolve(i)[:3, 3]

    return names, [float(times[f]) for f in frames], out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", default="", help="directory of .glb (see --file-list)")
    parser.add_argument("--file-list", default="",
                        help="text file of .glb paths, one per line. Needed on the O: remote "
                             "drive: os.listdir/scandir/pathlib all raise WinError -2146893818 "
                             "'Invalid Signature' there while direct open() works fine, so "
                             "Python cannot enumerate that folder -- produce the list with the "
                             "shell instead (ls .../*.glb > list.txt).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=15, help="frames between samples")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.file_list:
        files = [l.strip() for l in open(args.file_list) if l.strip()]
    else:
        files = sorted(glob.glob(os.path.join(args.clips, "*.glb")))
    if not files:
        raise SystemExit("no clips found; on the O: drive use --file-list (see --help)")
    if args.limit:
        files = files[:args.limit]

    clip_names, frame_idx, bone_names, xs, ys, zs = [], [], [], [], [], []
    for path in files:
        got = sample_clip(path, args.stride)
        if not got:
            continue
        names, times, pos = got
        clip = os.path.splitext(os.path.basename(path))[0]
        for fi, _t in enumerate(times):
            for bi, bname in enumerate(names):
                anny_bone = BONE_MAP.get(bname)
                if anny_bone is None:
                    continue  # root node / unmapped: not a constraint
                clip_names.append(clip); frame_idx.append(fi)
                bone_names.append(anny_bone)
                xs.append(pos[fi, bi, 0]); ys.append(pos[fi, bi, 1]); zs.append(pos[fi, bi, 2])
        print(f"  {clip}: {len(times)} frames")

    table = pa.table({
        "clip_name": pa.array(clip_names),
        "frame_index": pa.array(frame_idx, pa.int32()),
        "anny_bone": pa.array(bone_names),
        "x": pa.array(xs, pa.float32()),
        "y": pa.array(ys, pa.float32()),
        "z": pa.array(zs, pa.float32()),
    })
    pq.write_table(table, args.out, compression="zstd")
    print(f"{table.num_rows} joint-position rows from {len(set(clip_names))} clips -> {args.out}")


if __name__ == "__main__":
    main()
