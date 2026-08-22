"""BVH reader + forward kinematics, with no retargeting opinions baked in.

Kept separate from the retarget deliberately. Turning 100STYLE into corpus poses is two
jobs -- read the file, then map its skeleton onto ANNY's -- and the second is where this
project has been burned before (the finger-chain convention failure, where a per-joint
Euler mapping verified on arms and legs was never independently verified on fingers and
compounded down the chain). Keeping the reader honest and testable on its own means a
retarget bug cannot hide inside a parsing bug.

WHAT BVH ACTUALLY SAYS, and the traps in it:

  * Channel ORDER is per-joint and declared, not global. 100STYLE writes
    `Yrotation Xrotation Zrotation`, so the three numbers on a motion line are Y, X, Z --
    reading them positionally as X, Y, Z silently transposes every rotation. This parser
    stores rotations by NAME from the CHANNELS declaration and never by position.
  * Rotation order is INTRINSIC and follows the channel order: R = Ry @ Rx @ Rz for this
    file. That is a property of the file, so it is parsed, not assumed.
  * OFFSET is the bone's rest translation from its parent, in the file's own units
    (100STYLE is centimetres). Only the root carries per-frame translation.
  * `End Site` blocks are leaf tips with an offset and NO channels. They must be skipped
    when consuming motion values or every channel after the first leaf is misaligned.

Units and axes are reported, never assumed -- the SOMA centimetre/metre and Y-up/Z-up
confusion cost this project a retracted finding already.
"""

import numpy as np


class Bvh:
    """Parsed BVH: skeleton in `joints`, motion in `frames`."""

    def __init__(self, names, parents, offsets, channels, rot_orders,
                 frames, frame_time, root_translation):
        self.names = names                    # (J,) joint names, hierarchy order
        self.parents = parents                # (J,) parent index, -1 for root
        self.offsets = offsets                # (J,3) rest offset from parent
        self.channels = channels              # list of per-joint channel-name lists
        self.rot_orders = rot_orders          # (J,) e.g. "yxz", from the CHANNELS line
        self.frames = frames                  # (F,J,3) Euler degrees, in rot_order
        self.frame_time = frame_time          # seconds
        self.root_translation = root_translation   # (F,3)

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def fps(self):
        return 1.0 / self.frame_time if self.frame_time > 0 else 0.0


def parse(path):
    with open(path, "r") as fh:
        text = fh.read()
    head, _, motion = text.partition("MOTION")

    names, parents, offsets, channels, rot_orders = [], [], [], [], []
    stack = []
    tokens = head.split("\n")
    i = 0
    while i < len(tokens):
        line = tokens[i].strip()
        parts = line.split()
        if not parts:
            i += 1
            continue
        kw = parts[0]
        if kw in ("ROOT", "JOINT"):
            names.append(parts[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            channels.append([])
            rot_orders.append("")
            stack.append(len(names) - 1)
        elif kw == "End":
            # End Site: a leaf tip with an OFFSET and no CHANNELS. It contributes no
            # motion values; consuming one here would shift every later channel.
            depth = 0
            while i < len(tokens):
                if "{" in tokens[i]:
                    depth += 1
                if "}" in tokens[i]:
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
        elif kw == "OFFSET":
            offsets[stack[-1]] = [float(x) for x in parts[1:4]]
        elif kw == "CHANNELS":
            ch = parts[2:]
            channels[stack[-1]] = ch
            rot_orders[stack[-1]] = "".join(
                c[0].lower() for c in ch if c.endswith("rotation"))
        elif kw == "}":
            stack.pop()
        i += 1

    values = np.array([float(x) for x in motion.split("\n", 3)[3].split()],
                      dtype=np.float64)
    n_frames = int(motion.split("Frames:")[1].split()[0])
    frame_time = float(motion.split("Frame Time:")[1].split()[0])
    per_frame = sum(len(c) for c in channels)
    got = values.size // per_frame
    if got != n_frames:
        # The header's Frames: count is the file's own assertion. Disagreeing with the
        # data is a real defect, not a rounding detail -- fail rather than truncate.
        raise ValueError("%s: header says %d frames, data holds %d"
                         % (path, n_frames, got))
    values = values.reshape(n_frames, per_frame)

    n_j = len(names)
    rots = np.zeros((n_frames, n_j, 3))
    root_t = np.zeros((n_frames, 3))
    col = 0
    for j, ch in enumerate(channels):
        for name in ch:
            v = values[:, col]
            col += 1
            # BY NAME, never by position: the channel order is declared per joint and
            # 100STYLE declares Y,X,Z. Positional reads transpose every rotation.
            if name == "Xposition":
                root_t[:, 0] = v
            elif name == "Yposition":
                root_t[:, 1] = v
            elif name == "Zposition":
                root_t[:, 2] = v
            elif name == "Xrotation":
                rots[:, j, 0] = v
            elif name == "Yrotation":
                rots[:, j, 1] = v
            elif name == "Zrotation":
                rots[:, j, 2] = v
            else:
                raise ValueError("unknown channel %r in %s" % (name, path))

    return Bvh(names, np.array(parents), np.array(offsets, dtype=np.float64),
               channels, rot_orders, rots, frame_time, root_t)


def euler_to_matrix(xyz_deg, order):
    """Intrinsic Euler -> rotation matrix, applied in `order` (e.g. 'yxz').

    `xyz_deg` is always (rx, ry, rz) regardless of order -- the parser stores by name --
    and `order` says how to compose them."""
    rx, ry, rz = np.radians(xyz_deg)
    c, s = np.cos, np.sin
    mats = {
        "x": np.array([[1, 0, 0], [0, c(rx), -s(rx)], [0, s(rx), c(rx)]]),
        "y": np.array([[c(ry), 0, s(ry)], [0, 1, 0], [-s(ry), 0, c(ry)]]),
        "z": np.array([[c(rz), -s(rz), 0], [s(rz), c(rz), 0], [0, 0, 1]]),
    }
    out = np.eye(3)
    for axis in order:
        out = out @ mats[axis]
    return out


def forward_kinematics(bvh, frame):
    """World-space joint positions for one frame. (J,3), file units."""
    n_j = len(bvh.names)
    world_r = [None] * n_j
    world_p = np.zeros((n_j, 3))
    for j in range(n_j):
        local_r = euler_to_matrix(bvh.frames[frame, j], bvh.rot_orders[j] or "yxz")
        p = bvh.parents[j]
        if p < 0:
            world_r[j] = local_r
            world_p[j] = bvh.root_translation[frame]
        else:
            world_r[j] = world_r[p] @ local_r
            world_p[j] = world_p[p] + world_r[p] @ bvh.offsets[j]
    return world_p


def describe(bvh):
    """Report units and axes rather than assuming them."""
    p = forward_kinematics(bvh, 0)
    extent = p.max(0) - p.min(0)
    up = int(np.argmax(extent))
    span = float(extent[up])
    unit = "cm" if 50 < span < 250 else ("m" if 0.5 < span < 2.5 else "UNKNOWN")
    return dict(joints=len(bvh.names), frames=bvh.n_frames, fps=bvh.fps,
                up_axis=up, span=span, unit=unit, rot_order=bvh.rot_orders[0])


if __name__ == "__main__":
    import sys
    b = parse(sys.argv[1])
    d = describe(b)
    print("joints    %d" % d["joints"])
    print("frames    %d @ %.2f fps" % (d["frames"], d["fps"]))
    print("rot order %s (parsed from CHANNELS, not assumed)" % d["rot_order"])
    print("up axis   %d, span %.1f -> units look like %s" % (d["up_axis"], d["span"], d["unit"]))
    p = forward_kinematics(b, 0)
    print("\nframe 0 world positions (file units):")
    for n, xyz in zip(b.names, p):
        print("  %-16s %8.2f %8.2f %8.2f" % (n, *xyz))
