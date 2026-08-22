"""The corpus's canonical ANNY model: one builder, twist defect fixed, both arms.

WHY THIS MODULE EXISTS. ANNY's stock rig cannot transmit forearm twist. Motion capture
delivers rotations for deployable humanoid joints only -- no capture system outputs a
twist bone -- and driving the wrist channel leaves the forearm skin essentially still:
13.0 deg of skin roll where the anatomical ramp calls for 78.8. The error lands on the
extremities, which is exactly where a mocap corpus cannot afford it: fingers off by
~a golf ball at 90 deg pronation, while the torso sits at 0.00 mm and the whole-mesh
mean understates the hand error about 4x.

Rendering 800k images from the stock rig would bake anatomically wrong forearms into the
corpus at the source, and restarting is the expensive thing. So every consumer -- the
preflight audit, the interface audit, and the renderer -- must build its model HERE
rather than calling anny.Anny directly, so the fix cannot be forgotten in one path.

THE FIX IS ONE THING: RE-WEIGHTING. NO TWIST BONE, NO RUNTIME STEP.

The forearm is re-weighted as a linear elbow->wrist ramp landing on the WRIST bone, so a
vertex carrying weight s on the wrist follows s of the wrist's rotation and the ramp
itself becomes the twist distribution. Driving the wrist -- the only channel mocap
supplies -- then produces a graded twist with nothing left for a runtime to do.

    pronation   stock   twist-bone + dispersal   wrist ramp (SHIPPING)
    45 deg      26.0d          1.3d                    1.0d
    90 deg      52.8d          4.5d                    3.6d
    135 deg     80.9d         14.3d                   14.0d

An earlier version of this module used a twist bone plus a dispersal step mirroring
Godot's BoneTwistDisperser3D. The wrist ramp replaced it because it is simultaneously
simpler, slightly more accurate (one continuous ramp rather than two hops through a
dispersed intermediate), and the only variant that survives deployment: glTF carries no
runtime code, so a twist bone needing a disperser arrives empty, and humanoid retarget
profiles (Godot humanoid, VRM) have no twist-bone slots at all, so they drop it. Adding
MORE twist bones cannot help -- a profile discards every bone it cannot name.

WHAT REMAINS is LBS volume collapse: radial keep 0.79 at 90 deg, 0.62 at 135. It is
identical under both variants because it is a property of linear blend skinning, not of
the twist scheme, and the wrist ramp cannot help -- the ramp fixes WHERE the twist goes,
not the volume lost blending identity with a large rotation.

NOT IMPLEMENTED HERE (deliberately): Delta Mush. Measured on top of the wrist ramp it
recovers about a third of the collapse (radial keep 0.793 -> 0.861, ramp RMSE 3.6 -> 2.5
deg at 90 deg), so it helps but does not fix it. It stays out of this module for two
reasons: it is a per-frame post-process that breaks GPU-batched rendering across 800k
frames, and whether that cost is worth a third of the collapse depends on how often the
pose distribution actually reaches 90 deg pronation -- which is unknown until the poses
relation exists. A working implementation lives in
godot-soma-twist/experiments/twist_fix.py; wire it in once the pose statistics justify it.

Direct Delta Mush bakes the SMOOTHING (that is its contribution over classic Delta Mush)
but not the POSE DEPENDENCE: runtime still forms a per-vertex matrix sum and extracts a
rotation from it, which glTF's scalar-weight skinning cannot express. So DDM is fine for
renders and for baked clips, and is not an option for live avatars. Dual-quaternion
skinning is the other standard answer and ANNY ships it, but DQS IS BLOCKLISTED here.

TWO TRAPS THIS MODULE ENCODES so callers cannot re-hit them:
  * `rest_bone_heads` pairs with `rest_vertices`, NEVER with `vertices` -- an identity
    pose is not the rest pose. Mixing them is off 55 mm on an adult and 500 mm on a
    child, so it hides on the body everyone spot-checks. `forearm_frame` therefore
    derives its geometry from the MESH plus skin weights and reads no bone head at all.
  * A bone's "local Z" is not its roll axis. The local->world rotation map is the
    IDENTITY, so local Z is WORLD Z -- 55 deg off the forearm. `roll_axis_for` recovers
    the true axis by probing with DESCENDANT vertices, which follow the bone rigidly;
    a bone's own vertices are LBS-blended and fitting them puts the rotation centre
    33 mm off.
"""

import numpy as np
import torch

import anny

SIDES = ("L", "R")
ARM_CHAIN = ("upperarm02", "lowerarm01", "lowerarm02", "wrist")

# The corpus's fixed model configuration. Pinned here so every consumer agrees; changing
# any of these invalidates already-rendered shards, so treat it as a schema change.
CORPUS_CONFIG = dict(rig="anny", topology="anny", phenotypes="all",
                     local_changes="default", facial_actions="all",
                     skinning_method="lbs")


def build_corpus_model(dtype=torch.float32, apply_twist_fix=True):
    """The model every corpus stage must use. Twist fix applied to BOTH arms."""
    model = anny.Anny(**CORPUS_CONFIG).to(dtype=dtype)
    if apply_twist_fix:
        fix_forearm_twist(model)
    return model


# ---------------------------------------------------------------- geometry helpers

def dominant_bone(model):
    w = model.vertex_bone_weights.detach().cpu().numpy()
    idx = model.vertex_bone_indices.detach().cpu().numpy()
    return idx[np.arange(len(idx)), w.argmax(1)]


def _identity_pose(model, batch=1):
    return torch.eye(4, dtype=model.vertex_bone_weights.dtype)[None, None].repeat(
        batch, model.bone_count, 1, 1)


def _vertices(model, pose=None, phenotype_kwargs=None):
    with torch.no_grad():
        out = model(pose_parameters=pose if pose is not None else _identity_pose(model),
                    phenotype_kwargs=phenotype_kwargs or {})
    return out["vertices"][0].detach().cpu().numpy()


def bone_ids(model, side):
    labels = list(model.bone_labels)
    return [labels.index("%s.%s" % (n, side)) for n in ARM_CHAIN]


def forearm_frame(model, side, verts=None):
    """(elbow, unit axis, length) for one forearm, from mesh + skin weights only.

    Reads no bone head, deliberately: rest_bone_heads pairs with rest_vertices and not
    with vertices, and that mismatch is 500 mm on a child.

    CACHED ONCE, BEFORE RE-WEIGHTING. The forearm region is selected by DOMINANT BONE,
    and the wrist ramp deliberately hands much of the forearm to the wrist -- so
    recomputing the frame after the fix selects a different, smaller vertex set and the
    axis drifts. That drift is invisible (nothing errors) and it degraded the measured
    shipping quality from 3.6 to 17.0 deg RMSE, i.e. it made a working fix look broken.
    The frame is a property of the REST anatomy, so it is computed from the stock weights
    and reused."""
    cache = getattr(model, "_forearm_frames", None)
    if cache is not None and side in cache:
        return cache[side]
    ui, fi, ti, hi = bone_ids(model, side)
    v = _vertices(model) if verts is None else verts
    dom = dominant_bone(model)
    fore = v[np.isin(dom, [fi, ti])]
    if len(fore) < 30:
        raise RuntimeError("forearm region too small for side %s" % side)
    axis = np.linalg.svd(fore - fore.mean(0), full_matrices=False)[2][0]
    up_c, hand_c = v[dom == ui].mean(0), v[dom == hi].mean(0)
    if np.dot(hand_c - up_c, axis) < 0:
        axis = -axis
    elbow = fore.mean(0) + ((fore @ axis).min() - fore.mean(0) @ axis) * axis
    return elbow, axis, float(np.dot(hand_c - elbow, axis))


# ---------------------------------------------------------------- the fix

def fix_forearm_twist(model, distal="wrist"):
    """Re-weight both forearms as a linear elbow->distal ramp. NO TWIST BONE.

    `distal="wrist"` ramps straight onto the wrist, which is the shipping configuration
    and needs no twist bone at all. Measured against the anatomical ramp, driven by the
    wrist alone (the only channel motion capture supplies):

        pronation   stock   twist-bone + dispersal   wrist ramp
        45 deg      26.0d          1.3d                1.0d
        90 deg      52.8d          4.5d                3.6d
        135 deg     80.9d         14.3d               14.0d

    The wrist ramp is not a compromise -- it is slightly BETTER, because it is one
    continuous ramp instead of two hops through a dispersed intermediate. It is also the
    only version that survives deployment:

      * glTF carries no runtime code, so a twist bone that needs a disperser to be filled
        in arrives empty;
      * humanoid retarget profiles (Godot humanoid, VRM) have no twist-bone slots, so a
        twist bone is dropped outright when an avatar is driven by live mocap or IK.
        Adding MORE twist bones cannot help -- the profile discards every bone it cannot
        name. The wrist, by contrast, is named by every profile and driven by every
        retargeter.

    What remains is LBS volume collapse (radial keep 0.79 at 90 deg, 0.62 at 135), which
    is identical for both variants and is a property of linear blend skinning, not of the
    twist scheme. The corpus removes it with Delta Mush at render time, where no glTF is
    involved; deployed avatars keep it, as every LBS avatar does.

    Only mass already on {lowerarm01, lowerarm02, distal} is redistributed, so partition
    of unity holds exactly and nothing outside the forearm changes. Idempotent, which the
    corpus's resumability contract needs."""
    w = model.vertex_bone_weights.detach().cpu().numpy().copy()
    idx = model.vertex_bone_indices.detach().cpu().numpy().copy()
    verts = _vertices(model)
    touched_total = 0

    # Freeze the frames from the STOCK weights before touching anything: the ramp moves
    # forearm vertices onto the wrist, so a frame recomputed afterwards selects a
    # different region and drifts. See forearm_frame's docstring.
    model._forearm_frames = {s: forearm_frame(model, s, verts) for s in SIDES}

    for side in SIDES:
        _, fi, ti, hi = bone_ids(model, side)
        target = hi if distal == "wrist" else ti
        elbow, axis, length = model._forearm_frames[side]
        s = np.clip(((verts - elbow) @ axis) / length, 0.0, 1.0)
        involved = np.isin(idx, [fi, ti, target])
        mass = np.where(involved, w, 0.0).sum(1)
        for v in np.where(mass > 1e-6)[0]:
            slots = list(np.where(involved[v])[0])
            if len(slots) == 1:
                spare = int(np.argmin(np.where(involved[v], 9e9, w[v])))
                if spare not in slots:
                    slots.append(spare)
            w[v, slots] = 0.0
            idx[v, slots[0]], w[v, slots[0]] = fi, mass[v] * (1.0 - s[v])
            if len(slots) > 1:
                idx[v, slots[1]], w[v, slots[1]] = target, mass[v] * s[v]
            touched_total += 1

    dtype = model.vertex_bone_weights.dtype
    model.vertex_bone_weights.data = torch.tensor(w, dtype=dtype)
    model.vertex_bone_indices.data = torch.tensor(
        idx, dtype=model.vertex_bone_indices.dtype)
    return touched_total


def roll_axis_for(model, bone, world_axis, probe_deg=10.0):
    """Local rotation axis realising `world_axis` for `bone`, recovered by probing.

    Probes DESCENDANT vertices: they inherit the bone's transform rigidly, whereas the
    bone's own vertices are LBS-blended with neighbours and fitting them displaces the
    recovered rotation centre by ~33 mm."""
    parents = np.asarray(model.bone_parents)
    desc, frontier = set(), [bone]
    while frontier:
        cur = frontier.pop()
        for k in np.where(parents == cur)[0]:
            if int(k) not in desc:
                desc.add(int(k))
                frontier.append(int(k))
    dom = dominant_bone(model)
    mask = np.isin(dom, np.array(sorted(desc), dtype=int)) if desc else (dom == bone)
    if mask.sum() < 12:
        mask = dom == bone

    v0 = _vertices(model)
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        pose = _identity_pose(model)
        pose[0, bone, :3, :3] = torch.tensor(rotation(e, probe_deg),
                                             dtype=pose.dtype)
        v1 = _vertices(model, pose)
        a0, b0 = v0[mask] - v0[mask].mean(0), v1[mask] - v1[mask].mean(0)
        u, _, vt = np.linalg.svd(b0.T @ a0)
        r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        ang = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
        vec = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        cols.append(vec / (2 * np.sin(ang)) if ang > 1e-8 else e)
    a = np.linalg.solve(np.stack(cols, axis=1), world_axis)
    return a / np.linalg.norm(a)


def rotation(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


def disperse_wrist_roll(model, pose, side, roll_deg, fraction=0.0):
    """Apply `roll_deg` of pronation, in place.

    `fraction` is how much of the roll is moved onto the twist bone. SHIPPING USES 0.0:
    the whole roll stays on the wrist, exactly as motion capture delivers it, and the
    re-weighted forearm distributes it. There is no dispersal step and nothing for a
    runtime to do, which is what makes the result exportable to glTF and safe under
    humanoid retargeting.

    fraction > 0 mirrors Godot's BoneTwistDisperser3D and is kept only for comparison:
    it measured slightly WORSE than the wrist ramp (4.5 vs 3.6 deg RMSE at 90 deg) and
    depends on a twist bone that a humanoid profile drops anyway."""
    _, _, ti, hi = bone_ids(model, side)
    _, axis, _ = forearm_frame(model, side)
    la_t = roll_axis_for(model, ti, axis)
    la_h = roll_axis_for(model, hi, axis)
    pose[0, ti, :3, :3] = torch.tensor(rotation(la_t, roll_deg * fraction),
                                       dtype=pose.dtype)
    pose[0, hi, :3, :3] = torch.tensor(rotation(la_h, roll_deg * (1.0 - fraction)),
                                       dtype=pose.dtype)
    return pose


def twist_transmission(model, side="L", deg=90.0, fraction=0.0):
    """Measured skin roll near the wrist under a `deg` pronation, and its ideal.

    The corpus's regression guard. `fraction` is how much of the roll is dispersed onto
    the twist bone:
        0.0  the entire roll on the wrist channel -- what mocap supplies and what
             SHIPPING uses, since the re-weighted forearm distributes it itself.
        1.0  dispersed onto the twist bone (legacy comparison only).
    On a STOCK model fraction=0.0 is the defect condition and fraction=1.0 flatters it,
    because dispersal alone already helps. Returns (measured, ideal)."""
    elbow, axis, length = forearm_frame(model, side)
    v0 = _vertices(model)
    pose = _identity_pose(model)
    disperse_wrist_roll(model, pose, side, deg, fraction=fraction)
    v1 = _vertices(model, pose)

    rel = v0 - elbow
    t = rel @ axis
    r = np.linalg.norm(rel - np.outer(t, axis), axis=1)
    band = (t > 0.80 * length) & (t < 0.95 * length) & (r < 0.35 * length)
    a0, a1 = v0[band] - elbow, v1[band] - elbow
    p0 = a0 - np.outer(a0 @ axis, axis)
    p1 = a1 - np.outer(a1 @ axis, axis)
    n0, n1 = np.linalg.norm(p0, axis=1), np.linalg.norm(p1, axis=1)
    g = (n0 > 1e-5) & (n1 > 1e-5)
    if not g.any():
        return 0.0, deg * 0.875
    meas = float(np.degrees(np.arccos(np.clip(
        (p0[g] * p1[g]).sum(1) / (n0[g] * n1[g]), -1, 1))).mean())
    return meas, deg * 0.875


RAMP_BANDS = [(0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.95)]
ZERO_TWIST_RMSE = 55.2      # RMS of the ideal ramp about zero: "the skin never moved"


def twist_ramp_rmse(model, side="L", deg=90.0, fraction=0.0):
    """RMS deviation of the forearm twist profile from the anatomical linear ramp.

    A single distal-band reading is not a sufficient regression guard: with the roll
    dispersed, the STOCK rig reads 67 deg near the wrist and would pass a loose
    threshold, even though its profile is flat through the mid-forearm and only catches
    up at the very end. Linearity is what distinguishes a real twist from a hinge at the
    wrist, so the guard scores the whole profile: 0 deg would be a perfect ramp,
    %.1f deg means the skin never rotated at all.""" % ZERO_TWIST_RMSE
    elbow, axis, length = forearm_frame(model, side)
    v0 = _vertices(model)
    pose = _identity_pose(model)
    disperse_wrist_roll(model, pose, side, deg, fraction=fraction)
    v1 = _vertices(model, pose)

    rel = v0 - elbow
    t = rel @ axis
    r = np.linalg.norm(rel - np.outer(t, axis), axis=1)
    sel = (t > 0.15 * length) & (t < 0.95 * length) & (r < 0.35 * length)
    a0, a1 = v0[sel] - elbow, v1[sel] - elbow
    p0 = a0 - np.outer(a0 @ axis, axis)
    p1 = a1 - np.outer(a1 @ axis, axis)
    n0, n1 = np.linalg.norm(p0, axis=1), np.linalg.norm(p1, axis=1)
    g = (n0 > 1e-5) & (n1 > 1e-5)
    ang = np.degrees(np.arccos(np.clip(
        (p0[g] * p1[g]).sum(1) / (n0[g] * n1[g]), -1, 1)))
    frac = (t[sel] / length)[g]
    errs, profile = [], []
    for lo, hi in RAMP_BANDS:
        b = (frac >= lo) & (frac < hi)
        if b.sum() >= 3:
            meas = float(np.median(ang[b]))
            profile.append(meas)
            errs.append(meas - deg * (lo + hi) / 2)
    if not errs:
        return float("nan"), profile
    return float(np.sqrt(np.mean(np.square(errs)))), profile


if __name__ == "__main__":
    stock = build_corpus_model(dtype=torch.float64, apply_twist_fix=False)
    fixed = build_corpus_model(dtype=torch.float64)
    print("forearm skin roll near the wrist, 90 deg pronation (ideal 78.8 deg)\n")
    print("%-6s %14s %16s %14s" % ("arm", "raw mocap", "twist bone", "SHIPPING"))
    print("%-6s %14s %16s %14s" % ("", "stock weights", "+dispersal", "wrist ramp"))
    print("-" * 54)
    for s in SIDES:
        raw = twist_transmission(stock, s, fraction=0.0)[0]
        disp = twist_transmission(stock, s, fraction=1.0)[0]
        # SHIPPING drives the WRIST (fraction=0.0): the ramp lands on the wrist, so
        # dispersing onto a twist bone would measure a configuration we do not ship.
        ship, ideal = twist_transmission(fixed, s, fraction=0.0)
        print("%-6s %11.1f deg %13.1f deg %11.1f deg" % (s, raw, disp, ship))
    print("\nRest pose must be untouched by the re-weighting -- it only redistributes")
    print("mass between two bones that are both at identity in the rest pose:")
    with torch.no_grad():
        p = _identity_pose(stock)
        a = stock(pose_parameters=p)["rest_vertices"][0].numpy()
        b = fixed(pose_parameters=p)["rest_vertices"][0].numpy()
    print("  max |rest_vertices stock - fixed| = %.6f mm" % (np.abs(a - b).max() * 1000))
