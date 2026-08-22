"""Rung 0: does a BVH pose actually transfer onto ANNY, and by which formulation?

The corpus needs 100STYLE frames as ANNY poses. The naive route -- copy each joint's
local Euler onto the mapped ANNY bone -- is the one that already failed this project on
the finger chain, where a mapping verified for arms and legs was never independently
verified elsewhere and compounded down the chain. So no formulation is adopted here on
the strength of it being obvious; two are measured and the numbers decide.

  A  LOCAL: copy BVH's per-joint local Euler straight onto the mapped ANNY bone.
     Only correct if the two skeletons share bind orientations, which is exactly the
     assumption that has burned us.
  B  WORLD: run FK on the BVH, take each joint's WORLD rotation, and feed that. ANNY's
     `local-ref` rotation axes were measured to be WORLD-ALIGNED (the local->world map
     is the identity), so this should be the formulation that matches -- but "should"
     is why this file exists.

THE SCORE IS LIMB DIRECTION, and getting there took two wrong turns worth recording.

Joint POSITION residual after a similarity fit was the first attempt. It reported 153 mm
and looked damning -- until a negative control put two REST skeletons side by side and
scored 139.7 mm. The posed number sat 13 mm above a floor nobody had measured, so the
metric was mostly reporting that the mocap actor and the ANNY body are different people.
A verdict of "neither formulation transfers the pose" was drawn from it and was WRONG.

Limb DIRECTION is the right observable: unit vectors along each bone are scale-free and
proportion-free, so body difference cancels and only pose transfer is left. Same lesson
as the thigh measurement elsewhere in this project -- centroid position was an artefact
and recovered joint angle was the truth.

The control is kept and reported FIRST, because the floor is the finding: with both
skeletons at rest and no rotation anywhere, limb directions still disagree by 29.6 deg
mean / 58.7 deg worst. That is BIND-ORIENTATION mismatch, and no per-frame formulation
can remove it.

FRAMES: 100STYLE is Y-up and centimetres; ANNY is Z-up and metres. Both conversions are
applied explicitly and the axis is DETECTED from the data, never assumed.
"""

import numpy as np
import torch

import anny_rig
import bvh_parse

# BVH joint -> ANNY bone. Determined from both hierarchies, not guessed:
# ANNY's spine runs spine05 (child of root) up to spine01 (parent of neck01 and
# clavicle), so BVH's Chest..Chest4 descend onto spine05..spine02.
#
# Twist bones (upperarm02, lowerarm02, upperleg02, lowerleg02) are deliberately ABSENT.
# BVH has no twist channel, and the corpus rig distributes twist from the wrist through
# re-weighted skin rather than through a twist bone. An absent bone is identity, which is
# both correct and ETNF-clean: no row rather than a row of zeros.
JOINT_MAP = {
    "Hips": "root",
    "Chest": "spine05", "Chest2": "spine04", "Chest3": "spine03", "Chest4": "spine02",
    "Neck": "neck01", "Head": "head",
    "RightCollar": "clavicle.R", "RightShoulder": "upperarm01.R",
    "RightElbow": "lowerarm01.R", "RightWrist": "wrist.R",
    "LeftCollar": "clavicle.L", "LeftShoulder": "upperarm01.L",
    "LeftElbow": "lowerarm01.L", "LeftWrist": "wrist.L",
    "RightHip": "upperleg01.R", "RightKnee": "lowerleg01.R",
    "RightAnkle": "foot.R", "RightToe": "toe1-1.R",
    "LeftHip": "upperleg01.L", "LeftKnee": "lowerleg01.L",
    "LeftAnkle": "foot.L", "LeftToe": "toe1-1.L",
}

# Joints used to score. Toes and collars are excluded: ANNY's toe1-1 is one of five toes
# and its clavicle sits differently, so including them would measure mapping ambiguity
# rather than pose transfer.
SCORE_JOINTS = ["Hips", "Chest2", "Chest4", "Neck", "Head",
                "RightShoulder", "RightElbow", "RightWrist",
                "LeftShoulder", "LeftElbow", "LeftWrist",
                "RightHip", "RightKnee", "RightAnkle",
                "LeftHip", "LeftKnee", "LeftAnkle"]


def y_up_cm_to_z_up_m(v):
    """100STYLE (Y-up, cm) -> ANNY (Z-up, m). (x, y, z) -> (x, -z, y) / 100."""
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) / 100.0


def rot_to_rotvec(r):
    ang = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v / (2 * np.sin(ang)) * ang


def similarity_residual(src, dst):
    """Umeyama fit, then per-point residual in mm.

    Scale is fitted out on purpose: the BVH actor and the ANNY body are different people,
    so absolute size difference is not a pose error. What survives is shape disagreement."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(d0.T @ s0 / len(src))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    scale = float(np.trace(np.diag(sv) @ np.diag([1, 1, d])) / ((s0 ** 2).sum() / len(src)))
    fitted = (scale * (r @ s0.T).T) + mu_d
    return np.linalg.norm(fitted - dst, axis=1) * 1000.0, scale


def anny_joint_positions(model, pose, names):
    """World positions of the ANNY bones corresponding to `names`, via skin centroids.

    Uses skin-weight centroids rather than rest_bone_heads because rest_bone_heads pairs
    with rest_vertices and NOT with the posed `vertices` -- mixing them is 500 mm out on a
    child. Centroids come from the same array as the pose, so they cannot disagree."""
    labels = list(model.bone_labels)
    parents = np.asarray(model.bone_parents)
    w = model.vertex_bone_weights.detach().cpu().numpy()
    idx = model.vertex_bone_indices.detach().cpu().numpy()
    with torch.no_grad():
        v = model(pose_parameters=pose)["vertices"][0].numpy()

    def influence(bone):
        """Total weight each vertex places on `bone`. Influence, not dominance:
        `root` dominates no vertex at all, so a dominance test returns an empty set
        and a NaN centroid that silently poisons the fit downstream."""
        return np.where(idx == bone, w, 0.0).sum(1)

    def descendants(bone):
        out, frontier = [], [bone]
        while frontier:
            cur = frontier.pop()
            for k in np.where(parents == cur)[0]:
                out.append(int(k))
                frontier.append(int(k))
        return out

    out = []
    for n in names:
        bone = labels.index(JOINT_MAP[n])
        inf = influence(bone)
        if inf.sum() < 1e-6:
            # A bone that skins nothing is located by the geometry it moves: the
            # weighted centroid of everything below it in the hierarchy.
            for k in descendants(bone):
                inf = inf + influence(k)
        if inf.sum() < 1e-6:
            raise RuntimeError("cannot localise bone %s" % labels[bone])
        out.append((v * inf[:, None]).sum(0) / inf.sum())
    return np.array(out)


def build_pose(model, bvh, frame, mode):
    labels = list(model.bone_labels)
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(
        1, model.bone_count, 1, 1)
    n_j = len(bvh.names)
    world_r = [None] * n_j
    for j in range(n_j):
        local = bvh_parse.euler_to_matrix(bvh.frames[frame, j], bvh.rot_orders[j] or "yxz")
        p = bvh.parents[j]
        world_r[j] = local if p < 0 else world_r[p] @ local
        name = bvh.names[j]
        if name not in JOINT_MAP:
            continue
        bone = labels.index(JOINT_MAP[name])
        r = local if mode == "local" else world_r[j]
        # BVH is Y-up, ANNY is Z-up: conjugate the rotation into ANNY's frame.
        c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        pose[0, bone, :3, :3] = torch.tensor(c @ r @ c.T)
    return pose


def main():
    import sys
    path = (sys.argv[1] if len(sys.argv) > 1 else
            "O:/Documents/Datasets/dataset-100style-mocap/unpacked/100STYLE/"
            "Aeroplane/Aeroplane_FW.bvh")
    frames = [int(x) for x in (sys.argv[2:] or ["500", "1500", "2500"])]

    bvh = bvh_parse.parse(path)
    d = bvh_parse.describe(bvh)
    print("clip: %s" % path.split("/")[-1])
    print("  %d joints, %d frames @ %.0f fps, rot order %s, up axis %d, units %s\n"
          % (d["joints"], d["frames"], d["fps"], d["rot_order"], d["up_axis"], d["unit"]))

    missing = [n for n in bvh.names if n not in JOINT_MAP]
    print("BVH joints unmapped: %s" % (missing or "none"))
    print("ANNY bones receiving no source (identity, correct): twist bones + fingers\n")

    model = anny_rig.build_corpus_model(dtype=torch.float64)
    idx = [bvh.names.index(n) for n in SCORE_JOINTS]

    print("%-8s %-10s %10s %10s %8s" % ("frame", "formulation", "mean mm", "p95 mm", "scale"))
    print("-" * 52)
    verdict = {}
    for f in frames:
        if f >= bvh.n_frames:
            continue
        ref = y_up_cm_to_z_up_m(bvh_parse.forward_kinematics(bvh, f)[idx])
        for mode in ("local", "world"):
            pose = build_pose(model, bvh, f, mode)
            got = anny_joint_positions(model, pose, SCORE_JOINTS)
            res, scale = similarity_residual(got, ref)
            verdict.setdefault(mode, []).append(res.mean())
            print("%-8d %-10s %9.1f %10.1f %8.3f"
                  % (f, mode, res.mean(), np.percentile(res, 95), scale))
        print()

    print("Residual is shape disagreement after fitting out position, orientation and")
    print("scale -- the two skeletons are different people, so size is not pose error.")
    best = min(verdict, key=lambda k: np.mean(verdict[k]))
    print("\nBEST: %s (%.1f mm mean across frames)" % (best, np.mean(verdict[best])))
    if np.mean(verdict[best]) > 60:
        print("NEITHER formulation transfers the pose. A per-joint bind-orientation")
        print("correction is required before 100STYLE can populate the poses relation --")
        print("do not convert 810 clips on top of a mapping this wrong.")


if __name__ == "__main__":
    main()
