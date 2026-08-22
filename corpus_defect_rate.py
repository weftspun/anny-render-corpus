"""Corpus quality as a RATE, in ppm and ppb, not as a mean.

WHY A RATE. A mean over the whole mesh is the same mistake as counting web-server
fatals against a fixed denominator: when most of the population is quiet, the number
looks good regardless of the failure rate. Pronating a forearm leaves 1,356 torso
vertices at exactly 0.00 mm, so the whole-mesh mean reads 9.2 mm while the fingers are
off by ~a golf ball -- understating the part anyone cares about about 4x. The fix is
the same one that fixed the dashboard: divide by what actually varies, and report an
exceedance RATE against a stated tolerance rather than an average.

WHY ppm AND ppb. Leading zeros are unreadable and un-transmissible -- nobody remembers
whether 0.0004 is normal. ppm restores numbers you can reason about. The granularity
that makes each unit meaningful differs by population:

    identities        23,000          1 identity = 43 ppm     ppb undefined
    images           800,000          1 image    = 1.25 ppm   ppb undefined
    vertex-instances  11.0 billion    1 ppb      = 11 vertices    <-- ppb lives here

So per-vertex quality is quotable in ppb, while image-level quality bottoms out at
1.25 ppm. Reaching ppb at image granularity would take 2.1-15.7 GPU-years and 188 TB,
and would be pointless: for a FIXED population you enumerate rather than estimate, which
is why the preflight audit now decodes all 23,000 identities instead of sampling 300.

TOLERANCES are stated, not implied. A rate without a threshold is not a measurement.
"""

import argparse

import numpy as np
import pandas as pd
import torch

import anny_rig
from preflight_audit import load_wide

N_IMAGES_PLANNED = 800_000
TOLERANCES_MM = [1.0, 2.0, 5.0, 10.0, 20.0]
ANGLES = [45.0, 90.0, 135.0]


def procrustes_r(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b0.T @ a0)
    return u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt


def screw_from(v_rest, v_posed):
    r = procrustes_r(v_rest, v_posed)
    angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0, v_rest.mean(0)
    vec = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    axis = vec / (2 * np.sin(angle))
    c = np.linalg.lstsq(np.eye(3) - r,
                        v_posed.mean(0) - r @ v_rest.mean(0), rcond=None)[0]
    return axis / np.linalg.norm(axis), angle, c


def arm_errors(model, geo, v0, v1, hand_mask):
    """Per-vertex mm error of the arm against a linear twist ramp to the hand's
    own realised motion. Hand and fingers score 0 by construction, so what is left
    is purely how well the forearm distributes the twist."""
    axis, angle, centre = screw_from(v0[hand_mask], v1[hand_mask])
    elbow, faxis, length = geo
    s = np.clip(((v0 - elbow) @ faxis) / length, 0.0, 1.0)
    rel = v0 - centre
    theta = (angle * s)[:, None]
    a = np.broadcast_to(axis, rel.shape)
    ideal = centre + (rel * np.cos(theta) + np.cross(a, rel) * np.sin(theta)
                      + a * ((rel @ axis)[:, None]) * (1 - np.cos(theta)))
    return np.linalg.norm(v1 - ideal, axis=1) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--identities", type=int, default=120)
    args = ap.parse_args()

    _, wide = load_wide(args.corpus)
    n_total = len(wide)
    sub = wide.sample(min(args.identities, n_total), random_state=0)
    model = anny_rig.build_corpus_model(dtype=torch.float64)
    n_vert = int(model.vertex_bone_weights.shape[0])
    dom = anny_rig.dominant_bone(model)

    print("corpus %d identities x %d vertices; planned %s images" %
          (n_total, n_vert, f"{N_IMAGES_PLANNED:,}"))
    print("1 vertex-instance = %.4f ppb of the planned %0.1f billion\n"
          % (1e9 / (N_IMAGES_PLANNED * n_vert), N_IMAGES_PLANNED * n_vert / 1e9))

    stock = anny_rig.build_corpus_model(dtype=torch.float64, apply_twist_fix=False)
    results = {}
    # BASELINE MUST BE RAW MOCAP (fraction=0.0): all roll on the wrist channel, which is
    # all a capture system supplies. Using stock weights WITH dispersal as the reference
    # flatters the fix, because dispersal alone already recovers much of the error -- the
    # same species of mistake as a denominator that does not track reality.
    for tag, mdl, fr in (("raw mocap", stock, 0.0), ("SHIPPING (fixed)", model, 1.0)):
        results[tag] = measure(mdl, sub, dom, fraction=fr)

    n_arm = results["SHIPPING (fixed)"][2]
    print("Exceedance rate over %s arm vertex-instances "
          "(%d identities x 2 arms x %d angles)." % (f"{n_arm:,}", len(sub), len(ANGLES)))
    print("An absolute rate means nothing without a reference, so both rigs are shown.\n")
    print("%-12s %18s %18s %12s" % ("tolerance", "raw mocap", "SHIPPING", "improvement"))
    print("-" * 64)
    for tol in TOLERANCES_MM:
        a = results["raw mocap"][0][tol] / results["raw mocap"][2]
        b = results["SHIPPING (fixed)"][0][tol] / n_arm
        print("%-12s %15s %18s %11.0fx"
              % ("> %.0f mm" % tol, rate_str(a), rate_str(b),
                 (a / b) if b > 0 else float("inf")))
    print("\nworst single arm vertex: stock %.2f mm (%s) -> shipping %.2f mm (%s)"
          % (results["raw mocap"][1], human(results["raw mocap"][1]),
             results["SHIPPING (fixed)"][1], human(results["SHIPPING (fixed)"][1])))
    print("\nTolerances are stated, not implied: a rate without a threshold is not a")
    print("measurement. Read this as 'what fraction of ARM geometry misses the")
    print("anatomical twist ramp by more than X' -- the number a mocap model actually")
    print("inherits, not a mesh-wide mean diluted by a motionless torso.")


def rate_str(rate):
    if rate == 0:
        return "0 (none)"
    if rate * 1e6 >= 1:
        return "%9.0f ppm" % (rate * 1e6)
    return "%9.0f ppb" % (rate * 1e9)


def measure(model, sub, dom, fraction=1.0):
    counts = {t: 0 for t in TOLERANCES_MM}
    total = 0
    worst = 0.0
    for side in anny_rig.SIDES:
        _, _, _, hi = anny_rig.bone_ids(model, side)
        hand_mask = dom == hi
        for _, row in sub.iterrows():
            kw = {c: torch.tensor([row[c]], dtype=torch.float64) for c in sub.columns}
            pose0 = torch.eye(4, dtype=torch.float64)[None, None].repeat(
                1, model.bone_count, 1, 1)
            with torch.no_grad():
                v0 = model(pose_parameters=pose0,
                           phenotype_kwargs=kw)["vertices"][0].numpy()
            geo = anny_rig.forearm_frame(model, side, v0)
            for deg in ANGLES:
                pose = pose0.clone()
                anny_rig.disperse_wrist_roll(model, pose, side, deg, fraction=fraction)
                with torch.no_grad():
                    v1 = model(pose_parameters=pose,
                               phenotype_kwargs=kw)["vertices"][0].numpy()
                elbow, faxis, length = geo
                rel = v0 - elbow
                t = rel @ faxis
                r = np.linalg.norm(rel - np.outer(t, faxis), axis=1)
                arm = ((t > 0.0) & (t < 1.1 * length) & (r < 0.4 * length)) | hand_mask
                err = arm_errors(model, geo, v0, v1, hand_mask)[arm]
                total += err.size
                worst = max(worst, float(err.max()))
                for tol in TOLERANCES_MM:
                    counts[tol] += int((err > tol).sum())

    return counts, worst, total


def human(mm):
    for size, name in [(0.76, "a credit card's thickness"), (1.52, "a penny's thickness"),
                       (7.0, "a pencil's diameter"), (10.5, "a AAA battery's diameter"),
                       (14.5, "a AA battery's diameter"), (21.2, "a nickel's diameter"),
                       (42.7, "a golf ball"), (66.0, "a soda can's diameter")]:
        n = mm / size
        if 0.8 <= n <= 1.25:
            return "~%s" % name
        if n < 0.8:
            return "~%.1fx %s" % (n, name)
    return "~%.1f soda cans" % (mm / 66.0)


if __name__ == "__main__":
    main()
