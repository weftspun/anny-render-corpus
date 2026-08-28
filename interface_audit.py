"""Enumerate the pipeline's INTERFACES and check each one.

Organising principle, stated plainly because it earned its place: the edges and
interfaces are where the problems are. Every defect this project has hit lived at a
boundary between two things, never inside one of them:

  * ANNY gender 0=male vs GNM FEMALE=0        -- would have flipped sex on 800k images
  * ANNY Z-up/metres vs SOMA +Y-up/centimetres -- 100x scale or a body on its side
  * "local Z" vs the bone's actual roll axis   -- measured a bend and called it a twist
  * rest_bone_heads vs vertices                -- 55 mm on an adult, 500 mm on a child
  * lowerarm01 / lowerarm02 weight boundary    -- the forearm twist defect itself
  * twist bones absent from LabRCSF            -- 12 ANNY bones with no canonical name
  * forearm / hand mask boundary               -- mesh mean understated finger error 4x
  * train / val split boundary                 -- contamination

Components get tested because they are easy to name. Interfaces get missed because
they belong to nobody. So this file NAMES them, and every named interface must end in
one of three states -- never silently absent:

  OK        an executed check passed
  HAZARD    an executed check failed; the detail says how badly
  UNCHECKED no automated check exists yet. Printed loudly and counted, because a
            silent skip reads exactly like a pass. This project has been burned by
            that three times (the split-contamination check nested under an unrelated
            condition; the dimorphism and child checks skipping on empty subgroups).

Usage: python interface_audit.py
Exit code is non-zero if any interface is a HAZARD.
"""

import sys

import numpy as np

RESULTS = []


def record(edge, state, detail=""):
    RESULTS.append((edge, state, detail))
    print("  [%-9s] %-46s %s" % (state, edge, detail))


def ok(edge, passed, detail_ok="", detail_bad=""):
    record(edge, "OK" if passed else "HAZARD", detail_ok if passed else detail_bad)
    return passed


def unchecked(edge, why):
    record(edge, "UNCHECKED", why)


def main():
    import torch
    from scipy.spatial import cKDTree

    import anny_rig

    # Audit the model the corpus ACTUALLY SHIPS, not a bare anny.Anny. Building the
    # model here independently is how an audit ends up certifying something nothing
    # renders with.
    model = anny_rig.build_corpus_model(dtype=torch.float64)
    labels = list(model.bone_labels)

    def fwd(ph=None, pose=None):
        kw = {k: torch.tensor([v], dtype=torch.float64) for k, v in (ph or {}).items()}
        if pose is None:
            pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, model.bone_count, 1, 1)
        with torch.no_grad():
            return model(pose_parameters=pose, phenotype_kwargs=kw)

    CHILD = dict(age=0.1, height=0.2, gender=0.5)

    print("\nSKELETON <-> MESH")
    o_a, o_c = fwd(), fwd(CHILD)
    def prox(j, v):
        st = float((v.max(0) - v.min(0)).max())
        return 100.0 * float(np.median(cKDTree(v).query(j)[0])) / st
    matched = max(prox(o["rest_bone_heads"][0].numpy(), o["rest_vertices"][0].numpy())
                  for o in (o_a, o_c))
    mixed = max(prox(o["rest_bone_heads"][0].numpy(), o["vertices"][0].numpy())
                for o in (o_a, o_c))
    ok("rest skeleton pairs with rest mesh", matched < 1.0,
       "joint-to-surface %.2f%% of stature (adult+child)" % matched,
       "%.1f%% -- joints are outside the body" % matched)
    ok("mismatched pairing is detectably wrong", mixed > 3.0,
       "control: mispairing reaches %.1f%% of stature" % mixed,
       "control did not separate -- the check above proves nothing")

    print("\nUNITS AND ORIENTATION")
    v = o_a["rest_vertices"][0].numpy()
    ext = v.max(0) - v.min(0)
    ok("ANNY stature is metres, not centimetres", 0.4 < ext.max() < 2.5,
       "%.3f m" % ext.max(), "%.3f -- wrong unit" % ext.max())
    ok("ANNY up axis is Z", int(np.argmax(ext)) == 2,
       "axis 2", "largest extent on axis %d" % int(np.argmax(ext)))
    unchecked("ANNY <-> SOMA units and up-axis",
              "SOMA docs say cm/+Y; no SOMA export path is wired yet to test against")

    print("\nSEX POLARITY (cross-model)")
    n = 120
    g = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    kw = {k: torch.tensor(x, dtype=torch.float64) for k, x in
          dict(gender=g, height=np.full(n, .5), age=np.full(n, .75),
               muscle=np.full(n, .5), weight=np.full(n, .5)).items()}
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(n, model.bone_count, 1, 1)
    with torch.no_grad():
        vv = model(pose_parameters=pose, phenotype_kwargs=kw)["vertices"].numpy()
    st = (vv.max(1) - vv.min(1))[:, 2]
    dm = float(np.median(st[g == 0]) - np.median(st[g == 1]))
    ok("ANNY gender 0=male (males taller)", 0.05 < dm < 0.20,
       "dimorphism %+.1f cm" % (dm * 100), "dimorphism %+.1f cm -- polarity suspect" % (dm * 100))
    try:
        from gnm.shape import semantic_sampler
        inverted = (int(semantic_sampler.Gender.FEMALE) == 0
                    and int(semantic_sampler.Gender.MALE) == 1)
        ok("GNM<->ANNY gender polarity is INVERTED (and handled)", inverted,
           "GNM FEMALE=0/MALE=1 vs ANNY 0=male -- conversion required",
           "GNM enum changed; the conversion in headfit.py is now wrong")
    except Exception as exc:                                          # noqa: BLE001
        unchecked("GNM <-> ANNY gender polarity",
                  "gnm not importable here (%s)" % type(exc).__name__)

    print("\nPOSE PARAMETERS <-> BONE FRAMES")
    hi = labels.index("wrist.L")
    def probe(k):
        e = np.zeros(3); e[k] = 1.0
        t = np.radians(10.0)
        kx = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        r = np.eye(3) + np.sin(t) * kx + (1 - np.cos(t)) * (kx @ kx)
        p = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, model.bone_count, 1, 1)
        p[0, hi, :3, :3] = torch.tensor(r)
        w = model.vertex_bone_weights.numpy(); ix = model.vertex_bone_indices.numpy()
        dom = ix[np.arange(len(ix)), w.argmax(1)]
        m = dom == hi
        a, b = o_a["vertices"][0].numpy()[m], fwd(pose=p)["vertices"][0].numpy()[m]
        a0, b0 = a - a.mean(0), b - b.mean(0)
        u, _, vt = np.linalg.svd(b0.T @ a0)
        rr = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        ang = np.arccos(np.clip((np.trace(rr) - 1) / 2, -1, 1))
        vec = np.array([rr[2, 1] - rr[1, 2], rr[0, 2] - rr[2, 0], rr[1, 0] - rr[0, 1]])
        return vec / (2 * np.sin(ang)) if ang > 1e-8 else e
    m_map = np.stack([probe(k) for k in range(3)], axis=1)
    is_identity = float(np.abs(m_map - np.eye(3)).max()) < 0.08
    ok("local-ref rotation axes are WORLD-aligned, not bone-aligned", is_identity,
       "map is identity -- 'local Z' is world Z, NOT the bone roll axis",
       "map is not identity; recover the axis by probing before posing")

    print("\nRIG <-> DEPLOYMENT (Godot humanoid / LabRCSF)")
    twist = [b for b in labels if b.startswith(("lowerarm02", "upperarm02",
                                                "lowerleg02", "upperleg02"))]
    ok("twist bones exist in the source rig", len(twist) >= 4,
       "%d twist bones: %s" % (len(twist), ", ".join(sorted(twist)[:4])),
       "no twist bones found")
    unchecked("ANNY bone names -> LabRCSF canonical joints",
              "12 ANNY bones (8 twist + shoulder01.L/R + pelvis.L/R) have NO LabRCSF "
              "name; mapping is authored by hand and not machine-verified")

    print("\nSKINNING BOUNDARIES")
    w = model.vertex_bone_weights.numpy()
    ok("skin weights are a partition of unity", np.abs(w.sum(1) - 1.0).max() < 1e-4,
       "max |sum-1| = %.2e" % np.abs(w.sum(1) - 1.0).max(),
       "max |sum-1| = %.2e -- weights do not sum to 1" % np.abs(w.sum(1) - 1.0).max())
    ix = model.vertex_bone_indices.numpy()
    dom = ix[np.arange(len(ix)), w.argmax(1)]
    fi, ti = labels.index("lowerarm01.L"), labels.index("lowerarm02.L")

    # Score the twist PROFILE's linearity, not a single band. Two weaker versions of this
    # check both passed on a rig known to be broken:
    #   1. a dominance-share proxy -- passed outright;
    #   2. a single distal-band reading driven from the TWIST BONE -- 66.9 of an ideal
    #      78.8 deg, but nothing in the real pipeline drives a twist bone, because mocap
    #      supplies the wrist channel and no capture system outputs a twist joint.
    # Even wrist-driven, a single distal band is too lenient: with the roll dispersed the
    # STOCK rig reads 67 deg near the wrist while its mid-forearm is still flat. What
    # separates a real twist from a hinge at the wrist is LINEARITY along the forearm.
    rmses = {s: anny_rig.twist_ramp_rmse(model, s) for s in anny_rig.SIDES}  # default: wrist-driven, the shipping path
    worst = max(v[0] for v in rmses.values())
    ok("forearm transmits twist as a LINEAR ramp (both arms)", worst < 15.0,
       "RMSE L %.1f / R %.1f deg vs the anatomical ramp (%.1f = no twist at all)"
       % (rmses["L"][0], rmses["R"][0], anny_rig.ZERO_TWIST_RMSE),
       "RMSE %.1f deg -- twist is not reaching the forearm skin. This is the priority-1 "
       "defect; anny_rig.fix_forearm_twist() is the fix and build_corpus_model() applies "
       "it by default, so this failing means a consumer bypassed the canonical builder."
       % worst)
    # NEGATIVE CONTROL: the same measurement on an unfixed model must fail, or the pass
    # above proves nothing. This is the third revision of this check -- the two earlier
    # ones passed on known-broken input, which is worse than having no check at all.
    stock_rmse = anny_rig.twist_ramp_rmse(
        anny_rig.build_corpus_model(dtype=torch.float64, apply_twist_fix=False), "L")[0]
    ok("...and the guard REJECTS the unfixed rig (negative control)", stock_rmse > 15.0,
       "unfixed rig scores %.1f deg, well outside the %.0f deg gate" % (stock_rmse, 15.0),
       "unfixed rig scores %.1f deg and would PASS -- the guard is decoration" % stock_rmse)

    print("\nMIRROR SYMMETRY")
    xs = o_a["rest_vertices"][0].numpy()[:, 0]
    ok("rest pose is left/right symmetric", abs(xs.max() + xs.min()) < 0.02,
       "|xmax+xmin| = %.4f m" % abs(xs.max() + xs.min()),
       "|xmax+xmin| = %.4f m -- mirrored-axis bug" % abs(xs.max() + xs.min()))

    print("\nEXPRESSION INTERFACE")
    fa = getattr(model, "facial_action_labels", None)
    if fa is None:
        unchecked("ANNY facial_actions <-> the 52 blendshapes", "no facial_action_labels attribute")
    else:
        # BY NAME, NOT BY COUNT: two sets of 52 can be 52 different things.
        import check_facial_actions
        absent_anny, absent_standard, _, _ = check_facial_actions.compare(fa)
        ok("ANNY facial_actions are the 52 blendshapes BY NAME",
           not absent_anny and not absent_standard,
           "%d actions, every name in both" % len(fa),
           "%d actions; %d standard names absent from ANNY, %d the other way"
           % (len(fa), len(absent_anny), len(absent_standard)))

    print("\nCORPUS BOUNDARIES")
    unchecked("train / val split contamination",
              "checked by preflight_audit.py against a real corpus, not here")
    unchecked("renders <-> keypoint labels",
              "no render path exists yet; when it does, joints MUST come from "
              "bone_heads+vertices or rest_bone_heads+rest_vertices (never mixed)")

    haz = [r for r in RESULTS if r[1] == "HAZARD"]
    unc = [r for r in RESULTS if r[1] == "UNCHECKED"]
    print("\n%d interfaces: %d OK, %d HAZARD, %d UNCHECKED"
          % (len(RESULTS), len(RESULTS) - len(haz) - len(unc), len(haz), len(unc)))
    if unc:
        print("\nUNCHECKED interfaces are not passing interfaces. Outstanding:")
        for e, _, why in unc:
            print("  - %s (%s)" % (e, why))
    return 1 if haz else 0


if __name__ == "__main__":
    sys.exit(main())
