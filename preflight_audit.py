"""Pre-flight audit for the ANNY render corpus.

Premise (org rule): spending more compute is fast, RESTARTING is expensive. So
every class of silent corruption we have actually hit gets an automated check
that runs BEFORE the 800k-image run, not a promise to be careful.

Each check below exists because something real went wrong. The comment on each
names the incident, so nobody deletes a check without knowing what it caught.

Structural checks (NULLs, foreign keys, split hygiene) live in
anny_render_schema.validate(); this file adds the SEMANTIC and PHYSICAL ones,
which is where the expensive mistakes hide -- a corpus can be perfectly
well-formed and still have every sex label inverted.

Usage:  python preflight_audit.py <corpus-dir> [--sample 600]
Exit code is non-zero if any check fails, so it can gate a render run.

BATCH LADDER (Gall's law: a complex system that works evolved from a simple
system that worked). Never render 800k in one shot. Each rung must pass this
audit before the next is started, and each rung is cheap enough to throw away:

    rung   scenes    images   ~cost   what it is allowed to catch
    ----   ------    ------   -----   ---------------------------
    0          10       350   secs    plumbing: does anything render at all
    1         100     3,500   ~min    schema/IO shape, shard writing, resume
    2       1,000    35,000   ~10min  distribution sanity at small n
    3      10,000   350,000   ~hours  throughput, disk, long-run stability
    4      23,000   800,000   full    the corpus

Rung N+1 starts ONLY from a rung-N corpus that passed. Because ids are
deterministic and shards are disjoint, a rung is not thrown away when you climb
-- the next rung APPENDS to it. That is what makes "spend more compute" cheap
and "restart" avoidable.

NEGATIVE CONTROL: a check suite that never fails is worthless. `negative
control` mode corrupts a copy of the corpus in each way this project has
actually been burned and asserts the audit blocks it. Run it whenever a check
is added. It has already earned its place: it found that the split-contamination
check was nested under `if "scenes" in tables`, so it silently did not run
during identity generation -- the exact phase where contamination is created.
"""

import argparse
import sys

import numpy as np
import pandas as pd

RESULTS = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = True):
    RESULTS.append((name, ok, detail, fatal))
    mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def load_wide(corpus: str):
    ip = pd.read_parquet(f"{corpus}/identity_phenotype.parquet")
    ph = pd.read_parquet(f"{corpus}/phenotypes.parquet")
    ids = pd.read_parquet(f"{corpus}/identities.parquet")
    wide = (ip.assign(name=ip.phenotype_id.map(dict(zip(ph.phenotype_id, ph.name))))
              .pivot(index="identity_id", columns="name", values="value"))
    return ids, wide


def decode(wide: pd.DataFrame, n: int, seed: int = 0):
    """Decode identities to meshes. Returns (subset, vertices, model).

    Uses anny_rig.build_corpus_model so the audit measures the model the corpus ACTUALLY
    renders with -- twist fix applied, both arms. Constructing a bare anny.Anny here is
    how an audit ends up certifying a model nothing ships."""
    import torch
    import anny_rig
    model = anny_rig.build_corpus_model(dtype=torch.float32)
    sub = wide if n <= 0 or n >= len(wide) else wide.sample(n, random_state=seed)
    kw = {c: torch.tensor(sub[c].values, dtype=torch.float32) for c in sub.columns}
    pose = torch.eye(4)[None, None].repeat(len(sub), model.bone_count, 1, 1)
    with torch.no_grad():
        v = model(pose_parameters=pose, phenotype_kwargs=kw)["vertices"]
    return sub, v, model


def decode_summaries(wide: pd.DataFrame, n: int, chunk: int = 1000, seed: int = 0):
    """Per-identity bbox summaries for the WHOLE corpus, decoded in chunks.

    Holding 23,000 x 13,718 x 3 float32 vertices at once is 3.8 GB, but every check that
    uses geometry needs only per-identity extents and x-extremes, which are 6 floats each.
    Streaming those makes a full-corpus decode cost ~85 s and ~0 memory, which removes the
    reason to sample at all.

    WHY THIS REPLACED SAMPLING. A sampled audit can only see defects bigger than roughly
    3/n of the population (95% detection). At the old --sample 300 that is 10,000 ppm: the
    audit certified nothing below 1 identity in 100, while a defect touching 23 identities
    -- 800 images of an 800k corpus -- would slip through 95% of the time. A passing audit
    read like "the corpus is clean" when it only meant "no defect above 1%". Deciding to
    sample was never a considered trade; it was a default nobody priced."""
    import torch
    import anny_rig
    model = anny_rig.build_corpus_model(dtype=torch.float32)
    sub = wide if n <= 0 or n >= len(wide) else wide.sample(n, random_state=seed)
    ext, xlo, xhi = [], [], []
    for start in range(0, len(sub), chunk):
        part = sub.iloc[start:start + chunk]
        kw = {c: torch.tensor(part[c].values, dtype=torch.float32) for c in part.columns}
        pose = torch.eye(4)[None, None].repeat(len(part), model.bone_count, 1, 1)
        with torch.no_grad():
            v = model(pose_parameters=pose, phenotype_kwargs=kw)["vertices"]
        lo, hi = v.min(1).values.numpy(), v.max(1).values.numpy()
        ext.append(hi - lo)
        xlo.append(lo[:, 0])
        xhi.append(hi[:, 0])
    return sub, np.concatenate(ext), np.concatenate(xlo), np.concatenate(xhi), model


def detection_floor_ppm(n_sampled: int, n_total: int) -> float:
    """Smallest defect prevalence a run of this size can catch 95% of the time.

    1 - (1-p)^n >= 0.95  =>  p >= ~3/n. A full decode sees a single bad identity."""
    if n_sampled >= n_total:
        return 1e6 / max(n_total, 1)
    return 1e6 * 3.0 / max(n_sampled, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sample", type=int, default=0,
                    help="identities to decode; 0 = ALL (default, ~85 s for 23k)")
    ap.add_argument("--coverage-ppm", type=float, default=1000.0,
                    help="largest defect prevalence this run must be able to catch")
    ap.add_argument("--images", type=int, default=800000,
                    help="planned corpus image count, for translating ppm into images")
    args = ap.parse_args()

    ids, wide = load_wide(args.corpus)
    n_used = len(wide) if args.sample <= 0 else min(args.sample, len(wide))
    floor = detection_floor_ppm(n_used, len(wide))
    print(f"corpus: {len(ids)} identities; decoding {n_used}"
          f"{' (FULL)' if n_used >= len(wide) else ' (SAMPLED)'}")
    print(f"detection floor: {floor:.0f} ppm = {floor*len(wide)/1e6:.0f} identities"
          f" = {floor*args.images/1e6:.0f} of {args.images} planned images\n")

    print("NUMERIC INTEGRITY")
    # Silent NaN/Inf poisons a run without ever raising.
    finite = np.isfinite(wide.values).all()
    check("no NaN/Inf in phenotype values", bool(finite))
    # Phenotypes are normalized; out-of-range means a sampler or unit bug.
    in_range = bool(((wide.values >= 0.0) & (wide.values <= 1.0)).all())
    check("phenotypes within [0,1]", in_range,
          f"min {wide.values.min():.3f} max {wide.values.max():.3f}")
    # Duplicate ids would silently merge two identities on join.
    check("identity_id unique", ids.identity_id.is_unique,
          f"{len(ids) - ids.identity_id.nunique()} duplicates")
    # A constant column means the sampler collapsed -- 23k copies of one person.
    degenerate = [c for c in wide.columns if wide[c].nunique() < 10]
    check("no collapsed phenotype dimension", not degenerate, f"collapsed: {degenerate}")

    print("\nAUDIT POWER")
    # A sampled audit sees only defects bigger than ~3/n of the population. Stating that
    # floor is not pedantry: at the old --sample 300 it was 10,000 ppm, so a defect
    # touching 23 identities (800 images) slipped through 95% of the time while the audit
    # still printed all-PASS. An unstated floor lets a narrow check read like a broad one.
    check(f"run can catch defects at or below {args.coverage_ppm:.0f} ppm",
          floor <= args.coverage_ppm,
          f"floor is {floor:.0f} ppm -- decode ~{3e6/args.coverage_ppm:.0f} identities "
          f"or pass --sample 0 for the full corpus")

    print("\nUNITS AND AXES")
    sub, extent, x_lo, x_hi, model = decode_summaries(wide, args.sample)
    up_axis = int(np.argmax(extent.mean(0)))
    # INCIDENT: SOMA is centimetres/+Y-up, ANNY is metres/+Z-up. Mixing them
    # silently scales a corpus by 100x or lies flat on its side.
    check("up axis is Z (index 2)", up_axis == 2, f"largest mean extent on axis {up_axis}")
    stature = extent[:, up_axis]
    check("stature is in METRES not cm", bool(0.4 < np.median(stature) < 2.5),
          f"median {np.median(stature):.3f}")

    # INCIDENT: `rest_bone_heads` pairs with `rest_vertices`, NOT with `vertices` --
    # an identity `pose_parameters` is not the rest pose. Mixing them puts the skeleton
    # 54.9 mm off the mesh on a default adult and 500 mm off on a child (1022 mm under
    # rig="soma"). The error scales with distance from the default phenotype, so it
    # looks FINE on the body everyone spot-checks and is catastrophic on the small
    # identities -- and it already produced one false "the rig returns mesh and skeleton
    # in different frames" finding that had to be retracted. Any keypoint label exported
    # alongside a render inherits this silently.
    # The check runs at EXTREME phenotypes on purpose: at the default body it passes
    # even when the pairing is wrong, which is exactly how it hid the first time.
    # A systematic sweep of every joint-array x mesh-array pairing, across every rig and a
    # six-rung phenotype ladder, found exactly FOUR safe pairings -- all of them matched:
    #     rest_bone_heads + rest_vertices    bone_heads + vertices
    #     rest_bone_tails + rest_vertices    bone_tails + vertices
    # Every mismatched pairing is wrong. (godot-soma-twist/experiments/api_pairing_audit.py)
    #
    # NOTE ON THE METRIC: bbox containment is too weak to catch this. Under rig="mixamo",
    # the mismatched `rest_bone_heads + vertices` scores 100% containment on the default
    # adult and 0% on an infant -- it would pass every spot-check and ship. Nearest-vertex
    # PROXIMITY, as a fraction of stature, flags the same pairing at 7.2% on the default
    # body. So proximity is what is asserted here.
    import torch
    from scipy.spatial import cKDTree
    _extremes = {"default": {},
                 "child": dict(gender=0.5, height=0.2, muscle=0.3, weight=0.4, age=0.1),
                 "small": dict(gender=1.0, height=0.05, muscle=0.1, weight=0.2, age=0.7),
                 "large": dict(gender=0.0, height=0.95, muscle=0.9, weight=0.8, age=0.7)}

    def _proximity(joints, verts):
        """Median joint-to-nearest-vertex distance, as % of stature. A joint lives
        inside the body; one far from any vertex is floating in space."""
        stature = float((verts.max(0) - verts.min(0)).max())
        return 100.0 * float(np.median(cKDTree(verts).query(joints)[0])) / max(stature, 1e-9)

    _bad, _control, _worst = [], [], 0.0
    for _name, _ph in _extremes.items():
        _kw = {k: torch.tensor([v], dtype=torch.float32) for k, v in _ph.items()}
        _pose = torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1)
        with torch.no_grad():
            _o = model(pose_parameters=_pose, phenotype_kwargs=_kw)
        _rv, _v = _o["rest_vertices"][0].numpy(), _o["vertices"][0].numpy()
        _h = _o["rest_bone_heads"][0].numpy()
        _matched = _proximity(_h, _rv)          # correct pairing
        _mixed = _proximity(_h, _v)             # the mistake this check exists to catch
        if _matched > 1.0:
            _bad.append(f"{_name} {_matched:.1f}%")
        _control.append(_mixed)
        _worst = max(_worst, float(np.abs(_rv - _v).max()))
    check("matched skeleton/mesh pairing is tight at extreme phenotypes", not _bad,
          "; ".join(_bad) if _bad else "median joint-to-surface <1% of stature at all four")
    # NEGATIVE CONTROL: a check that cannot fail is decoration. Deliberately mis-pair the
    # arrays and assert the metric rejects it -- otherwise the check above proves nothing.
    check("...and the metric REJECTS the mismatched pairing (negative control)",
          max(_control) > 3.0,
          f"mismatched pairing reaches {max(_control):.1f}% of stature")
    check("identity pose is NOT assumed to be the rest pose", _worst > 0.005,
          f"they differ by {_worst*1000:.0f} mm -- pair rest with rest, posed with posed")

    print("\nPHYSICAL PLAUSIBILITY")
    adult = sub["age"].values > 0.55
    g = sub["gender"].values
    # INCIDENT: gender was read inverted (ANNY: 0=male, 1=female, verified from
    # anny_inverter's mixture weights). The inverted read produced males
    # SHORTER than females -- impossible, and it would have flipped the sex
    # label on all 800k images. This check is the tripwire for that class.
    male, female = adult & (g < 0.35), adult & (g > 0.65)
    # A check whose precondition is unmet must FAIL, never silently skip.
    # INCIDENT: the red test "single-sex population" (all gender=0.5) was NOT
    # caught, because both subgroups came out empty and the dimorphism checks
    # simply did not run -- silence that reads exactly like a pass. Same bug
    # class as the split-contamination check that was nested under an
    # unrelated condition. If the corpus cannot support a check, that is a
    # blocking fact about the corpus.
    have_sexes = check("sex subgroups present to test dimorphism",
                       male.sum() > 20 and female.sum() > 20,
                       f"male n={int(male.sum())}, female n={int(female.sum())}")
    if have_sexes:
        dm = float(np.median(stature[male]) - np.median(stature[female]))
        check("male median stature > female (sex label not inverted)", dm > 0,
              f"dimorphism {dm:+.3f} m")
        check("dimorphism magnitude plausible (0.05-0.20 m)", 0.05 < dm < 0.20,
              f"{dm:.3f} m")
    # Adults must land in a human range; children must actually be shorter.
    if adult.sum() > 20:
        check("adult stature p05..p95 within 1.35-2.05 m",
              bool(np.percentile(stature[adult], 5) > 1.35 and
                   np.percentile(stature[adult], 95) < 2.05),
              f"{np.percentile(stature[adult],5):.3f}..{np.percentile(stature[adult],95):.3f}")
    child = sub["age"].values < 0.25
    # Same rule: no children in the sample is itself a finding. The corpus is
    # meant to span infants to elders (that is why ANNY was chosen over the
    # lab-adult .b3d population), so an all-adult corpus is a defect.
    have_children = check("child subgroup present (corpus spans ages)",
                          child.sum() > 10, f"n={int(child.sum())}")
    if have_children:
        check("children shorter than adults",
              float(np.median(stature[child])) < float(np.median(stature[adult])),
              f"child {np.median(stature[child]):.3f} vs adult {np.median(stature[adult]):.3f}")
    # INCIDENT: a mirrored-axis bug put both arms on one side during the pose
    # retarget. In the REST pose the body must be near-symmetric about X.
    asym = float(np.abs(x_hi + x_lo).mean())
    check("rest pose is left/right symmetric", asym < 0.05, f"|xmax+xmin| mean {asym:.4f} m")

    print("\nTARGET POPULATION COVERAGE")
    # Reference adult means (published national surveys). Coverage = our
    # sampled 5-95% band contains the reference mean, per sex.
    ref = {"EU-tall(NL)": (1.84, 1.70), "EU-mid(IT/ES)": (1.76, 1.63), "US": (1.77, 1.63),
           "Oceania(AU/NZ)": (1.79, 1.66), "Japan": (1.72, 1.58)}
    if have_sexes:
        lo_m, hi_m = np.percentile(stature[male], [5, 95])
        lo_f, hi_f = np.percentile(stature[female], [5, 95])
        for k, (mm, ff) in ref.items():
            check(f"covers {k}", bool(lo_m <= mm <= hi_m and lo_f <= ff <= hi_f),
                  f"m {mm} in [{lo_m:.2f},{hi_m:.2f}], f {ff} in [{lo_f:.2f},{hi_f:.2f}]")

    print("\nRIG CORRECTNESS")
    # INCIDENT (priority 1): ANNY's stock rig cannot transmit forearm twist. Motion
    # capture supplies the wrist channel only -- no capture system outputs a twist bone --
    # and driving the wrist leaves the forearm skin still. The error lands on the
    # extremities, where a mocap corpus cannot afford it: fingers off by ~a golf ball at
    # 90 deg pronation while the torso sits at 0.00 mm, so the whole-mesh mean understates
    # it about 4x. Rendering 800k images from the stock rig bakes that in at the source.
    # Gate on the PROFILE's linearity, not one band: two weaker versions of this check
    # passed on the known-broken rig before this one. Both arms, because the mirrored-axis
    # class of bug is real here.
    import anny_rig
    _ramp = {s: anny_rig.twist_ramp_rmse(model, s)[0] for s in anny_rig.SIDES}
    check("forearm transmits twist as a linear ramp (both arms)",
          max(_ramp.values()) < 15.0,
          f"RMSE L {_ramp['L']:.1f} / R {_ramp['R']:.1f} deg "
          f"({anny_rig.ZERO_TWIST_RMSE:.0f} would mean no twist at all)")
    check("left/right forearms behave symmetrically",
          abs(_ramp["L"] - _ramp["R"]) < 3.0,
          f"|L-R| = {abs(_ramp['L'] - _ramp['R']):.1f} deg")

    print("\nDETERMINISM (the resumability contract)")
    # The whole append-don't-restart design rests on "same seed -> same output".
    # If this breaks, a resumed run silently produces DIFFERENT data than the
    # first pass and the corpus becomes a mixture of two populations.
    import anny_render_schema as S
    a = S.deterministic_id("x", 1, "y")
    b = S.deterministic_id("x", 1, "y")
    check("deterministic_id stable within process", a == b)
    import subprocess
    out = subprocess.run([sys.executable, "-c",
        "import anny_render_schema as S; print(S.deterministic_id('x',1,'y'))"],
        capture_output=True, text=True, cwd=".")
    check("deterministic_id stable ACROSS processes", out.stdout.strip() == str(a),
          "blake2b, not salted hash()")
    # Reproducibility needs actual vertices, so it uses a small fixed subset rather than
    # the streamed summaries -- two full decodes would be 7.6 GB to compare.
    _, va, _ = decode(wide, 200)
    _, vb, _ = decode(wide, 200)
    check("mesh decode is reproducible", bool(np.allclose(va.numpy(), vb.numpy())),
          "same phenotypes -> identical vertices (200-identity probe)")

    print("\nSPLIT HYGIENE")
    probs = [p for p in S.validate(args.corpus) if "missing relation" not in p]
    check("schema validator clean (NULLs, FKs, splits)", not probs, "; ".join(probs[:2]))
    check("val split is non-empty and small", 0 < (ids.split == "val").mean() < 0.2,
          f"{(ids.split=='val').mean():.3%}")

    fails = [r for r in RESULTS if not r[1] and r[3]]
    warns = [r for r in RESULTS if not r[1] and not r[3]]
    print(f"\n{len(RESULTS)} checks: {len(RESULTS)-len(fails)-len(warns)} pass, "
          f"{len(fails)} FAIL, {len(warns)} warn")
    if fails:
        print("BLOCKING -- do not start the render run:")
        for n, _, d, _ in fails:
            print(f"  - {n} ({d})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
