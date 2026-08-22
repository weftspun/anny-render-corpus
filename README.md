# dataflow-coco-gemx

The ANNY render corpus pipeline: schema, identity sampling, the canonical rigged model,
the audits that gate a render run, and the 100STYLE pose reader.

**Every number below is machine-checked.** Run `python check_readme_claims.py` — it
re-derives each figure from the live code and exits non-zero on drift. If this README is
wrong, that command says so. That is the point: a document that fails loudly when it stops
being true is worth trusting when it passes.

```
python check_readme_claims.py     # verify this README against reality
python preflight_audit.py <corpus>  # gate before a render run (full corpus, ~95 s)
python interface_audit.py         # the pipeline's edges
python test_preflight.py <corpus> # red/green: prove every check can fail
```

## What is here

| file | what it does |
|---|---|
| `anny_render_schema.py` | 19 ETNF relations, FK graph, `validate()`, deterministic ids |
| `anny_rig.py` | **the canonical model.** Every stage builds from here, never bare `anny.Anny` |
| `sample_identities.py` | 23,000 identities (22,511 train / 489 val) |
| `preflight_audit.py` | 29 semantic/physical checks; gates the render run |
| `test_preflight.py` | red/green — every check must be *able* to fail |
| `interface_audit.py` | the 17 named interfaces between components |
| `corpus_defect_rate.py` | quality as an exceedance **rate**, not a mean |
| `bvh_parse.py` | 100STYLE BVH reader + FK, no retarget opinions |
| `bvh_retarget_probe.py` | which retarget formulation transfers a pose — none clears the bind-orientation floor |
| `coco_zip_to_etnf.py`, `filter_coco_licenses.py` | COCO ingest, license filtering |

## The rig fix, in one line

ANNY's stock rig cannot transmit forearm twist: driving the wrist — the only channel motion
capture supplies — leaves the forearm skin nearly still. The corpus re-weights the forearm
as a linear elbow→wrist ramp landing on the **wrist bone**, so the ramp itself becomes the
twist distribution. **No twist bone, no runtime step.**

| | RMSE vs the anatomical ramp |
|---|---|
| stock rig | <!--claim:twist_rmse_stock_L=52.8 tol=3.0-->52.8° |
| **shipping (wrist ramp)** | <!--claim:twist_rmse_90_L=3.6 tol=1.0-->3.6° (L) / <!--claim:twist_rmse_90_R=4.1 tol=1.0-->4.1° (R) |
| "no twist at all" baseline | <!--claim:zero_twist_baseline=55.2 tol=0.5-->55.2° |

The re-weighting is provably rest-neutral — it only moves mass between bones that are both
at identity at rest: <!--claim:rest_pose_shift_mm=0.0 tol=0.001-->0.000 mm.

ANNY has <!--claim:anny_bone_count=104 tol=0.5-->104 bones.
100STYLE has <!--claim:bvh_clip_count=810 tol=0.5-->810 clips across 100 styles.

## Interfaces, not components

`interface_audit.py` names <!--claim:interfaces_total=17 tol=0.5-->17 interfaces, of which
<!--claim:interfaces_unchecked=5 tol=0.5-->5 are still UNCHECKED and reported loudly.

Every defect this project has hit lived at a boundary, never inside a component. Full list
and the recurring failure modes: **`weftspun/logbook/PITFALLS.md`**.

## Superseded — claims that do not hold

This section grows. It carries the most weight in the document: a reader who knows the dead
ends is better off than one who knows only the current answer.

| claim | why it does not hold |
|---|---|
| "no ratio works; RMS ~39° at every ratio" | measures about **world Z** rather than the bone roll axis — the local→world map is the identity, so "local Z" sits 55° off the forearm, and the run measures a bend rather than a pronation. |
| "rig=soma returns mesh and skeleton in different frames" | compares a **rest** skeleton against an **identity-pose** mesh. Paired correctly, containment reaches 100% for every rig and phenotype. |
| "~9.7° size-correlated thigh error" | a weak observable. Centroid direction reads +9.7° where principal-axis reads −4.0° on the same runs. Joint angle gives thigh 2.2°, arms <1°. |
| "neither BVH formulation transfers the pose" | rests on a 153 mm residual with **no baseline**. Two *rest* skeletons score 139.7 mm. `local` sits on the floor; the blocker is bind orientation. |
| "Delta Mush is available in `anny_rig`" | no such code exists there. `grep` finds zero occurrences. |

## Falsifiers

What would show these answers do not hold:

- **Twist fix** — a pronation angle where the wrist ramp exceeds ~15° RMSE, or an L/R
  asymmetry above 3°. Both are gated in `preflight_audit.py`.
- **Audit power** — a defect affecting fewer identities than the stated detection floor.
  The audit prints its own floor (43 ppm on a full decode) and **fails** if asked to
  certify below what its sample size can resolve.
- **Skeleton/mesh pairing** — any joint sitting >1% of stature from the nearest vertex at
  an extreme phenotype. Checked with a negative control that must reject the mispairing.

## COCO lineage

COCO person images (license-filtered, commercial-safe) → GEM-X →
SOMA-X pose + identity coefficients. feed the earlier line of work, which remains here (`filter_coco_licenses.py`:
val2017 523/5,000; train2017 12,620/118,287) ) and is right-sized for **evaluation and domain adaptation** rather than from-scratch
training. The training-scale source is self-generated synthetic ANNY renders, which the rest
of this repo builds.

## Open work

Tracked as issues, not prose — see this repo's issue list. Critical path is **#1**
(100STYLE bind-orientation correction) → poses → scenes → rung 0 of the render ladder.
