# The image corpus, as designed on 2026-08-26

What the corpus is, what it currently contains, and what it does not. Every number below was
measured from the live code or the published data on the date in the title, and the command
that produced it is named beside it.

The shape of this document is deliberate: the cardinality table comes before the design
argument, because most axes are still at one sample and a design read without that reads as
a description of something that exists.

## What it is for

Two goals share the corpus and are not the same question.

| goal | camera generator | what it needs |
| --- | --- | --- |
| image / render coverage | `sphere_hammersley_sequence` | pitch and yaw nobody argued for |
| language camera conditioning | the 96-pose grid | a camera the vocabulary can say in words |

`render_grid.py` states why neither approximates the other: asking the sequence to reproduce
the grid costs 1,536 renders to land every cell within 5 degrees and never lands on one, and
a text instruction cannot be given a Hammersley yaw. They run independently.

**They are not the same size.** `render_view.py --views` defaults to **8**; `grid_cameras()`
enumerates **96**. That is a twelvefold gap between two goals that share one corpus.

Raising the sequence to 96 is not free, because it is not scale invariant in pitch:

| | n=8 | n=96 |
| --- | ---: | ---: |
| pitch min | -90.00 | -90.00 |
| pitch max | 56.44 | 80.44 |
| pitch mean | 4.66 | 15.40 |
| lands on a named elevation step | 3 of 8 | 3 of 96 |

So the image goal at 96 trains on a different distribution rather than a denser one. Two
written claims go stale the moment `--views` changes — `weft_loop.py`'s "five of the eight
views land between steps", and `check_view_selection.py`'s pitch table — and both should be
derived from `--views` rather than restated.

## Cardinality, which is where the design actually stands

| axis | levels available | levels in the corpus |
| --- | ---: | ---: |
| camera pose (grid) | 96 | **96** |
| camera pose (sequence) | any n | 8 by default |
| identity | 23,000 | **1** |
| body pose | a library | **1** |
| skin tone | 10 Monk levels | **1** |
| material model | PBR, MToon | **1** |
| depth / pose controls | one per frame | **1 of 96** |

Only the grid is populated. Every other axis is a single sample, so the corpus has **zero
degrees of freedom for within-cell variance** and no error bar can be computed from it.

`sample_identities.py` draws from ANNY's own WHO-calibrated shape prior, which spans infants
to elders and carries no ethnicity axis in its default six phenotypes. `phenotypes="all"`
loads eleven, three of them `african`, `asian`, `caucasian`. `anny_render_schema.py` still
states that ANNY has no ethnicity axis at all, which `gnm-anny-headfit` already recorded as
false, and that contradiction is live.

## What counts as a replicate

Not all variation is replication, and the distinction decides the budget.

| varying | measures | breaks the identity confound |
| --- | --- | --- |
| seed | generator stability with one body | no |
| style restyle | appearance of one body | no |
| body pose | one body under motion | partly |
| **identity** | **the thing the corpus claims** | **yes** |

A restyle is deliberately not an independent draw: `test_pose_survives_restyle.py` and
`verify_restyle.py` exist to confirm the pose did **not** move. Counting five styles of one
body as n=5 is the same body five times.

Identities are constructed synthetic, deterministic from a rig held locally, so they carry no
generated-synthetic conditions and are legal on the evaluation arm.

| replicates | renders | what it buys |
| ---: | ---: | --- |
| 1, today | 96 | nothing, zero df |
| 5 | 480 | a first estimate of across-body variance |
| 10 | 960 | a usable error bar per cell |
| 20 | 1,920 | enough spread for allocation to be worth doing |

**Allocation is equitable rather than equal.** Uniform seeds spend the same on a cell that is
already stable and on one where the model falls apart. Allocation proportional to measured
spread, with a floor of two so every cell keeps a degree of freedom, redistributes the same
budget by roughly tenfold between the cheapest and dearest cell. The floor is not optional: a
cell at one sample has no variance to allocate against.

Two cells are a different quantity entirely. Azimuths 45 and 135 produced no angle at all,
because the detector found no person, so what is measured there is a proportion. A proportion
sizes by its own detection floor — at 8 seeds a miss rate below 0.375 is invisible, at 24 it
is 0.125 — and allocating them as if they were angle cells measures nothing.

## Materials

The corpus renders two material models, and they are not convertible into one another.

**Photographic**, meaning the render itself, is Mitsuba `principled`. Its skin is a constant:

    "skin": {"type": "principled", "base_color": {"type": "rgb", "value": [0.76, 0.62, 0.54]},
             "roughness": 0.55, "metallic": 0.0}

That is `#E2CEC2` in sRGB, L\* 84.2, nearest Monk tone **2** at dE 8.6. Every frame in the
corpus carries it. Nothing samples the UV layout, so the material is flat regardless of what
the mesh holds, and the phenotype axes cannot reach it — `african`, `asian` and `caucasian`
are corrective blendshapes that move vertices, and vertices carry no colour.

**Anime** is MToon, which extends glTF `pbrMetallicRoughness` rather than replacing it. The
lit colour is `baseColorFactor`, the same field the PBR path reads, and the extension adds a
shade colour that PBR does not constrain, interpolated by
`linearstep(-1 + shadingToonyFactor, 1 - shadingToonyFactor, shading)`. So there is no
conversion to derive between the two; there is a second colour with no PBR counterpart.

**A single shade multiplier is not equitable**, and this is the finding the anime layer exists
to encode:

| Monk tone | flat x0.6 | solved for constant contrast |
| ---: | ---: | ---: |
| 1 | dE 17.28 | x0.7080, dE 12.00 |
| 5 | dE 15.15 | x0.6722, dE 12.00 |
| 10 | dE 4.83 | x0.1770, dE 12.00 |

The flat multiplier spans dE 4.83 to 17.28, a factor of 3.6. The cel terminator is the shading
cue a detector reads, so one multiplier across the scale builds a detection gap into the corpus
that would later present as a model defect. The multiplier is solved per tone instead, and the
dark end is bounded rather than clamped: holding dE at 12 needs shade L\* 2.9 at Monk 10, and
the gate prints what each tone achieved.

Applying the Monk scale to anime is a substitution and is recorded as one in
`anime-materials.cff`. The scale is defined on photographs of real skin; anime skin is an
artistic choice. No anime skin-tone standard exists to use instead — Danbooru's tags carry no
colour values and are defined against "the usual Eurasian skin tone" — so the cells give
coverage of a perceptual range and claim nothing about a population. Characters with
non-human skin have no Monk code and need a cell of their own.

**The tone sweep is uniform, not population weighted.** The Nem x Mila survey of 1,012 social
VR users finds 66 percent prefer an avatar unlike their physical selves, so avatar appearance
is chosen rather than inherited and no user demographic supplies a prior. That survey measures
no skin tone and no ethnicity, and its sample is 73.3 percent Japan, so weighting to it would
encode the sampling frame and call it reality.

## The UV layout, measured

| | |
| --- | ---: |
| basemesh | MakeHuman HM08, via MPFB2, CC0 |
| vertices | 13,718 |
| texture coordinates | 21,334 |
| faces | 27,420 |
| seam vertices | 1,165 |
| vertices after a seam split | 14,898 (+8.6%) |

ANNY authors UVs per face corner and Mitsuba carries one per vertex, so seams are split.
`uv_seams.py` does it and agrees with Mitsuba's own obj loader corner for corner to
4.469e-08, which is 0.37 of a float32 ulp.

**Mitsuba's obj loader flips V.** Compared as authored the deviation is 0.980786; flipped it
is at the float32 floor. A texture bound without the flip is mirrored vertically and still
looks like a body. That belongs beside rotation order, up axis and units as a convention that
is parsed rather than assumed.

Assets that ship with the basemesh: the UV'd `base.obj`, a 2048 square `sss.png`, and the
`enhanced_skin` and `makeskin` node trees. No diffuse albedo map ships. The node trees are
Blender shader graphs and Blender is blocklisted for reproducibility, so the shading has to be
re-authored as a Mitsuba BSDF rather than loaded.

## The shading model, checked against the reference

`mtoon.py` implements MToon 1.0 and is held against `@pixiv/three-vrm` pixel for pixel, in
headless Chromium, on a unit sphere under an orthographic camera framed so that pixel
(u, v) carries normal (u, v, sqrt(1 - u^2 - v^2)). No plateau finding and no fitting.

| shadingToonyFactor | px compared | over 4/255 | p99 | max |
| ------------------ | ----------- | ---------- | ------ | ------ |
| 0.9                | 3048        | 0          | 0.0020 | 0.0025 |
| 0.5                | 3048        | 0          | 0.0020 | 0.0021 |
| 0.0                | 3048        | 0          | 0.0020 | 0.0021 |
| 1.0                | 3048        | 2          | 0.0018 | 0.1351 |

One 8-bit readback step is 0.0039, so three of the four agree below the noise floor. The two
pixels at a hard ramp are the terminator rather than the model: a tessellated normal
disagrees with the analytic one by a hair and the step turns that into the whole base-to-shade
gap, which over pi is 0.1337 against the 0.1351 measured.

**THE ONE REAL DIFFERENCE IS A FACTOR OF 1/PI.** Unscaled, their pixels over ours is a
near-constant 0.3167 to 0.3183 against 1/pi = 0.31831, flat across every toony value. The
VRM 1.0 pseudocode ends `color = color * lightColor` with no such term and three.js applies
the Lambertian convention to direct light, so a corpus rendered with our integrator sits pi
times brighter than the same material in a three-vrm viewer.

That is not cosmetic here. The tone ladder solves for a dE between plateaus and CIELAB is
not scale invariant, so the same material at two exposures does not give the same dE. The
targets have to be verified at the exposure the viewer uses rather than at ours, and that
check does not exist yet.

## Constructed and generated, kept apart

Renders, depth maps and pose skeletons are **constructed**: deterministic from assets held
locally, labels true by construction, no generative model involved. They are legal on the
evaluation arm.

Restyles are **generated** and carry all five conditions. `restyle_qwen.py` writes
`corpus_eligible: false` for itself, because Qwen-Image-Edit is 57.7 GB and runs here only
quantised. The OmniGen2 edit path at bf16 fits a 24 GB card and can satisfy condition 5, but
condition 3 forbids it being the sole distribution and condition 4 keeps it out of evaluation
entirely.

The separation is structural: constructed data is published as `anny-render-corpus` and
generated output as `anny-render-corpus-generated`, two repositories rather than two folders.

## Gates, and what each can reject

| gate | controls | rejects |
| --- | ---: | --- |
| `uv_seams.py` | 22 | a naive per-vertex bind, a dropped V flip, a folded island, a non-deterministic split |
| `check_anime_materials.py` | 5 | a flat shade multiplier, a light-only scale, an unreachable target silently clamped |
| `compare_camera_obedience.py` | 4 | two slopes compared across different view sets |
| `mtoon.py` | 11 | a ramp that drifted from the VRM 1.0 pseudocode |
| `mtoon_integrator.py` | 8 | a render where the shade plateau never reaches film |
| `check_mtoon_reference.py` | 5 | a port that disagrees with three-vrm beyond quantisation |
| `check_controls.py` | — | depth and pose disagreeing with the render in pixels |
| `check_view_selection.py` | — | a hand-picked subset of the sequence |
| `preflight_audit.py` | 29 checks | a render run that should not start |
| `test_preflight.py` | red/green | a preflight check that cannot fail |

`uv_seams.py` was mutation tested: five defects introduced into the split and the flip, each
caught by at least two controls, none surviving.

## What is not measured, named and counted

1. **Within-cell variance, on every axis.** Every cell is n=1, so no error bar exists anywhere
   in the corpus. The slope difference between the base model and the adapter, 0.005 against
   0.099 on the six views both scored, is two point estimates rather than a difference.
2. **Depth conditioning.** Conditions B and C were generated for 3 of 8 azimuths and never
   fitted. There is no `azimuth_recovery_B.json` or `_C.json`, so the question the three
   condition ladder was built to answer is open.
3. **Controls coverage.** 1 frame of 96 carries a depth map and a pose skeleton, and they are
   exact rather than estimated, so the gap is a run that has not happened rather than a
   method that does not exist.
4. **Provenance of published generated output.** `ladder/ladder.json` records 4 views against
   14 images in that directory, so 10 published images have no run record.
5. **The exposure the tone targets are verified at.** The ladder solves in material space
   and the renderer is pi times brighter than a three-vrm viewer, so the rendered dE has
   not been checked at the exposure a viewer uses.
9. **Skin tone in the render path.** The UV reaches the model and stops there; nothing samples
   it, so the anime tone layer has no renderer to drive yet.
6. **Subsurface.** The scene material is `principled` with no subsurface, while an sss map
   ships unused. Dark and light albedos do not merely scale in brightness, so an albedo-only
   tone sweep understates real variation.
7. **Files under 100 non-blank lines** are outside the comment-density ladder, and
   `comment_density.py` is itself under that floor.
8. **Demographic coverage.** ANNY offers three ethnicity axes and GNM four classes, and both
   omit South Asian, Hispanic or Latino, and Pacific Islander. The last is a stated target
   population. Neither set is coverage and neither should be reported as such.

## Order of work that follows from this

Depth first, because `make_controls.py` writes it from the frame's own sidecar camera and the
posed rig without re-running the path tracer, and it unlocks two thirds of an experiment that
is already designed. Then the UV bind, because the tone axis has no renderer without it. Then
identities, because they are the only replication that breaks the confound. Coverage last: the
sequence at 96 changes what the image goal trains on, and that is a decision to take
deliberately rather than to inherit from a default.
