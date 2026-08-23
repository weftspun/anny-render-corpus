"""Schema for the ANNY synthetic render corpus (~800k images / ~23k identities).

Design premise: generating more data is cheap, RESTARTING is expensive. So the
schema is fixed first and is built to be appended to, resumed, and audited --
never migrated mid-run.

Format: ETNF zstd parquet (org rule). Concretely that means
  * repeated text interned into its own relation (bone names, phenotype names,
    clip names, licenses) -- never repeated per row
  * optional/sparse attributes in satellite relations, so NO column is nullable
  * entity facts separate from observation facts
  * no derivable columns, except one deliberate materialization (2D keypoints,
    flagged below with its justification)

Five properties make a run resumable and extensible, which is the whole point:
  1. deterministic IDs      - an id is a pure function of its inputs, so a
                              re-run of the same scene produces the same id and
                              appends idempotently instead of duplicating
  2. seeds stored per scene - any single image can be regenerated alone
  3. disjoint shard ranges  - parallel workers write different files, never
                              coordinate, never corrupt a shared file
  4. run provenance         - anny version + renderer + git sha per batch, so a
                              corpus assembled from several runs stays auditable
  5. license lineage        - pose -> clip -> source dataset -> license, so
                              attribution survives into the trained artifact

Split hygiene (memory: train-split-only): `split` is assigned at IDENTITY and
at SOURCE-SUBJECT level, never per image. Two renders of the same identity, or
of two identities sampled from the same AddBiomechanics subject, can never land
on opposite sides of the train/val line.
"""

import pyarrow as pa

# --------------------------------------------------------------------------
# Interned vocabularies (entity relations for repeated text)
# --------------------------------------------------------------------------

BONES = pa.schema([
    ("bone_id", pa.int16()),            # index into ANNY's bone_labels
    ("name", pa.string()),              # e.g. "lowerarm02.L"
    ("parent_bone_id", pa.int16()),     # -1 for root; -1 is a value, not a NULL
])

PHENOTYPES = pa.schema([
    ("phenotype_id", pa.int16()),
    ("name", pa.string()),              # gender, age, muscle, weight, height, proportions
])
# GENDER SEMANTICS -- verified from ANNY's own source, not assumed. In
# anny_inverter.py the mixture weights are
#     log_mix_boys  = log1p(-gender) + boys_logp
#     log_mix_girls = log(gender)    + girls_logp
# so the boys' weight is (1 - gender) and the girls' weight is gender:
#     gender = 0.0 -> MALE,  gender = 1.0 -> FEMALE.
# Recorded here because reading it the other way inverts every downstream
# label. Caught in practice: an inverted read produced "male" medians SHORTER
# than "female" (1.630 m vs 1.762 m), which is impossible; corrected it gives
# 1.762 / 1.630 m and 0.132 m of dimorphism, matching real populations.
#
# CROSS-MODEL TRAP -- google/GNM USES THE OPPOSITE CONVENTION. In
# google/GNM `gnm/shape/semantic_sampler.py`:
#     class Gender(enum.IntEnum):  FEMALE = 0;  MALE = 1
# while ANNY is 0 = male, 1 = female. The two are EXACTLY INVERTED. Any future
# pipeline that samples an ANNY body and a GNM head for the same character must
# convert (gnm_gender = 1 - anny_gender), or every character gets a head of the
# opposite sex to its body -- a defect that is invisible in the data and only
# shows up on inspection of renders.
#
# GNM reference (evaluated 2026-08-14, github.com/google/GNM, Apache-2.0):
#   * ships `gnm/shape` ONLY = "GNM Head", a head/face model. There is NO body
#     model yet (the roadmap promises a wider suite), so it cannot replace
#     ANNY as the identity/population source for this corpus.
#   * `IdentitySampler.sample_identity(gender_class, ethnicity_class, n)` and
#     `blend_identities(gender_weights, ethnicity_weights, n)` -- the only
#     ETHNICITY-conditioned sampler available to us; ANNY has no ethnicity axis
#     at all. That makes GNM the natural complement for FACE identity diversity
#     once body and head are composed.
#   * Its ethnicity taxonomy is coarse and incomplete for our targets:
#     MIDDLE_EASTERN, ASIAN, WHITE, BLACK (4 classes). No South Asian, no
#     Hispanic/Latino, no Pacific Islander -- the last is a real gap against
#     the Oceania target population. Do not treat these 4 classes as coverage.
#   * SOMA-X has NO GNM bridge (it maps SMPL / SMPL-X / MHR / Anny /
#     GarmentMeasurements), so head-to-body composition is unsolved work, not a
#     library call.

LOCAL_CHANGES = pa.schema([
    ("change_id", pa.int16()),
    ("name", pa.string()),              # ANNY local_changes / corrective blendshape name
])

LICENSES = pa.schema([
    ("license_id", pa.int16()),
    ("name", pa.string()),              # "CC-BY-4.0", "Apache-2.0", ...
    ("url", pa.string()),
])

SOURCE_DATASETS = pa.schema([
    ("dataset_id", pa.int16()),
    ("name", pa.string()),              # "100STYLE", "o3de-motion-matching", "addbiomechanics"
    ("license_id", pa.int16()),
])

# --------------------------------------------------------------------------
# Identity: the characteristics half. Constant per character by construction.
# --------------------------------------------------------------------------

# Long form, not wide columns: adding a phenotype in a later ANNY version must
# not change the schema (that would force a restart, the exact failure we are
# designing against).
IDENTITIES = pa.schema([
    ("identity_id", pa.int32()),
    ("seed", pa.int64()),               # regenerates this identity alone
    ("source_subject", pa.string()),    # .b3d subject the anthropometrics came from
    ("split", pa.string()),             # "train" | "val" -- assigned HERE, never per image
])

IDENTITY_PHENOTYPE = pa.schema([
    ("identity_id", pa.int32()),
    ("phenotype_id", pa.int16()),
    ("value", pa.float32()),
])

# Sparse by design: an absent row means 0.0. That is a documented default, not
# a NULL, and it keeps ~1184 possible blendshapes from becoming 1184 columns.
IDENTITY_LOCAL_CHANGE = pa.schema([
    ("identity_id", pa.int32()),
    ("change_id", pa.int16()),
    ("value", pa.float32()),
])

# --------------------------------------------------------------------------
# Pose: the motion half. Shared across identities -- this reuse is what makes
# 23k identities x 35 views reach 800k images without 800k unique poses.
# --------------------------------------------------------------------------

MOTION_CLIPS = pa.schema([
    ("clip_id", pa.int32()),
    ("dataset_id", pa.int16()),
    ("clip_name", pa.string()),
    ("fps", pa.float32()),
])

POSES = pa.schema([
    ("pose_id", pa.int32()),
    ("clip_id", pa.int32()),
    ("frame_index", pa.int32()),        # provenance back to the exact source frame
])

# Long form again: 104 rows per pose. Rotation vectors (axis-angle) because
# they are what ANNY's forward pass consumes and they have no gimbal issue.
POSE_ROTATIONS = pa.schema([
    ("pose_id", pa.int32()),
    ("bone_id", pa.int16()),
    ("rx", pa.float32()), ("ry", pa.float32()), ("rz", pa.float32()),
])

# --------------------------------------------------------------------------
# Scene = identity x pose x environment. One scene yields many camera views.
# --------------------------------------------------------------------------

ENVIRONMENTS = pa.schema([
    ("env_id", pa.int16()),
    ("name", pa.string()),
    ("hdri_asset", pa.string()),
    ("license_id", pa.int16()),
])

SCENES = pa.schema([
    ("scene_id", pa.int32()),
    ("identity_id", pa.int32()),
    ("pose_id", pa.int32()),
    ("env_id", pa.int16()),
    ("seed", pa.int64()),               # regenerates every view of this scene
    ("split", pa.string()),             # must equal identities.split (validator enforces)
])

CAMERAS = pa.schema([
    ("camera_id", pa.int64()),
    ("scene_id", pa.int32()),
    ("view_index", pa.int16()),
    ("fx", pa.float32()), ("fy", pa.float32()),
    ("cx", pa.float32()), ("cy", pa.float32()),
    ("width", pa.int16()), ("height", pa.int16()),
    # extrinsics as quaternion + translation: 7 floats, no 4x4 redundancy
    ("qx", pa.float32()), ("qy", pa.float32()), ("qz", pa.float32()), ("qw", pa.float32()),
    ("tx", pa.float32()), ("ty", pa.float32()), ("tz", pa.float32()),
])

# --------------------------------------------------------------------------
# Renders and their payload
# --------------------------------------------------------------------------

RENDER_RUNS = pa.schema([
    ("run_id", pa.int16()),
    ("anny_version", pa.string()),
    ("renderer", pa.string()),          # e.g. "nvdiffrast-0.4.0"
    ("git_sha", pa.string()),
    ("started_utc", pa.string()),       # ISO8601; string keeps it engine-neutral
])

RENDERS = pa.schema([
    ("render_id", pa.int64()),
    ("camera_id", pa.int64()),
    ("run_id", pa.int16()),
])

# The payload. Sharded by scene range so parallel workers never touch the same
# file. Same proven pattern as the COCO conversion: image bytes live INSIDE
# parquet, so a consumer can filter on metadata and pull only the row groups it
# needs -- no archive to decompress first.
RENDER_DATA = pa.schema([
    ("render_id", pa.int64()),
    ("image", pa.binary()),             # PNG/WebP bytes
])

# --------------------------------------------------------------------------
# Ground-truth labels. Free here: they come from the parameters, not from
# another learned model (memory: no secondary generation).
# --------------------------------------------------------------------------

# DELIBERATE MATERIALIZATION, the one exception to "no derivable columns":
# 2D keypoints are a pure projection of pose x camera, so strictly they are
# derivable. They are stored anyway because every training step reads them and
# recomputing per epoch is pointless work. Recorded here as a conscious
# trade, and they are regenerable from POSE_ROTATIONS + CAMERAS if ever suspect.
KEYPOINTS_2D = pa.schema([
    ("render_id", pa.int64()),
    ("bone_id", pa.int16()),
    ("x", pa.float32()), ("y", pa.float32()),
    # int8, NOT bool. Three independent reasons, each sufficient:
    #
    #   2  visible
    #   1  projects inside the silhouette and fails the depth test, so present
    #      and hidden
    #   0  outside the frame entirely
    #
    # COCO carries the same three states, and masked training has to tell NOT
    # ANNOTATED from ANNOTATED AS OCCLUDED. A boolean cannot: you skip the first
    # and learn the second.
    #
    # The middle state is also the one a render knows and a real dataset guesses.
    # Z-test each projected joint against the rendered depth and the answer is
    # exact, which is the whole argument for rendering labels rather than
    # annotating them. Collapsing it to a boolean throws away the part that
    # makes occlusion learnable.
    ("visibility", pa.int8()),
])

SEGMENTATION = pa.schema([
    ("render_id", pa.int64()),
    ("mask", pa.binary()),              # PNG-encoded label map
])

# Pixal3D's half of the corpus. RFD 0122 fixes the output set at three: the
# image, the keypoint positions, and the 3D shape. The first two were here and
# this was not, so the renderer had nowhere to put the one thing the second
# consumer asks for.
#
# Geometry rather than a path: the same argument RENDER_DATA already makes for
# image bytes. A consumer filters on metadata and pulls the row groups it needs,
# with no second store to keep in step.
MESHES = pa.schema([
    ("render_id", pa.int64()),
    ("geometry", pa.binary()),          # USD, per RFD 0053
])

# --------------------------------------------------------------------------
# Generated synthetic, kept apart from everything above. Every relation before
# this point is CONSTRUCTED -- rendered deterministically from assets we hold,
# labels true by construction. What follows is sampled from a generative model,
# and CLAUDE.md admits it only under four conditions. Two of them are structural
# rather than procedural, so they are built here instead of being remembered:
#
#   condition 1  the generating model, checkpoint and conditioning recorded
#                WITH THE DATA. That is what these four relations are. A hash
#                in a server log, or a JSON file beside a directory of PNGs, is
#                next to the data rather than with it: the two part company the
#                first time somebody copies the images.
#   condition 2  stored and manifested separately from constructed and real
#                data, never merged into an undifferentiated pool. That is why
#                the pixels land in EDITED_RENDERS and never in RENDER_DATA,
#                and why validate() fails a corpus that merges them.
#
# Conditions 3 (not the sole distribution) and 4 (evaluate on real or
# constructed only) are properties of a training run, not of a table, so they
# are not represented here and are not silently claimed.
# --------------------------------------------------------------------------

EDIT_MODELS = pa.schema([
    ("edit_model_id", pa.int16()),
    ("repo_id", pa.string()),           # "OmniGen2/OmniGen2"
    # THE RESOLVED COMMIT, NEVER A MOVING TAG. `main` is not a checkpoint: it
    # names whatever the vendor last pushed, so a corpus stamped with it stops
    # resolving to the weights that produced it -- the same failure that
    # blocklists hosted-API generators, arriving through a slower door.
    ("revision", pa.string()),
])

# Interned because a prompt is long repeated text and it repeats once per image.
# Positive and negative prompts share this relation: they are the same kind of
# thing, and giving each its own table would duplicate the vocabulary.
EDIT_PROMPTS = pa.schema([
    ("prompt_id", pa.int32()),
    ("text", pa.string()),
])

EDIT_RUNS = pa.schema([
    ("edit_run_id", pa.int32()),
    ("edit_model_id", pa.int16()),
    ("prompt_id", pa.int32()),
    ("negative_prompt_id", pa.int32()),
    # "bf16" | "nf4". A bare string, following `identities.split`, because a
    # two-value closed enum interns into a vocabulary nobody reads.
    #
    # NO `corpus_eligible` COLUMN, and its absence is the ETNF rule rather than
    # an omission. Eligibility is a pure function of this field -- condition 5
    # says quantised weights do not produce corpus data -- so a column for it
    # would be derivable, and a derivable column is a second place the fact
    # lives and a second place it can disagree. validate() derives it instead.
    ("precision", pa.string()),
    ("steps", pa.int16()),
    ("text_guidance_scale", pa.float32()),
    ("image_guidance_scale", pa.float32()),
    ("started_utc", pa.string()),
    # NO `domain` COLUMN either. "photographic" and "colour-sketch" are names
    # for prompts, and `prompt_id` already carries which one ran.
])

# The generated pixels, and the join back to what they were generated FROM.
#
# NO KEYPOINT ROWS OF ITS OWN. An edit that moved a joint is discarded by T07's
# verification rather than relabelled, so a surviving edited frame carries its
# source render's labels exactly. Copying them here would materialise a
# derivable fact and invite the copy to drift from the original.
EDITED_RENDERS = pa.schema([
    ("edit_id", pa.int64()),
    ("render_id", pa.int64()),          # the constructed frame this was edited from
    ("edit_run_id", pa.int32()),
    ("seed", pa.int64()),               # regenerates this image alone
    ("image", pa.binary()),
])

RELATIONS = {
    "bones": BONES, "phenotypes": PHENOTYPES, "local_changes": LOCAL_CHANGES,
    "licenses": LICENSES, "source_datasets": SOURCE_DATASETS,
    "identities": IDENTITIES, "identity_phenotype": IDENTITY_PHENOTYPE,
    "identity_local_change": IDENTITY_LOCAL_CHANGE,
    "motion_clips": MOTION_CLIPS, "poses": POSES, "pose_rotations": POSE_ROTATIONS,
    "environments": ENVIRONMENTS, "scenes": SCENES, "cameras": CAMERAS,
    "render_runs": RENDER_RUNS, "renders": RENDERS, "render_data": RENDER_DATA,
    "keypoints_2d": KEYPOINTS_2D, "segmentation": SEGMENTATION,
    "meshes": MESHES,
    "edit_models": EDIT_MODELS, "edit_prompts": EDIT_PROMPTS,
    "edit_runs": EDIT_RUNS, "edited_renders": EDITED_RENDERS,
}

# The generated half, named rather than inferred from a prefix. validate()
# reads this to check condition 2, and a name-level guess is the failure mode
# `extract_poses.py`'s docstring warns about three times over.
GENERATED_RELATIONS = {"edit_models", "edit_prompts", "edit_runs", "edited_renders"}

# Condition 5: these produce device-sizing evidence, not corpus data.
QUANTISED_PRECISIONS = {"nf4", "int8", "int4", "q4_k_m", "gguf"}

# Foreign keys, checked by validate().
FOREIGN_KEYS = [
    ("identity_phenotype", "identity_id", "identities", "identity_id"),
    ("identity_phenotype", "phenotype_id", "phenotypes", "phenotype_id"),
    ("identity_local_change", "identity_id", "identities", "identity_id"),
    ("identity_local_change", "change_id", "local_changes", "change_id"),
    ("motion_clips", "dataset_id", "source_datasets", "dataset_id"),
    ("poses", "clip_id", "motion_clips", "clip_id"),
    ("pose_rotations", "pose_id", "poses", "pose_id"),
    ("pose_rotations", "bone_id", "bones", "bone_id"),
    ("scenes", "identity_id", "identities", "identity_id"),
    ("scenes", "pose_id", "poses", "pose_id"),
    ("scenes", "env_id", "environments", "env_id"),
    ("cameras", "scene_id", "scenes", "scene_id"),
    ("renders", "camera_id", "cameras", "camera_id"),
    ("renders", "run_id", "render_runs", "run_id"),
    ("render_data", "render_id", "renders", "render_id"),
    ("keypoints_2d", "render_id", "renders", "render_id"),
    ("source_datasets", "license_id", "licenses", "license_id"),
    # Condition 1, as integrity rather than as a promise. An edited image whose
    # run, model or prompt has gone missing fails here instead of entering a
    # corpus unprovenanced.
    ("edit_runs", "edit_model_id", "edit_models", "edit_model_id"),
    ("edit_runs", "prompt_id", "edit_prompts", "prompt_id"),
    ("edit_runs", "negative_prompt_id", "edit_prompts", "prompt_id"),
    ("edited_renders", "render_id", "renders", "render_id"),
    ("edited_renders", "edit_run_id", "edit_runs", "edit_run_id"),
]


def deterministic_id(*parts) -> int:
    """Stable 63-bit id from the inputs that define a row. Same inputs -> same
    id on any machine, any run, so re-running a shard appends idempotently
    rather than duplicating. blake2b (not hash()) because Python's hash is
    salted per process."""
    import hashlib
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") & 0x7FFFFFFFFFFFFFFF


def validate(root: str) -> list:
    """ETNF + integrity check over a corpus directory. Returns a list of
    problems; empty means clean. Run this before any training consumes it."""
    import os
    import pyarrow.parquet as pq

    problems = []
    tables = {}
    for name, schema in RELATIONS.items():
        path = os.path.join(root, f"{name}.parquet")
        if not os.path.exists(path):
            # payload/label relations may be sharded into a subdirectory
            shard_dir = os.path.join(root, name)
            if os.path.isdir(shard_dir):
                continue
            problems.append(f"missing relation: {name}")
            continue
        t = pq.read_table(path)
        tables[name] = t
        # ETNF: no nulls anywhere
        for col in t.column_names:
            if t[col].null_count:
                problems.append(f"{name}.{col}: {t[col].null_count} NULLs violate ETNF")
        # declared schema honoured
        for field in schema:
            if field.name not in t.column_names:
                problems.append(f"{name}: missing column {field.name}")

    # Foreign key integrity
    for child, ckey, parent, pkey in FOREIGN_KEYS:
        if child not in tables or parent not in tables:
            continue
        have = set(tables[parent][pkey].to_pylist())
        missing = set(tables[child][ckey].to_pylist()) - have
        if missing:
            problems.append(f"{child}.{ckey}: {len(missing)} values absent from {parent}.{pkey}")

    # Split hygiene, part 1: a scene must inherit its identity's split.
    if "scenes" in tables and "identities" in tables:
        id_split = dict(zip(tables["identities"]["identity_id"].to_pylist(),
                            tables["identities"]["split"].to_pylist()))
        bad = sum(1 for i, s in zip(tables["scenes"]["identity_id"].to_pylist(),
                                    tables["scenes"]["split"].to_pylist())
                  if id_split.get(i) != s)
        if bad:
            problems.append(f"scenes: {bad} rows whose split disagrees with their identity")

    # Split hygiene, part 2: a source subject must never span both splits.
    # BUG FIXED (found by the preflight negative control): this used to be
    # nested under `if "scenes" in tables`, so on a corpus that has identities
    # but no scenes yet -- exactly the state during identity generation, when
    # contamination is introduced -- the check silently did not run and a
    # deliberately contaminated fixture passed. It only depends on identities.
    if "identities" in tables:
        subj = {}
        for s, subject in zip(tables["identities"]["split"].to_pylist(),
                              tables["identities"]["source_subject"].to_pylist()):
            subj.setdefault(subject, set()).add(s)
        leaked = [k for k, v in subj.items() if len(v) > 1]
        if leaked:
            problems.append(f"CONTAMINATION: {len(leaked)} source subjects span both splits")

    # Condition 2: the generated half stays a separate pool. The way this gets
    # violated is not a merge anybody decides on -- it is a convenience write
    # that puts an edited PNG into render_data so one loader can read one table.
    # Then the corpus has generated pixels in the constructed relation, with
    # nothing recording which rows they are.
    if "render_data" in tables and "edited_renders" in tables:
        merged = (set(tables["render_data"]["render_id"].to_pylist())
                  & set(tables["edited_renders"]["edit_id"].to_pylist()))
        if merged:
            problems.append(
                f"POOL MERGE: {len(merged)} edit_ids also present in render_data.render_id; "
                "generated frames must not share the constructed pool (condition 2)")

    # Condition 5, derived rather than stored. A quantised run is device-sizing
    # evidence, so its output is not corpus data -- and an UNMET PRECONDITION IS
    # A FAIL: a corpus carrying edited_renders with no edit_runs to say what
    # produced them is unprovenanced, which reads exactly like a clean corpus if
    # this is written as a skip.
    if "edited_renders" in tables:
        if "edit_runs" not in tables:
            problems.append(
                "edited_renders present with no edit_runs relation: generated frames whose "
                "precision cannot be established (condition 1 and condition 5 both unmet)")
        else:
            precision = dict(zip(tables["edit_runs"]["edit_run_id"].to_pylist(),
                                 tables["edit_runs"]["precision"].to_pylist()))
            ineligible = [r for r in tables["edited_renders"]["edit_run_id"].to_pylist()
                          if str(precision.get(r, "")).lower() in QUANTISED_PRECISIONS]
            if ineligible:
                problems.append(
                    f"NOT CORPUS DATA: {len(ineligible)} edited renders come from a quantised "
                    "run (condition 5); they are device-sizing evidence, not corpus")
            unknown = [r for r in tables["edited_renders"]["edit_run_id"].to_pylist()
                       if r not in precision]
            if unknown:
                problems.append(
                    f"{len(unknown)} edited renders name an edit_run that does not exist, so "
                    "their precision is unknown and unchecked")

    return problems


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        issues = validate(sys.argv[1])
        print("\n".join(issues) if issues else "clean: ETNF, FK integrity, split hygiene all OK")
    else:
        print(f"{len(RELATIONS)} relations, {len(FOREIGN_KEYS)} foreign keys")
        for n, s in RELATIONS.items():
            print(f"  {n:24s} {len(s)} cols")
