"""urn:oid:1.3.6.1.4.1.66606.1.2.2165.1 -- unified 8-stub MaskScore schema.

Every MaskScore stub emit (mesh, depth, pose, keypoints, multimodal, speech,
text, video) imports these schemas and vocabularies rather than restating
them. A drift between two stubs' definitions is a bug the module refuses to
let be written.

Three shared table schemas per stub, plus two extras satellites for
stub-specific attributes that would otherwise be nullable columns:

    <stub>.parquet                   root row per input asset
    <stub>_candidates.parquet        N candidates per root
    <stub>_scores.parquet            long-form: metric_name, metric_value
    <stub>_root_extras.parquet       stub-specific root fields (canonical_text,
                                     poses, frame_range, ...)
    <stub>_candidate_extras.parquet  stub-specific candidate fields
                                     (candidate_target_text, ...)

The 5 shipped Rung 1 stubs used a wide-form scores table (depth_l1, normal_l1,
normal_dot as separate columns) and view_index as a first-class column. This
module keeps view_index (per ETNF: -1 is a value, not a null, for per-clip
metrics) and moves the three depth metrics into the long-form metric_name.
Speech stub already matches; migration script is a separate follow-up.

References:
  urn:oid:1.3.6.1.4.1.66606.1.1.1173   MaskScore parent
  urn:oid:1.3.6.1.4.1.66606.1.2.2165   unified 8-stub emit refactor
"""
from __future__ import annotations

import pyarrow as pa


# ---- Interned vocabularies. Kept as tuples so tests iterate them. ----

TASK_TYPES: tuple[str, ...] = (
    "pose_change", "depth_edit", "expression_change",
    "cross_modal_compose", "speech_edit", "instruction_edit",
    "video_edit",
)

DIMENSIONS: tuple[str, ...] = ("instruction_following", "overall")

INPUT_ASSET_KINDS: tuple[str, ...] = (
    "soma_mesh", "aov_npz", "soma_rotations", "keypoints_json",
    "png", "wav_16khz", "text_utf8", "video_mp4_h264",
)

CANDIDATE_AXES: tuple[str, ...] = (
    "edit", "audio", "transcript", "view", "frame",
)

CANDIDATE_ASSET_KINDS: tuple[str, ...] = INPUT_ASSET_KINDS + ("webvtt",)

METRIC_NAMES: tuple[str, ...] = (
    "depth_l1", "normal_l1", "normal_dot",
    "wavlm_cos", "wer",
    "clip_cos", "lpips",
)

EXTRA_KINDS: tuple[str, ...] = (
    "text_utf8", "json_path", "frame_span", "npz_pointer",
)

# The candidate_kind vocabulary spans stubs. Not enumerated here (an audio
# rank_kind like `wrong_subject` and a mesh edit_kind like `facs_02_brow_up`
# do not share meaning), so validation matches the axis's known set below.

CANDIDATE_KINDS_BY_AXIS: dict[str, tuple[str, ...]] = {
    "audio": (
        "identity", "pitch_up_mild", "pitch_down_mild",
        "tempo_fast", "tempo_slow", "combined_subtle",
        "wrong_text_next", "wrong_text_fixed",
        "wrong_subject", "wrong_all",
    ),
    "transcript": (
        "voxtral", "whisper", "parakeet", "gemma-auto", "wav2vec2",
        "ipa-whisper-s", "ipa-whisper-b", "voxtral-ipa", "gemma-ipa",
        "allosaurus", "allosaurus-eng", "allosaurus-rus",
    ),
    "edit": (),   # populated once mesh emit refactor names its 15 edits
    "view": (),
    "frame": (),
}


# ---- Shared table schemas. ----

ROOT = pa.schema([
    ("key",              pa.string()),
    ("task_type",        pa.string()),
    ("dimension",        pa.string()),
    ("input_column",     pa.string()),
    ("input_asset",      pa.string()),
    ("input_asset_kind", pa.string()),
])

CANDIDATES = pa.schema([
    ("row_key",              pa.string()),
    ("candidate_axis",       pa.string()),
    ("rank",                 pa.int32()),
    ("candidate_asset",      pa.string()),
    ("candidate_asset_kind", pa.string()),
    ("candidate_kind",       pa.string()),
])

SCORES = pa.schema([
    ("row_key",        pa.string()),
    ("candidate_axis", pa.string()),
    ("candidate_rank", pa.int32()),
    ("view_index",     pa.int32()),   # -1 for per-clip metrics; ETNF: -1 is a value
    ("metric_name",    pa.string()),
    ("metric_value",   pa.float64()),
])

ROOT_EXTRAS = pa.schema([
    ("row_key",     pa.string()),
    ("extra_name",  pa.string()),
    ("extra_value", pa.string()),
    ("extra_kind",  pa.string()),
])

CANDIDATE_EXTRAS = pa.schema([
    ("row_key",        pa.string()),
    ("candidate_axis", pa.string()),
    ("candidate_rank", pa.int32()),
    ("extra_name",     pa.string()),
    ("extra_value",    pa.string()),
    ("extra_kind",     pa.string()),
])


# Standard suffix per stub, so a caller writes paths from one place.
def paths_for(stub: str, out_dir):
    """(root, candidates, scores, root_extras, candidate_extras) file paths."""
    import pathlib
    d = pathlib.Path(out_dir)
    return (
        d / f"maskscore_{stub}.parquet",
        d / f"maskscore_{stub}_candidates.parquet",
        d / f"maskscore_{stub}_scores.parquet",
        d / f"maskscore_{stub}_root_extras.parquet",
        d / f"maskscore_{stub}_candidate_extras.parquet",
    )


# ---- Validator. Rejects every ETNF-contract violation the schema encodes. ----

def _assert_schema(name: str, table, expected: pa.Schema) -> list[str]:
    problems = []
    got = table.schema
    got_names = {f.name for f in got}
    want_names = {f.name for f in expected}
    for missing in want_names - got_names:
        problems.append(f"{name}: missing column {missing!r}")
    for extra in got_names - want_names:
        problems.append(f"{name}: unexpected column {extra!r}")
    for f in expected:
        if f.name in got_names and got.field(f.name).type != f.type:
            problems.append(
                f"{name}: column {f.name!r} has type {got.field(f.name).type}, "
                f"expected {f.type}"
            )
    return problems


def _assert_no_nulls(name: str, table) -> list[str]:
    return [f"{name}: column {c!r} has {table[c].null_count} nulls"
            for c in table.column_names if table[c].null_count > 0]


def _assert_interned(name: str, table, col: str, vocab: tuple[str, ...]) -> list[str]:
    if col not in table.column_names or not vocab:
        return []
    seen = set(table[col].to_pylist())
    bad = seen - set(vocab)
    return [f"{name}: column {col!r} carries out-of-vocabulary value {v!r} "
            f"(known: {sorted(vocab)})" for v in sorted(bad)]


def _assert_join(name: str, table, key_col: str, roots: set[str]) -> list[str]:
    if table.num_rows == 0:
        return []
    orphans = set(table[key_col].to_pylist()) - roots
    return [f"{name}: {key_col} {v!r} does not join any root key"
            for v in sorted(orphans)]


def validate(root_t=None, cands_t=None, scores_t=None,
             root_extras_t=None, cand_extras_t=None) -> None:
    """Raise ValueError with every ETNF-contract violation across the tables.

    Every counter carries a control (a self-test constructs both a compliant
    and a broken set of tables and asserts the module accepts one and rejects
    the other).
    """
    problems: list[str] = []

    if root_t is not None:
        problems += _assert_schema("root", root_t, ROOT)
        problems += _assert_no_nulls("root", root_t)
        problems += _assert_interned("root", root_t, "task_type", TASK_TYPES)
        problems += _assert_interned("root", root_t, "dimension", DIMENSIONS)
        problems += _assert_interned("root", root_t, "input_asset_kind", INPUT_ASSET_KINDS)

    roots = set(root_t["key"].to_pylist()) if root_t is not None else set()

    if cands_t is not None:
        problems += _assert_schema("candidates", cands_t, CANDIDATES)
        problems += _assert_no_nulls("candidates", cands_t)
        problems += _assert_interned("candidates", cands_t, "candidate_axis", CANDIDATE_AXES)
        problems += _assert_interned("candidates", cands_t,
                                     "candidate_asset_kind", CANDIDATE_ASSET_KINDS)
        problems += _assert_join("candidates", cands_t, "row_key", roots)
        # candidate_kind checked per-axis
        for axis, vocab in CANDIDATE_KINDS_BY_AXIS.items():
            if not vocab:
                continue
            sub = cands_t.filter(pa.compute.equal(cands_t["candidate_axis"], axis))
            problems += _assert_interned(f"candidates[axis={axis}]",
                                         sub, "candidate_kind", vocab)

    if scores_t is not None:
        problems += _assert_schema("scores", scores_t, SCORES)
        problems += _assert_no_nulls("scores", scores_t)
        problems += _assert_interned("scores", scores_t, "candidate_axis", CANDIDATE_AXES)
        problems += _assert_interned("scores", scores_t, "metric_name", METRIC_NAMES)
        problems += _assert_join("scores", scores_t, "row_key", roots)

    if root_extras_t is not None:
        problems += _assert_schema("root_extras", root_extras_t, ROOT_EXTRAS)
        problems += _assert_no_nulls("root_extras", root_extras_t)
        problems += _assert_interned("root_extras", root_extras_t, "extra_kind", EXTRA_KINDS)
        problems += _assert_join("root_extras", root_extras_t, "row_key", roots)

    if cand_extras_t is not None:
        problems += _assert_schema("candidate_extras", cand_extras_t, CANDIDATE_EXTRAS)
        problems += _assert_no_nulls("candidate_extras", cand_extras_t)
        problems += _assert_interned("candidate_extras", cand_extras_t,
                                     "candidate_axis", CANDIDATE_AXES)
        problems += _assert_interned("candidate_extras", cand_extras_t,
                                     "extra_kind", EXTRA_KINDS)
        problems += _assert_join("candidate_extras", cand_extras_t, "row_key", roots)

    if problems:
        raise ValueError("schema validation failed:\n  " + "\n  ".join(problems))


# ---- Self-test with controls. ----

def _self_test() -> int:
    """Constructs a compliant set (must accept) and a broken set (must reject).

    Returns 0 on success, nonzero on failure. Invoked by `python -m
    maskscore_stub_schema` and by the corpus CI.
    """
    key = "rung1/speech/sub_100_trial_1_cmd_1"

    root = pa.table({
        "key":              [key],
        "task_type":        ["speech_edit"],
        "dimension":        ["instruction_following"],
        "input_column":     ["input_audio"],
        "input_asset":      ["build/audio/sub_100_trial_1_cmd_1.wav"],
        "input_asset_kind": ["wav_16khz"],
    }, schema=ROOT)

    cands = pa.table({
        "row_key":              [key, key],
        "candidate_axis":       ["audio", "audio"],
        "rank":                 pa.array([1, 9], type=pa.int32()),
        "candidate_asset":      ["build/clones/rank_01.wav", "build/clones/rank_09.wav"],
        "candidate_asset_kind": ["wav_16khz", "wav_16khz"],
        "candidate_kind":       ["identity", "wrong_subject"],
    }, schema=CANDIDATES)

    scores = pa.table({
        "row_key":        [key, key],
        "candidate_axis": ["audio", "audio"],
        "candidate_rank": pa.array([1, 9], type=pa.int32()),
        "view_index":     pa.array([-1, -1], type=pa.int32()),
        "metric_name":    ["wavlm_cos", "wavlm_cos"],
        "metric_value":   [0.92, 0.55],
    }, schema=SCORES)

    root_extras = pa.table({
        "row_key":     [key],
        "extra_name":  ["canonical_text"],
        "extra_value": ["turn on the lights"],
        "extra_kind":  ["text_utf8"],
    }, schema=ROOT_EXTRAS)

    cand_extras = pa.table({
        "row_key":        [key],
        "candidate_axis": ["audio"],
        "candidate_rank": pa.array([1], type=pa.int32()),
        "extra_name":     ["candidate_target_text"],
        "extra_value":    ["turn on the lights"],
        "extra_kind":     ["text_utf8"],
    }, schema=CANDIDATE_EXTRAS)

    # Positive: compliant tables validate clean.
    try:
        validate(root, cands, scores, root_extras, cand_extras)
    except ValueError as e:
        print(f"FAIL (positive control): {e}")
        return 1

    # Negative controls: each must be caught. If any escapes, the corresponding
    # validator branch is decoration certifying the defect (rule 2).
    controls = [
        ("out-of-vocab task_type",
         lambda: validate(root.set_column(
             root.schema.get_field_index("task_type"), "task_type",
             pa.array(["FAKE_TASK"], type=pa.string())))),
        ("out-of-vocab candidate_kind for audio axis",
         lambda: validate(root, cands.set_column(
             cands.schema.get_field_index("candidate_kind"), "candidate_kind",
             pa.array(["identity", "not_a_real_audio_kind"], type=pa.string())),
             None, None, None)),
        ("orphan row_key in candidates",
         lambda: validate(root, cands.set_column(
             cands.schema.get_field_index("row_key"), "row_key",
             pa.array([key, "rung1/speech/does_not_exist"], type=pa.string())),
             None, None, None)),
        ("out-of-vocab metric_name",
         lambda: validate(root, cands,
             scores.set_column(scores.schema.get_field_index("metric_name"),
                 "metric_name",
                 pa.array(["FAKE_METRIC", "wavlm_cos"], type=pa.string())),
             None, None)),
    ]
    for label, fn in controls:
        try:
            fn()
        except ValueError:
            continue
        print(f"FAIL (negative control escaped): {label}")
        return 1

    print("ok: schema module accepts compliant tables and rejects each planted defect")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
