"""Red/green control for the generated-synthetic checks in anny_render_schema.validate().

Same protocol as test_preflight.py: for every check, prove both directions. A check that
cannot fail certifies the defect it was written to catch, and this file exists because the
three checks under test are exactly the kind that pass by doing nothing -- two of them fire
only when a relation is present, and the third fires only when two relations disagree.

Self-contained: it builds its own minimal corpus in a temp directory rather than needing a
clean one to exist, because a test that skips when its fixture is absent reads like a pass.

Usage: python test_edit_provenance.py
"""

import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import anny_render_schema as S


def write(root, name, rows):
    """One relation to parquet, in the declared schema, zstd per the archive rule."""
    schema = S.RELATIONS[name]
    cols = {f.name: [r[f.name] for r in rows] for f in schema}
    pq.write_table(pa.table(cols, schema=schema),
                   os.path.join(root, f"{name}.parquet"), compression="zstd")


def clean_corpus(root):
    """The smallest corpus that exercises the generated half: one constructed frame,
    one bf16 edit of it, and every row the foreign keys need to resolve."""
    write(root, "identities", [{"identity_id": 1, "seed": 7, "source_subject": "s01",
                                "split": "train"}])
    write(root, "licenses", [{"license_id": 1, "name": "CC-BY-4.0", "url": "https://x"}])
    write(root, "source_datasets", [{"dataset_id": 1, "name": "100STYLE", "license_id": 1}])
    write(root, "motion_clips", [{"clip_id": 1, "dataset_id": 1, "clip_name": "Aeroplane_BR",
                                  "fps": 60.0}])
    write(root, "poses", [{"pose_id": 1, "clip_id": 1, "frame_index": 0}])
    write(root, "environments", [{"env_id": 1, "name": "studio", "hdri_asset": "a.exr",
                                  "license_id": 1}])
    write(root, "scenes", [{"scene_id": 1, "identity_id": 1, "pose_id": 1, "env_id": 1,
                            "seed": 7, "split": "train"}])
    write(root, "cameras", [{"camera_id": 1, "scene_id": 1, "view_index": 0,
                             "fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5,
                             "width": 1024, "height": 1024,
                             "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                             "tx": 0.0, "ty": 0.0, "tz": 2.0}])
    write(root, "render_runs", [{"run_id": 1, "anny_version": "0.0.0",
                                 "renderer": "mitsuba-3.9.1", "git_sha": "0" * 40,
                                 "started_utc": "2026-08-23T00:00:00Z"}])
    write(root, "renders", [{"render_id": 100, "camera_id": 1, "run_id": 1}])
    write(root, "render_data", [{"render_id": 100, "image": b"png"}])

    write(root, "edit_models", [{"edit_model_id": 1, "repo_id": "OmniGen2/OmniGen2",
                                 "revision": "b" * 40}])
    write(root, "edit_prompts", [{"prompt_id": 1, "text": "Turn this into a photograph."},
                                 {"prompt_id": 2, "text": "(((deformed))), blurry"}])
    write(root, "edit_runs", [{"edit_run_id": 1, "edit_model_id": 1, "prompt_id": 1,
                               "negative_prompt_id": 2, "precision": "bf16", "steps": 50,
                               "text_guidance_scale": 5.0, "image_guidance_scale": 2.0,
                               "started_utc": "2026-08-23T00:00:00Z"}])
    write(root, "edited_renders", [{"edit_id": 900, "render_id": 100, "edit_run_id": 1,
                                    "seed": 0, "image": b"png"}])


# ---- corruptions: each targets ONE check -----------------------------------

def c_quantised(root):
    """Condition 5. The edit is well-formed and fully provenanced; it was made at four
    bits, which is device-sizing evidence rather than corpus data."""
    write(root, "edit_runs", [{"edit_run_id": 1, "edit_model_id": 1, "prompt_id": 1,
                               "negative_prompt_id": 2, "precision": "nf4", "steps": 50,
                               "text_guidance_scale": 5.0, "image_guidance_scale": 2.0,
                               "started_utc": "2026-08-23T00:00:00Z"}])


def c_pool_merge(root):
    """Condition 2. The edited frame is also written into the constructed pool, which is
    the convenience write that makes one loader read one table."""
    write(root, "render_data", [{"render_id": 100, "image": b"png"},
                                {"render_id": 900, "image": b"edited"}])


def c_no_edit_runs(root):
    """Condition 1, as an unmet precondition. The pixels survive and the record naming
    what produced them does not."""
    os.remove(os.path.join(root, "edit_runs.parquet"))


def c_dangling_run(root):
    """The run row exists for a different id, so precision is unknown. Distinct from the
    case above: nothing is missing at the relation level, so a check keyed on presence
    alone passes here."""
    write(root, "edited_renders", [{"edit_id": 900, "render_id": 100, "edit_run_id": 42,
                                    "seed": 0, "image": b"png"}])


def c_orphan_source(root):
    """The constructed frame the edit claims to come from does not exist, so the join
    back to true labels is broken."""
    write(root, "edited_renders", [{"edit_id": 900, "render_id": 999, "edit_run_id": 1,
                                    "seed": 0, "image": b"png"}])


def c_unprovenanced_model(root):
    """The model row the run names is gone, so the checkpoint cannot be answered later."""
    os.remove(os.path.join(root, "edit_models.parquet"))


CORRUPTIONS = [
    ("quantised run", c_quantised, "NOT CORPUS DATA"),
    ("pool merge", c_pool_merge, "POOL MERGE"),
    ("edit_runs deleted", c_no_edit_runs, "no edit_runs relation"),
    ("dangling edit_run", c_dangling_run, "precision is unknown"),
    ("orphan source render", c_orphan_source, "absent from renders.render_id"),
    ("model row deleted", c_unprovenanced_model, "missing relation: edit_models"),
]


def main():
    root = tempfile.mkdtemp(prefix="edit-prov-")
    failures = 0
    try:
        clean_corpus(root)

        # GREEN first. A red suite that passes against a fixture the checks reject for
        # some unrelated reason proves nothing about the checks.
        problems = S.validate(root)
        green = [p for p in problems if not p.startswith("missing relation:")]
        print(f"GREEN  clean corpus -> {len(green)} problems")
        for p in green:
            print(f"         {p}")
        if green:
            failures += 1

        for label, corrupt, expect in CORRUPTIONS:
            work = tempfile.mkdtemp(prefix="edit-prov-red-")
            try:
                shutil.rmtree(work)
                shutil.copytree(root, work)
                corrupt(work)
                got = S.validate(work)
                hit = [p for p in got if expect in p]
                verdict = "RED  ok  " if hit else "RED  MISS"
                print(f"{verdict} {label:22s} -> {hit[0] if hit else 'check did not fire'}")
                if not hit:
                    failures += 1
            finally:
                shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{len(CORRUPTIONS)} corruptions, {failures} unproven")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
