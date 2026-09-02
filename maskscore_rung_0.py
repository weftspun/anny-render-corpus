"""Rung 0 of the MaskScore extraction ladder: one trial, eight rows.

Rung 0's purpose is to prove the pipeline runs end-to-end: one input asset in,
eight stub rows out — one per modality (text, mesh, speech, multimodal,
keypoints, depth, video, pose). Each row conforms to the schema RFD 1173's
`MASKSCORE.md` defines. Scoring is via render-and-compare through the ANNY
canonical mesh via Mitsuba 3 on `sphere_hammersley_sequence` views (L1 on
depth/normals, SSIM on normals, LPIPS on normals) — the one universal metric.

The input is the canonical ANNY animation fixture
(`../../3-interactor/datasource-flow-project/art/canonical_anny/anny_anim_test.usdz`,
verified 2026-09-01 by the mitsuba correspondence check). SpeakingFaces plugs
in at Rung 1 once the download + AnnyInverter + LBFGS chain is wired.

    # In the anny-render-corpus/.pixi/envs/default/ env (has mitsuba):
    #   ...python.exe maskscore_rung_0.py --fixture <path.usdz> --out rows.parquet
    #
    # The usd env (has pxr) is used for skinning bake; the default env for the
    # mitsuba render + parquet write. The two-env split is a known constraint;
    # a merged env is a follow-on when it can be built.

Rung 0 is scoped to succeed with the canonical fixture; Rung 1 adds the
SpeakingFaces derivation. The eight schemas are load-bearing here — the row
shapes must match `MASKSCORE.md` verbatim so the bench/reward-train/rl-train
datasets stay coordinate-compatible as they scale up.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path


HERE = Path(__file__).resolve().parent

TASK_TYPES = (
    "part_add", "part_remove", "part_replace",
    "expression_change", "surface_edit", "lighting_change", "skin_tone",
    "depth_edit", "region_extract", "background_change",
    "pose_change", "speech_edit", "temporal_edit", "cross_modal_compose",
)

DIMENSIONS = ("instruction_following", "consistency", "overall")


@dataclass
class Row:
    """One MaskScore row. Schema shared across the eight stubs plus per-stub extras."""

    key: str
    instruction: str
    task_type: str
    dimension: str
    scores: list[float] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"key": self.key, "instruction": self.instruction,
                "task_type": self.task_type, "dimension": self.dimension,
                "scores": self.scores, **self.extras}


def emit_text_row(key: str) -> Row:
    """TextEditReward: mouth bone activation agreement + render-and-compare on mouth."""
    return Row(
        key=key, instruction="change the command to 'hello world'",
        task_type="cross_modal_compose", dimension="instruction_following",
        extras={"input_text": "hello", "conditioning_image": "rung0_view0.png",
                "output_texts": ["hello world", "hello there", "howdy"]},
    )


def emit_mesh_row(key: str) -> Row:
    """MeshEditReward: render-and-compare via Mitsuba 3 (L1/SSIM/LPIPS on views)."""
    return Row(
        key=key, instruction="edit the mouth to match frame B",
        task_type="expression_change", dimension="instruction_following",
        extras={"input_mesh": "rung0_input.glb", "conditioning_image": "rung0_view0.png",
                "output_meshes": ["rung0_cand0.glb", "rung0_cand1.glb"]},
    )


def emit_speech_row(key: str) -> Row:
    """SpeechEditReward: SOMA face bone rotation agreement + mouth render-and-compare."""
    return Row(
        key=key, instruction="resynthesize the mouth to say 'hello world'",
        task_type="speech_edit", dimension="instruction_following",
        extras={"input_audio": "rung0_input.wav", "conditioning_image": "rung0_view0.png",
                "output_audios": ["rung0_cand0.wav"]},
    )


def emit_multimodal_row(key: str) -> Row:
    """MultimodalEditReward: pairwise render-and-compare across derivation paths."""
    return Row(
        key=key, instruction="match the mouth across modalities to frame B",
        task_type="cross_modal_compose", dimension="overall",
        extras={"input_modality": "image", "output_modality": "mesh",
                "input_data": "rung0_view0.png",
                "output_candidates": ["rung0_cand0.glb"]},
    )


def emit_keypoint_row(key: str) -> Row:
    """KeypointEditReward: displace vertices, re-render, render-and-compare."""
    return Row(
        key=key, instruction="refit the mouth landmarks to frame B",
        task_type="expression_change", dimension="instruction_following",
        extras={"input_keypoints": [], "conditioning_image": "rung0_view0.png",
                "output_keypoints": []},
    )


def emit_depth_row(key: str) -> Row:
    """DepthEditReward: render-and-compare on Mitsuba depth AOV."""
    return Row(
        key=key, instruction="reconstruct the face depth to match frame B",
        task_type="depth_edit", dimension="instruction_following",
        extras={"input_depth": "rung0_view0.exr", "conditioning_image": "rung0_view0.png",
                "output_depths": ["rung0_cand0.exr"]},
    )


def emit_video_row(key: str) -> Row:
    """VideoEditReward: per-frame render-and-compare + ANNY face activation consistency."""
    return Row(
        key=key, instruction="transition the mouth from frame A to frame B",
        task_type="temporal_edit", dimension="consistency",
        extras={"input_video": "rung0_input_frames", "aligned_audio": "rung0_input.wav",
                "output_videos": ["rung0_cand0_frames"]},
    )


def emit_pose_row(key: str) -> Row:
    """PoseEditReward: FK+skin to mesh, render-and-compare via Mitsuba 3."""
    return Row(
        key=key, instruction="repose the mouth to frame B's expression",
        task_type="pose_change", dimension="instruction_following",
        extras={"input_pose": [], "conditioning_image": "rung0_view0.png",
                "output_poses": []},
    )


STUB_EMITTERS = (
    emit_text_row, emit_mesh_row, emit_speech_row, emit_multimodal_row,
    emit_keypoint_row, emit_depth_row, emit_video_row, emit_pose_row,
)


def rung_0(fixture: Path, out: Path) -> int:
    """Emit one row per stub for the fixture. Real render-and-compare scores wire in later."""
    if not fixture.is_file():
        print(f"FAIL fixture missing: {fixture}", file=sys.stderr)
        return 1
    trial_key = f"rung0/{fixture.stem}"
    rows = [emit(trial_key) for emit in STUB_EMITTERS]
    if len(rows) != 8:
        print(f"FAIL expected 8 rows, got {len(rows)}", file=sys.stderr)
        return 1
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print(f"WARN pyarrow missing; writing jsonl fallback to {out.with_suffix('.jsonl')}", file=sys.stderr)
        import json
        with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r.to_dict()) + "\n")
        return 0
    dicts = [r.to_dict() for r in rows]
    keys = sorted({k for d in dicts for k in d.keys()})
    columns = {k: [d.get(k) for d in dicts] for k in keys}
    table = pa.table(columns)
    pq.write_table(table, out)
    print(f"ok rung 0: 8 rows -> {out}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path,
                    default=HERE.parent.parent / "3-interactor" / "datasource-flow-project" /
                            "art" / "canonical_anny" / "anny_anim_test.usdz")
    ap.add_argument("--out", type=Path, default=HERE / "maskscore_rung_0.parquet")
    a = ap.parse_args(argv)
    return rung_0(a.fixture, a.out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
