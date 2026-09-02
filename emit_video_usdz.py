"""Emit one .usdz per (edit, candidate) with animated blendshapes + audio + transcript.

Each Video-stub artifact carries in a single self-contained USDZ package:

  SkelRoot
   |- Mesh (rest ANNY body, 18,056 verts)
   |- Skeleton (78 SOMA joints at rest -- edits do not move bones)
   |- SkelAnimation (blendShapeWeights time samples, 224 frames)
   |- BlendShape x52 (per-FACS-action sparse vertex offsets)
   `- SpatialAudio (references SpeakingFaces WAV embedded in the usdz)
  Custom metadata on the root: {webvtt_path: 'transcript.vtt'} + provenance dict

Time samples are stored in `blendShapeWeights` as one array per frame; only the
edit's target actions ramp from 0 -> endpoint over the frame range, others stay 0.
CandidateS differ in their endpoint values per MASKSCORE.md's 5-rank spec.

USDZ per RFD 1173 / CLAUDE.md's OpenUSD rule; usdz is the container the zip ban
exempts.

Usage:
    pixi run --environment usd python emit_video_usdz.py \
        [--edit-dir build/edit] [--blendshapes build/blendshapes.npz] \
        [--out-dir build/edit/video] [--n-frames 224] [--fps 28] \
        [--audio-src /Users/ernest.lee/DatasetsAllowlist/SpeakingFaces-rgb-audio/sub_100_ia/trial_1/mic1_audio_cmd_trim] \
        [--wav-basename 100_1_2_1_97_1.wav]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import wave

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdMedia, UsdSkel, Vt


HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = ["rank1", "rank2", "rank3", "rank4", "rank5"]


def wav_duration(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def build_stage(work_dir: pathlib.Path, blendshapes_npz: pathlib.Path,
                weights_per_frame: np.ndarray, endpoint_actions: dict,
                fps: int, audio_path: pathlib.Path, audio_duration: float,
                webvtt_name: str, provenance: dict, out_usda: pathlib.Path) -> None:
    """Write a .usda referencing the audio (copied into work_dir alongside)."""
    bs = np.load(blendshapes_npz)
    rest_verts = bs["rest_verts"]
    faces = bs["faces"]
    labels = list(bs["labels"])
    deltas = bs["deltas"]              # (52, V, 3) float32

    stage = Usd.Stage.CreateNew(str(out_usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(weights_per_frame.shape[0] - 1)

    root = UsdSkel.Root.Define(stage, "/anny_video")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/anny_video/body")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(rest_verts.astype(np.float32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * faces.shape[0]))

    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())

    skel = UsdSkel.Skeleton.Define(stage, "/anny_video/skeleton")
    joint_names = ["root"]     # bones stay at rest for facial-action edits
    skel.CreateJointsAttr(joint_names)
    skel.CreateBindTransformsAttr([Gf.Matrix4d(1.0)])
    skel.CreateRestTransformsAttr([Gf.Matrix4d(1.0)])
    binding.CreateSkeletonRel().SetTargets([skel.GetPrim().GetPath()])

    anim = UsdSkel.Animation.Define(stage, "/anny_video/animation")
    anim.CreateBlendShapesAttr(labels)
    anim.CreateJointsAttr([])            # no joint animation
    weights_attr = anim.CreateBlendShapeWeightsAttr()
    for t in range(weights_per_frame.shape[0]):
        weights_attr.Set(Vt.FloatArray.FromNumpy(weights_per_frame[t].astype(np.float32)),
                         Usd.TimeCode(t))
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])

    binding.CreateBlendShapesAttr(labels)
    target_paths = []
    for i, label in enumerate(labels):
        bs_prim = UsdSkel.BlendShape.Define(stage, f"/anny_video/body/{_sanitize(label)}")
        # SPARSE: only carry the vertices that actually moved beyond eps.
        d = deltas[i]
        moved = np.where(np.linalg.norm(d, axis=1) > 1e-6)[0].astype(np.int32)
        offsets = d[moved].astype(np.float32)
        bs_prim.CreateOffsetsAttr(Vt.Vec3fArray.FromNumpy(offsets))
        bs_prim.CreatePointIndicesAttr(Vt.IntArray.FromNumpy(moved))
        target_paths.append(bs_prim.GetPrim().GetPath())
    binding.CreateBlendShapeTargetsRel().SetTargets(target_paths)

    audio = UsdMedia.SpatialAudio.Define(stage, "/anny_video/audio")
    audio.CreateFilePathAttr(Sdf.AssetPath(audio_path.name))
    audio.CreateStartTimeAttr(0.0)
    audio.CreateEndTimeAttr(audio_duration)

    # Provenance + transcript reference land on the root as customData. USD passes
    # arbitrary dicts through, so the WebVTT path stays a first-class artifact of the
    # usdz package rather than a stray sidecar.
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("provenance", provenance)
    root_prim.SetCustomDataByKey("webvtt_asset", Sdf.AssetPath(webvtt_name))
    root_prim.SetCustomDataByKey("endpoint_actions", {k: float(v) for k, v in endpoint_actions.items()})

    stage.GetRootLayer().Save()


def _sanitize(name: str) -> str:
    """USD path components are [A-Za-z0-9_], make labels safe."""
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def weights_over_time(labels: list, endpoint_actions: dict, n_frames: int) -> np.ndarray:
    """Linear ramp per action from rest (0) to endpoint over n_frames."""
    w = np.zeros((n_frames, len(labels)), dtype=np.float32)
    for k, v in endpoint_actions.items():
        idx = labels.index(k)
        w[:, idx] = np.linspace(0.0, float(v), n_frames, dtype=np.float32)
    return w


def emit_usdz(work_dir: pathlib.Path, out_usdz: pathlib.Path) -> None:
    """Package the work_dir's files into an unpacked usdz per the UsdUtils API."""
    from pxr import UsdUtils
    usda_path = work_dir / "root.usda"
    if not usda_path.exists():
        raise SystemExit(f"missing {usda_path}")
    UsdUtils.CreateNewUsdzPackage(str(usda_path), str(out_usdz))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-dir",   type=pathlib.Path, default=HERE / "build" / "edit")
    ap.add_argument("--blendshapes", type=pathlib.Path, default=HERE / "build" / "blendshapes.npz")
    ap.add_argument("--out-dir",    type=pathlib.Path, default=HERE / "build" / "edit" / "video")
    ap.add_argument("--n-frames",   type=int, default=224)
    ap.add_argument("--fps",        type=int, default=28)
    ap.add_argument("--audio-src",  type=pathlib.Path,
                    default=pathlib.Path("/Users/ernest.lee/DatasetsAllowlist/SpeakingFaces-rgb-audio/"
                                         "sub_100_ia/trial_1/mic1_audio_cmd_trim"))
    ap.add_argument("--wav-basename", default="100_1_2_1_97_1.wav")
    ap.add_argument("--transcript",   type=pathlib.Path,
                    default=HERE / "build" / "transcripts" / "100_1_2_1_97_1.vtt",
                    help="webvtt sidecar to embed by reference (created by later ASR step)")
    a = ap.parse_args(argv[1:])

    manifest = json.loads((a.edit_dir / "manifest.json").read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = a.audio_src / a.wav_basename
    if not audio_path.is_file():
        raise SystemExit(f"missing audio: {audio_path}")
    audio_dur = wav_duration(audio_path)

    if not a.transcript.is_file():
        # Placeholder: emit a stub VTT if ASR hasn't produced one yet, so the
        # usdz reference resolves. Real content lands when transcribe_asr.py runs.
        a.transcript.parent.mkdir(parents=True, exist_ok=True)
        a.transcript.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:%02d.000\n(transcript pending)\n"
                                 % int(audio_dur))

    bs = np.load(a.blendshapes)
    labels = list(bs["labels"])
    swap_attr = manifest["phenotype_swap_attribute"]
    swap_val  = manifest["phenotype_swap_value"]

    total = 0
    for edit_name, edit_spec in manifest["edits"].items():
        target = edit_spec["target_actions"]
        wrong  = edit_spec["wrong_actions"]
        edit_out = a.out_dir / edit_name
        edit_out.mkdir(parents=True, exist_ok=True)
        for cand in CANDIDATES:
            if cand == "rank1":
                endpoint = target
            elif cand == "rank2":
                endpoint = {k: v * 0.5 for k, v in target.items()}
            elif cand == "rank3":
                endpoint = {k: v * 0.3 for k, v in target.items()}
            elif cand == "rank4":
                endpoint = wrong
            elif cand == "rank5":
                # Phenotype swap doesn't animate here (bones stay); the animation is
                # the same as rank1; the swap is recorded in customData/provenance so
                # a consumer that re-renders with the phenotype override sees it.
                endpoint = target

            w = weights_over_time(labels, endpoint, a.n_frames)

            work = pathlib.Path(tempfile.mkdtemp(prefix=f"vid_{edit_name}_{cand}_"))
            try:
                shutil.copy(audio_path, work / audio_path.name)
                shutil.copy(a.transcript, work / a.transcript.name)
                provenance = {
                    "edit": edit_name, "candidate": cand, "n_frames": a.n_frames,
                    "fps": a.fps, "endpoint_actions": {k: float(v) for k, v in endpoint.items()},
                    "endpoint_phenotype_nondefault":
                        {swap_attr: float(swap_val)} if cand == "rank5" else {},
                    "audio_source": str(audio_path.relative_to(pathlib.Path.home())),
                    "blendshape_labels_count": len(labels),
                }
                build_stage(work_dir=work, blendshapes_npz=a.blendshapes,
                            weights_per_frame=w, endpoint_actions=endpoint,
                            fps=a.fps, audio_path=audio_path,
                            audio_duration=audio_dur,
                            webvtt_name=a.transcript.name,
                            provenance=provenance,
                            out_usda=work / "root.usda")
                emit_usdz(work, edit_out / f"{cand}.usdz")
                total += 1
                print(f"  {edit_name}/{cand}.usdz", flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)

    print(f"done. {total} usdz files under {a.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
