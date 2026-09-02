"""urn:oid:1.3.6.1.4.1.66606.1.2.2164.4 -- emit MaskScore Speech-stub parquets.

Three ZSTD parquets in ETNF form per RFD 1173.2164:

  maskscore_speech.parquet             root row per SpeakingFaces clip (15 rows).
  maskscore_speech_candidates.parquet  20 candidates per clip (10 audio + 10
                                       transcript) = 300 rows.
  maskscore_speech_scores.parquet      wavlm_cos + wer per audio candidate; edit-
                                       distance to canonical per transcript
                                       candidate. Long-form metric_name/value.

Audio candidates come from voice_clone_10rank.py (RFD 2164.2.1/.2.2). Transcript
candidates come from the 12-track ASR panel (RFD 2164.1) reduced to 10 by
dropping the two allosaurus language-control tracks and ordering the rest by
canonical closeness: voxtral -> whisper -> parakeet -> gemma-auto -> wav2vec2 ->
ipa-whisper-s -> ipa-whisper-b -> voxtral-ipa -> gemma-ipa -> allosaurus.

Scores read from build/voice_clones/scores.json (RFD 2164.3.2 rerun).

Usage:
    pixi run --environment anny-mac python emit_speech_stub.py \
        --transcripts-dir build/transcripts \
        --clones-dir build/voice_clones \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --out-dir .
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq


HERE = pathlib.Path(__file__).resolve().parent

TRANSCRIPT_RANK_ORDER = [
    "voxtral", "whisper", "parakeet", "gemma-auto", "wav2vec2",
    "ipa-whisper-s", "ipa-whisper-b", "voxtral-ipa", "gemma-ipa", "allosaurus",
]

AUDIO_RANKS = list(range(1, 11))

# Rank scheme baked in as convention rather than looked up from voice_clones/manifest.json
# because RFD 2164.2.2 partial re-clone overwrote the manifest with only rank9/10 entries.
# The wavs for ranks 1-10 do exist on disk; this dict reconstructs their intent.
AUDIO_KIND_BY_RANK = {
    1:  "identity",         2: "pitch_up_mild",     3: "pitch_down_mild",
    4:  "tempo_fast",       5: "tempo_slow",        6: "combined_subtle",
    7:  "wrong_text_next",  8: "wrong_text_fixed",  9: "wrong_subject",
    10: "wrong_all",
}
WRONG_TEXT_FIXED = "The quick brown fox jumps over the lazy dog."


def wer(reference: str, hypothesis: str) -> float:
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[-1][-1] / max(1, len(r))


def relpath(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(HERE))
    except ValueError:
        return str(p)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--clones-dir",      type=pathlib.Path, required=True)
    ap.add_argument("--audio-dir",       type=pathlib.Path, required=True)
    ap.add_argument("--out-dir",         type=pathlib.Path, default=HERE)
    a = ap.parse_args(argv[1:])

    voice_scores = json.loads((a.clones_dir / "scores.json").read_text())
    clone_manifest = json.loads((a.clones_dir / "manifest.json").read_text())

    stems = sorted(voice_scores["clips"].keys())

    root_rows = []
    cand_rows = []
    score_rows = []

    for stem in stems:
        key = f"rung1/speech/{stem}"
        clip_scores = voice_scores["clips"][stem]
        canonical_text = clip_scores["canonical_text"]

        root_rows.append({
            "key": key,
            "task_type": "speech_edit",
            "dimension": "instruction_following",
            "input_column": "input_audio",
            "input_asset": relpath(a.audio_dir / f"{stem}.wav"),
            "input_asset_kind": "wav_16khz",
            "canonical_text": canonical_text,
        })

        # 10 audio candidates from voice_clone_10rank output. Target text derived from
        # rank number per AUDIO_KIND_BY_RANK convention; ranks 7 use the NEXT clip's
        # canonical text as their "wrong content" source, ranks 8/10 use a fixed phrase.
        stem_idx = stems.index(stem)
        next_stem = stems[(stem_idx + 1) % len(stems)]
        next_canonical = voice_scores["clips"][next_stem]["canonical_text"]
        for r in AUDIO_RANKS:
            path = a.clones_dir / stem / f"rank_{r:02d}.wav"
            kind = AUDIO_KIND_BY_RANK[r]
            if kind in ("wrong_text_next",):
                target_text = next_canonical
            elif kind in ("wrong_text_fixed", "wrong_all"):
                target_text = WRONG_TEXT_FIXED
            else:
                target_text = canonical_text
            cand_rows.append({
                "row_key": key, "candidate_axis": "audio", "rank": r,
                "candidate_asset": relpath(path),
                "candidate_asset_kind": "wav_16khz",
                "candidate_kind": kind,
                "candidate_target_text": target_text,
            })
            s = clip_scores["ranks"][str(r)]
            for m in ("wavlm_cos", "wer"):
                if m in s:
                    score_rows.append({"row_key": key, "candidate_axis": "audio",
                                        "candidate_rank": r, "metric_name": m,
                                        "metric_value": float(s[m])})

        # 10 transcript candidates from the 12-track ASR panel.
        idx = json.loads((a.transcripts_dir / f"{stem}.index.json").read_text())
        for r, track in enumerate(TRANSCRIPT_RANK_ORDER, start=1):
            t = idx["tracks"].get(track, {})
            if not t or "vtt" not in t:
                continue
            cand_text = t.get("text", "")
            cand_rows.append({
                "row_key": key, "candidate_axis": "transcript", "rank": r,
                "candidate_asset": f"build/transcripts/{t['vtt']}",
                "candidate_asset_kind": "webvtt",
                "candidate_kind": track,
                "candidate_target_text": cand_text,
            })
            score_rows.append({"row_key": key, "candidate_axis": "transcript",
                                "candidate_rank": r, "metric_name": "wer",
                                "metric_value": wer(canonical_text, cand_text)})

    root_t   = pa.Table.from_pylist(root_rows)
    cands_t  = pa.Table.from_pylist(cand_rows)
    scores_t = pa.Table.from_pylist(score_rows)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(root_t,   a.out_dir / "maskscore_speech.parquet",            compression="zstd")
    pq.write_table(cands_t,  a.out_dir / "maskscore_speech_candidates.parquet", compression="zstd")
    pq.write_table(scores_t, a.out_dir / "maskscore_speech_scores.parquet",     compression="zstd")

    print(f"root:       {root_t.num_rows} rows x {root_t.num_columns} cols", flush=True)
    print(f"candidates: {cands_t.num_rows} rows x {cands_t.num_columns} cols", flush=True)
    print(f"scores:     {scores_t.num_rows} rows x {scores_t.num_columns} cols", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
