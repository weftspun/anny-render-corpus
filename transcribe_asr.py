"""Transcribe SpeakingFaces WAVs with Parakeet TDT 0.6B v3 -> per-word WebVTT + JSON.

MASKSCORE.md's Speech stub carries audio + transcript. Parakeet TDT 0.6B v3 delivers
6.32% WER (vs Whisper large-v3's 7.44%) and native word-level timestamps, both under
CC-BY-4.0 (attribution recorded in the JSON sidecar).

For each WAV we emit two files next to the source path:
  <basename>.json  transcript, model provenance, word timings
  <basename>.vtt   WebVTT with one cue per word, precise start/end from Parakeet

Usage:
    pixi run --environment asr python transcribe_asr.py \
        --audio-dir /Users/ernest.lee/DatasetsAllowlist/SpeakingFaces-rgb-audio/sub_100_ia/trial_1/mic1_audio_cmd_trim \
        --out-dir build/transcripts [--limit 15]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


MODEL = "nvidia/parakeet-tdt-0.6b-v3"
LICENSE = "cc-by-4.0"
CITATION = "NVIDIA NeMo Parakeet TDT 0.6B v3"


def sec_to_vtt(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def word_cues_to_vtt(words: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for w in words:
        start = float(w.get("start", w.get("start_offset", 0.0)))
        end   = float(w.get("end",   w.get("end_offset",   start)))
        text  = str(w.get("word", "")).strip()
        if not text:
            continue
        lines.append(f"{sec_to_vtt(start)} --> {sec_to_vtt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir",   type=pathlib.Path, required=True)
    ap.add_argument("--limit",     type=int, default=15)
    a = ap.parse_args(argv[1:])

    a.out_dir.mkdir(parents=True, exist_ok=True)

    import nemo.collections.asr as nemo_asr
    print(f"loading {MODEL}...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
    print(f"loaded {type(model).__name__}", flush=True)

    wavs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]
    if not wavs:
        raise SystemExit(f"no wavs under {a.audio_dir}")

    results = model.transcribe([str(w) for w in wavs], timestamps=True)
    for wav, h in zip(wavs, results):
        text = h.text
        word_ts = h.timestamp.get("word", []) if h.timestamp else []
        segments = h.timestamp.get("segment", []) if h.timestamp else []
        out = {
            "audio": wav.name,
            "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "model": MODEL, "license": LICENSE, "citation": CITATION,
            "text": text,
            "word_timings": [{"word": w.get("word", ""),
                              "start": float(w.get("start", w.get("start_offset", 0.0))),
                              "end":   float(w.get("end",   w.get("end_offset",   0.0)))}
                             for w in word_ts],
            "segments": [{"segment": s.get("segment", ""),
                          "start": float(s.get("start", s.get("start_offset", 0.0))),
                          "end":   float(s.get("end",   s.get("end_offset",   0.0)))}
                         for s in segments],
        }
        (a.out_dir / (wav.stem + ".json")).write_text(json.dumps(out, indent=2))
        (a.out_dir / (wav.stem + ".vtt")).write_text(word_cues_to_vtt(word_ts))
        print(f"  {wav.name} -> {text!r}  ({len(word_ts)} words)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
