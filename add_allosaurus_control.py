"""Add two more allosaurus tracks -- eng-restricted and rus-restricted -- as controls
for the universal allosaurus track that ships with the 10-track panel.

The control tests whether allosaurus's universal output on Kazakh L2 English is
capturing L1 (Russian/Kazakh) transfer or hallucinating from its 4,282-language
inventory. Kazakh is not directly in allosaurus's phone-file set; Russian is the
closest neighbour Kazakh speakers acquire English through, so rus stands in.

Emits:
    <stem>.allosaurus-eng.vtt   allosaurus lang_id='eng'
    <stem>.allosaurus-rus.vtt   allosaurus lang_id='rus'
Updates <stem>.index.json with the two new track entries.

If universal ~= rus and both diverge from eng, the model is capturing L1 transfer.
If universal ~= eng, the phone inventory is doing most of the work.

Usage:
    pixi run --environment asr python add_allosaurus_control.py \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --transcripts-dir build/transcripts [--limit 15]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import wave


def sec_to_vtt(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def wav_duration(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def write_vtt(path: pathlib.Path, cues: list[tuple[float, float, str]]) -> None:
    lines = ["WEBVTT", ""]
    for start, end, text in cues:
        if not text:
            continue
        lines.append(f"{sec_to_vtt(start)} --> {sec_to_vtt(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=pathlib.Path, required=True)
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args(argv[1:])

    wavs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]

    from allosaurus.app import read_recognizer
    r = read_recognizer()

    for lang, tag in [("eng", "allosaurus-eng"), ("rus", "allosaurus-rus")]:
        print(f"[{tag}] transcribing {len(wavs)} clips...", flush=True)
        for wav in wavs:
            stem = wav.stem
            ipa = r.recognize(str(wav), lang_id=lang)
            dur = wav_duration(wav)
            write_vtt(a.transcripts_dir / f"{stem}.{tag}.vtt",
                      [(0.0, dur, ipa)])
            idx_path = a.transcripts_dir / f"{stem}.index.json"
            idx = json.loads(idx_path.read_text()) if idx_path.exists() else {
                "stem": stem, "audio": wav.name, "tracks": {}
            }
            idx["tracks"][tag] = {
                "kind": "ipa", "license": "mit",
                "vtt": f"{stem}.{tag}.vtt", "text": ipa,
                "lang_id": lang,
                "notes": ("allosaurus restricted to " + lang +
                          " phone inventory; pair with universal to test L1 transfer"),
            }
            idx_path.write_text(json.dumps(idx, indent=2))
            print(f"  [{tag}] {stem}: {ipa[:80]!r}", flush=True)

    # Refresh multitrack_index.json to include the new tracks.
    all_idx = []
    for wav in wavs:
        p = a.transcripts_dir / f"{wav.stem}.index.json"
        if p.exists():
            all_idx.append(json.loads(p.read_text()))
    (a.transcripts_dir / "multitrack_index.json").write_text(json.dumps(all_idx, indent=2))
    print(f"done. 12-track panel (10 base + 2 allosaurus controls) under {a.transcripts_dir}/",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
