"""Add Whisper auto-detect + Gemma-GBNF IPA tracks alongside Parakeet's as-heard track.

For each WAV that already has a `<stem>.json` + `<stem>.vtt` from transcribe_asr.py:

  <stem>.as-heard.vtt  Parakeet (auto-detect, may render kk-accented en as ru-Cyrillic)
  <stem>.whisper.vtt   Whisper large-v3 auto-detect, per-word timing preserved
  <stem>.ipa.vtt       Gemma-4-12B + GBNF constrained decode -> direct audio-to-IPA;
                       one cue for the whole clip (no per-word timing from Gemma)
  <stem>.index.json    ties the three tracks together

Design notes captured elsewhere but repeated here for reproducibility:
  * text-then-g2p via epitran was dropped because epitran's flite backend for English
    is not installed in this env and pulling GPL flite would violate the license rule.
  * Gemma constrained to IPA characters is direct audio->IPA; its ASR accuracy is
    worse than Parakeet/Whisper's, but the corpus wants tracks with different error
    modes, not one true answer.

Usage:
    pixi run --environment asr python add_multitrack_transcripts.py \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --transcripts-dir build/transcripts \
        --gemma-cli /path/to/llama-mtmd-cli \
        --gemma-model ~/Models/gemma-4-12B-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf \
        --gemma-mmproj ~/Models/gemma-4-12B-qat-gguf/mmproj-gemma-4-12b-it-qat-q4_0.gguf \
        --gemma-grammar build/ipa.gbnf [--limit 15]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


IPA_PROMPT = "Transcribe this audio into IPA phonetic notation only."


def sec_to_vtt(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_vtt(path: pathlib.Path, cues: list[tuple[float, float, str]]) -> None:
    lines = ["WEBVTT", ""]
    for start, end, text in cues:
        if not text:
            continue
        lines.append(f"{sec_to_vtt(start)} --> {sec_to_vtt(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines))


def whisper_transcribe(wav_paths: list[pathlib.Path], model_name: str) -> list[dict]:
    import whisper
    model = whisper.load_model(model_name)
    return [model.transcribe(str(w), word_timestamps=True, fp16=False, verbose=False)
            for w in wav_paths]


def wav_duration(path: pathlib.Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def gemma_ipa(wav: pathlib.Path, cli: pathlib.Path, model: pathlib.Path,
              mmproj: pathlib.Path, grammar: pathlib.Path, n_predict: int = 100,
              ) -> str:
    """Run llama-mtmd-cli with GBNF-constrained IPA output. Returns raw stdout text."""
    proc = subprocess.run(
        [str(cli), "-m", str(model), "--mmproj", str(mmproj), "--jinja",
         "--grammar-file", str(grammar), "--audio", str(wav), "-p", IPA_PROMPT,
         "--temp", "0.0", "--seed", "0", "-n", str(n_predict), "--no-warmup"],
        capture_output=True, text=True, timeout=300,
    )
    # llama.cpp writes progress to stderr; the constrained-decoded IPA lands on stdout.
    text = proc.stdout.strip()
    # Filter to the trailing IPA-looking content: strip common prefixes.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=pathlib.Path, required=True)
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--gemma-cli", type=pathlib.Path,
                    default=pathlib.Path("/Users/ernest.lee/Desktop/weftspun/"
                                          "3-interactor/llama-cpp-npu-vision-upstream/"
                                          "build/bin/llama-mtmd-cli"))
    ap.add_argument("--gemma-model", type=pathlib.Path,
                    default=pathlib.Path.home() / "Models/gemma-4-12B-qat-gguf/"
                                                  "gemma-4-12b-it-qat-q4_0.gguf")
    ap.add_argument("--gemma-mmproj", type=pathlib.Path,
                    default=pathlib.Path.home() / "Models/gemma-4-12B-qat-gguf/"
                                                  "mmproj-gemma-4-12b-it-qat-q4_0.gguf")
    ap.add_argument("--gemma-grammar", type=pathlib.Path,
                    default=pathlib.Path("build/ipa.gbnf"))
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args(argv[1:])

    wavs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]

    for wav in wavs:
        old = a.transcripts_dir / (wav.stem + ".vtt")
        new = a.transcripts_dir / (wav.stem + ".as-heard.vtt")
        if old.exists() and not new.exists():
            old.rename(new)

    print(f"loading whisper {a.whisper_model}...", flush=True)
    whisper_results = whisper_transcribe(wavs, a.whisper_model)

    index = []
    for wav, w in zip(wavs, whisper_results):
        stem = wav.stem
        text = str(w.get("text", "")).strip()
        detected_lang = str(w.get("language", "unknown"))
        words = []
        for seg in w.get("segments", []):
            for word in seg.get("words", []) or []:
                words.append({"word": str(word.get("word", "")).strip(),
                              "start": float(word.get("start", 0.0)),
                              "end":   float(word.get("end", 0.0))})
        write_vtt(a.transcripts_dir / f"{stem}.whisper.vtt",
                  [(x["start"], x["end"], x["word"]) for x in words])

        # Gemma constrained IPA: one cue for the whole clip. No per-word timing from
        # audio-only Gemma without alignment; alignment could layer in later.
        ipa_line = gemma_ipa(wav, a.gemma_cli, a.gemma_model, a.gemma_mmproj, a.gemma_grammar)
        dur = wav_duration(wav)
        write_vtt(a.transcripts_dir / f"{stem}.ipa.vtt", [(0.0, dur, ipa_line)])

        idx = {
            "stem": stem, "audio": wav.name,
            "whisper_detected_language": detected_lang,
            "tracks": {
                "as-heard": {"vtt": f"{stem}.as-heard.vtt", "json": f"{stem}.json",
                             "model": "nvidia/parakeet-tdt-0.6b-v3",
                             "license": "cc-by-4.0",
                             "notes": "Parakeet auto-detect; kk-accented en often renders as ru-Cyrillic"},
                "whisper":  {"vtt": f"{stem}.whisper.vtt", "text": text,
                             "words": len(words), "detected_language": detected_lang,
                             "model": f"openai/whisper-{a.whisper_model}",
                             "license": "apache-2.0",
                             "notes": "auto-detect language; code-switching preserved"},
                "ipa":      {"vtt": f"{stem}.ipa.vtt", "ipa": ipa_line,
                             "model": "google/gemma-4-12B-it-qat-q4_0-gguf (GBNF-constrained)",
                             "grammar_file": str(a.gemma_grammar),
                             "prompt_sha256_note": "prompt fixed: " + IPA_PROMPT,
                             "license": "apache-2.0",
                             "notes": "direct audio-to-IPA via llama-mtmd-cli; format-clean, ASR accuracy inherited from Gemma"},
            },
        }
        (a.transcripts_dir / f"{stem}.index.json").write_text(json.dumps(idx, indent=2))
        index.append(idx)
        print(f"  {stem}: lang={detected_lang}  text={text!r}  ipa={ipa_line!r}",
              flush=True)

    (a.transcripts_dir / "multitrack_index.json").write_text(json.dumps(index, indent=2))
    print(f"done. {len(index)} clips x 3 tracks under {a.transcripts_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
