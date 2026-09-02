"""Emit the full 10-track ASR panel: 5 text judges + 5 IPA judges per audio clip.

Panel design (user's ask: "add more judges for more gradients"):

  Text tracks (auto-language, no forced English):
    1. parakeet   nvidia/parakeet-tdt-0.6b-v3   CC-BY-4.0
    2. whisper    openai/whisper-large-v3       Apache-2.0
    3. voxtral    mistralai/Voxtral-Mini-3B-2507 Apache-2.0
    4. wav2vec2   facebook/wav2vec2-large-960h-lv60-self Apache-2.0
    5. gemma-auto google/gemma-4-12B-it-qat-q4_0-gguf Apache-2.0 (no grammar)

  IPA tracks (International Phonetic Alphabet):
    6.  gemma-ipa      Gemma-4-12B + GBNF character-class constraint
    7.  voxtral-ipa    Voxtral chat-prompted "output IPA only"
    8.  ipa-whisper-s  neurlang/ipa-whisper-small  Apache-2.0
    9.  ipa-whisper-b  neurlang/ipa-whisper-base   Apache-2.0
    10. allosaurus     universal phone recognizer  MIT

Each track ships as its own .vtt sidecar next to the source WAV. index.json ties them.

Judges that failed in earlier experiments and are dropped from this panel:
  * Whisper large-v3 with IPA initial_prompt (didn't shift output)
  * wav2vec2-lv-60-espeak-cv-ft (requires GPL phonemizer/espeak backend)
  * Kyutai STT (moshi API not one-shot friendly)

Usage:
    pixi run --environment asr python emit_10track_panel.py \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --transcripts-dir build/transcripts [--limit 15]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import wave


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


def wav_duration(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def load_16k(path: pathlib.Path):
    import librosa
    return librosa.load(str(path), sr=16000, mono=True)


# ---- text judges -----------------------------------------------------------

def parakeet_text(wavs):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    results = m.transcribe([str(w) for w in wavs], timestamps=True)
    return {w.stem: (h.text.strip(), h.timestamp.get("word", []) if h.timestamp else [])
            for w, h in zip(wavs, results)}


def whisper_text(wavs, model_name="large-v3"):
    import whisper
    m = whisper.load_model(model_name)
    out = {}
    for w in wavs:
        r = m.transcribe(str(w), word_timestamps=True, fp16=False, verbose=False)
        words = [(word["start"], word["end"], word["word"].strip())
                 for seg in r.get("segments", []) for word in (seg.get("words") or [])]
        out[w.stem] = (r.get("text", "").strip(), words)
    return out


def voxtral_text(wavs, model_id="mistralai/Voxtral-Mini-3B-2507"):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import torch
    proc = AutoProcessor.from_pretrained(model_id)
    m = VoxtralForConditionalGeneration.from_pretrained(model_id).eval()
    out = {}
    for wav in wavs:
        conv = [{"role":"user","content":[
            {"type":"audio","path":str(wav)},
            {"type":"text","text":"Transcribe the audio verbatim. Output only the transcription."},
        ]}]
        inputs = proc.apply_chat_template(conv, return_dict=True, return_tensors="pt")
        with torch.no_grad():
            g = m.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = proc.batch_decode(g[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
        out[wav.stem] = (text, [])
    return out


def wav2vec2_text(wavs, model_id="facebook/wav2vec2-large-960h-lv60-self"):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch
    p = Wav2Vec2Processor.from_pretrained(model_id)
    m = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
    out = {}
    for wav in wavs:
        a, sr = load_16k(wav)
        ins = p(a, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            l = m(ins.input_values).logits
        i = torch.argmax(l, dim=-1)
        text = p.batch_decode(i)[0].strip()
        out[wav.stem] = (text, [])
    return out


def gemma_cli_run(wav, prompt, cli, model, mmproj, grammar_file=None, n=200):
    args = [str(cli), "-m", str(model), "--mmproj", str(mmproj), "--jinja",
            "--audio", str(wav), "-p", prompt, "--temp", "0.0",
            "--seed", "0", "-n", str(n), "--no-warmup"]
    if grammar_file:
        args += ["--grammar-file", str(grammar_file)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=300)
    # Grab the last non-empty line; Gemma's chain-of-thought stops before the answer
    # for constrained-grammar runs, and after a channel marker for free runs.
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]
    # For free runs, strip the "<channel|>..." prefix if present.
    if "<channel|>" in last:
        last = last.split("<channel|>")[-1]
    return last.strip()


def gemma_auto_text(wavs, cli, model, mmproj):
    out = {}
    for wav in wavs:
        t = gemma_cli_run(wav, "Transcribe this audio verbatim. Output the transcription only.",
                          cli, model, mmproj)
        out[wav.stem] = (t, [])
    return out


# ---- IPA judges ------------------------------------------------------------

def gemma_ipa(wavs, cli, model, mmproj, grammar_file):
    out = {}
    for wav in wavs:
        t = gemma_cli_run(wav, "Transcribe this audio into IPA phonetic notation only.",
                          cli, model, mmproj, grammar_file=grammar_file, n=200)
        out[wav.stem] = t
    return out


def voxtral_ipa(wavs, model_id="mistralai/Voxtral-Mini-3B-2507"):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import torch
    proc = AutoProcessor.from_pretrained(model_id)
    m = VoxtralForConditionalGeneration.from_pretrained(model_id).eval()
    out = {}
    for wav in wavs:
        conv = [{"role":"user","content":[
            {"type":"audio","path":str(wav)},
            {"type":"text","text":"Transcribe this audio into International Phonetic Alphabet (IPA) notation only. Output IPA characters, no words."},
        ]}]
        inputs = proc.apply_chat_template(conv, return_dict=True, return_tensors="pt")
        with torch.no_grad():
            g = m.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = proc.batch_decode(g[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
        out[wav.stem] = text
    return out


def ipa_whisper(wavs, model_id):
    from transformers import pipeline
    p = pipeline("automatic-speech-recognition", model=model_id)
    return {w.stem: p(str(w)).get("text", "").strip() for w in wavs}


def allosaurus_ipa(wavs):
    from allosaurus.app import read_recognizer
    r = read_recognizer()
    return {w.stem: r.recognize(str(w)) for w in wavs}


# ---- driver ----------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir",       type=pathlib.Path, required=True)
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--gemma-cli",       type=pathlib.Path,
                    default=pathlib.Path("/Users/ernest.lee/Desktop/weftspun/"
                                          "3-interactor/llama-cpp-npu-vision-upstream/"
                                          "build/bin/llama-mtmd-cli"))
    ap.add_argument("--gemma-model",     type=pathlib.Path,
                    default=pathlib.Path.home() / "Models/gemma-4-12B-qat-gguf/"
                                                  "gemma-4-12b-it-qat-q4_0.gguf")
    ap.add_argument("--gemma-mmproj",    type=pathlib.Path,
                    default=pathlib.Path.home() / "Models/gemma-4-12B-qat-gguf/"
                                                  "mmproj-gemma-4-12b-it-qat-q4_0.gguf")
    ap.add_argument("--gemma-grammar",   type=pathlib.Path,
                    default=pathlib.Path("build/ipa.gbnf"))
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--skip",  action="append", default=[])
    a = ap.parse_args(argv[1:])

    wavs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]
    a.transcripts_dir.mkdir(parents=True, exist_ok=True)

    all_judges = [
        # (short_name, kind, license, callable, args)
        ("parakeet",       "text", "cc-by-4.0", parakeet_text, ()),
        ("whisper",        "text", "apache-2.0", whisper_text, ()),
        ("voxtral",        "text", "apache-2.0", voxtral_text, ()),
        ("wav2vec2",       "text", "apache-2.0", wav2vec2_text, ()),
        ("gemma-auto",     "text", "apache-2.0", gemma_auto_text,
             (a.gemma_cli, a.gemma_model, a.gemma_mmproj)),
        ("gemma-ipa",      "ipa",  "apache-2.0", gemma_ipa,
             (a.gemma_cli, a.gemma_model, a.gemma_mmproj, a.gemma_grammar)),
        ("voxtral-ipa",    "ipa",  "apache-2.0", voxtral_ipa, ()),
        ("ipa-whisper-s",  "ipa",  "apache-2.0", ipa_whisper, ("neurlang/ipa-whisper-small",)),
        ("ipa-whisper-b",  "ipa",  "apache-2.0", ipa_whisper, ("neurlang/ipa-whisper-base",)),
        ("allosaurus",     "ipa",  "mit",        allosaurus_ipa, ()),
    ]

    index_per_clip = {w.stem: {"stem": w.stem, "audio": w.name, "tracks": {}} for w in wavs}
    for name, kind, license_, fn, args in all_judges:
        if name in a.skip:
            print(f"[{name}] skipped", flush=True)
            continue
        print(f"[{name}] ({kind}) transcribing {len(wavs)} clips...", flush=True)
        try:
            results = fn(wavs, *args)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            for stem in index_per_clip:
                index_per_clip[stem]["tracks"][name] = {"error": str(e), "kind": kind}
            continue
        for wav in wavs:
            stem = wav.stem
            if isinstance(results.get(stem), tuple):
                text, word_ts = results[stem]
            else:
                text = results.get(stem, "")
                word_ts = []
            dur = wav_duration(wav)
            if word_ts:
                cues = [(float(w.get("start", w.get("start_offset", 0.0))),
                         float(w.get("end",   w.get("end_offset",   0.0))),
                         str(w.get("word", "")).strip())
                        if isinstance(w, dict)
                        else (float(w[0]), float(w[1]), str(w[2]).strip())
                        for w in word_ts]
            else:
                cues = [(0.0, dur, text)]
            vtt_path = a.transcripts_dir / f"{stem}.{name}.vtt"
            write_vtt(vtt_path, cues)
            index_per_clip[stem]["tracks"][name] = {
                "kind": kind, "license": license_, "vtt": vtt_path.name, "text": text,
            }
            print(f"  [{name}] {stem}: {text[:80]!r}", flush=True)

    for stem, idx in index_per_clip.items():
        (a.transcripts_dir / f"{stem}.index.json").write_text(json.dumps(idx, indent=2))
    (a.transcripts_dir / "multitrack_index.json").write_text(
        json.dumps(list(index_per_clip.values()), indent=2))
    print(f"done. {len(index_per_clip)} clips x {len(all_judges)} tracks under {a.transcripts_dir}/",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
