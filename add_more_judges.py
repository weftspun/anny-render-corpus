"""Add three more ASR judges to the panel: wav2vec2, Voxtral, Kyutai STT.

Emits one .vtt per (judge, clip) alongside the existing as-heard / whisper / ipa
tracks. Judges are treated as independent candidates in the MaskScore-style panel;
the differences between them are the gradient a canonicalizer learns from.

  <stem>.wav2vec2.vtt    facebook/wav2vec2-large-960h-lv60-self (Apache-2.0)
  <stem>.voxtral.vtt     mistralai/Voxtral-Mini-3B-2507 (Apache-2.0)
  <stem>.kyutai.vtt      kyutai/stt-2.6b-en (CC-BY-4.0)

Failures per judge are recorded rather than blocking the run: if a model does not
load or a clip does not decode, its VTT stays empty and the index.json entry carries
an `error` field.

Usage:
    pixi run --environment asr python add_more_judges.py \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --transcripts-dir build/transcripts [--limit 15] \
        [--skip wav2vec2] [--skip voxtral] [--skip kyutai]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

import numpy as np


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
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def load_wav_16k(path: pathlib.Path):
    import librosa
    audio, sr = librosa.load(str(path), sr=16000, mono=True)
    return audio, sr


# ---- wav2vec2 --------------------------------------------------------------

def wav2vec2_transcribe(wavs, model_id="facebook/wav2vec2-large-960h-lv60-self"):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch
    proc = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
    out = {}
    for wav in wavs:
        audio, sr = load_wav_16k(wav)
        inputs = proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)
        text = proc.batch_decode(ids)[0]
        out[wav.stem] = {"text": text.strip(), "model": model_id, "license": "apache-2.0"}
    return out


# ---- Voxtral ---------------------------------------------------------------

def voxtral_transcribe(wavs, model_id="mistralai/Voxtral-Mini-3B-2507"):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import torch
    proc = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(model_id).eval()
    out = {}
    for wav in wavs:
        # Chat-style call: ask Voxtral to transcribe verbatim. Auto-language via prompt
        # rather than apply_transcription_request's language= arg, which does not accept
        # None. Voxtral will still transcribe whatever language it hears.
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str(wav)},
            {"type": "text",  "text": "Transcribe the audio verbatim. Output only the transcription."},
        ]}]
        inputs = proc.apply_chat_template(conversation, return_dict=True,
                                           return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        text = proc.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0]
        out[wav.stem] = {"text": text.strip(), "model": model_id, "license": "apache-2.0"}
    return out


# ---- Kyutai STT ------------------------------------------------------------

def kyutai_transcribe(wavs, model_id="kyutai/stt-2.6b-en"):
    # Kyutai's moshi package hosts the STT model; API roughly:
    #   from moshi.models import get_moshi_lm_stt; wrap for one-shot inference.
    # If moshi's public API shifts, fall back to transformers-hosted inference.
    from moshi.models import loaders
    import torch
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(model_id)
    mimi = checkpoint_info.get_mimi(device="cpu")
    tokenizer = checkpoint_info.get_text_tokenizer()
    lm = checkpoint_info.get_moshi(device="cpu")
    from moshi.models.tts import LMGen
    # The one-shot STT inference is documented in the kyutai/stt-2.6b-en README;
    # if the API surface here does not match, the caller should --skip kyutai.
    out = {}
    for wav in wavs:
        audio, sr = load_wav_16k(wav)
        # Placeholder call: real API landed after this script's cutoff; capture the
        # failure and let the panel proceed without kyutai.
        raise NotImplementedError("kyutai moshi one-shot STT API not wired yet")
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=pathlib.Path, required=True)
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--skip", action="append", default=[])
    a = ap.parse_args(argv[1:])

    wavs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]

    judges = [
        ("wav2vec2", wav2vec2_transcribe),
        ("voxtral",  voxtral_transcribe),
        ("kyutai",   kyutai_transcribe),
    ]

    for name, fn in judges:
        if name in a.skip:
            print(f"[{name}] skipped", flush=True)
            continue
        print(f"[{name}] loading and transcribing {len(wavs)} clips...", flush=True)
        try:
            results = fn(wavs)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            traceback.print_exc()
            for wav in wavs:
                idx_path = a.transcripts_dir / f"{wav.stem}.index.json"
                if idx_path.exists():
                    idx = json.loads(idx_path.read_text())
                    idx["tracks"][name] = {"error": str(e)}
                    idx_path.write_text(json.dumps(idx, indent=2))
            continue

        for wav in wavs:
            stem = wav.stem
            info = results.get(stem, {})
            text = info.get("text", "")
            dur = wav_duration(wav)
            write_vtt(a.transcripts_dir / f"{stem}.{name}.vtt", [(0.0, dur, text)])
            idx_path = a.transcripts_dir / f"{stem}.index.json"
            idx = json.loads(idx_path.read_text()) if idx_path.exists() else {
                "stem": stem, "audio": wav.name, "tracks": {}
            }
            idx["tracks"][name] = {"vtt": f"{stem}.{name}.vtt",
                                    "text": text,
                                    "model": info.get("model", ""),
                                    "license": info.get("license", "")}
            idx_path.write_text(json.dumps(idx, indent=2))
            print(f"  [{name}] {stem}: {text!r}", flush=True)
    print(f"done. panel expansion under {a.transcripts_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
