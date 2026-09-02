"""Exhaustively score voice-clone rank candidates via 3 metrics + report gradient.

Metrics:
  A. wavlm_cos    speaker embedding cosine similarity (microsoft/wavlm-base-plus-sv)
                  captures voice-only match; sensitive to pitch/timbre/speaker
  B. wer          Whisper large-v3 transcribes candidate, WER against canonical text
                  captures content match; sensitive to spoken words
  C. gemma_score  Gemma-4-12B GBNF-constrained integer 0-100 from paired audio input
                  captures perceptual judgement; slow but multi-dimensional

For each of 15 SpeakingFaces reference clips x 10 ranks = 150 candidates, compute
all three scores and write build/voice_clones/scores.json plus a summary table.

Rank1 is identity-clone -> should score near 1.0 / low WER / high gemma.
Rank10 is wrong-subject+wrong-text -> should score low across all three.

Usage:
    pixi run --environment asr python score_voice_clones.py \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --clones-dir build/voice_clones \
        --transcripts-dir build/transcripts \
        --skip gemma_score  # optional if we want to defer the slow one
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np


CANDIDATE_RANKS = list(range(1, 11))


def load_16k(path: pathlib.Path):
    import librosa
    a, sr = librosa.load(str(path), sr=16000, mono=True)
    return a.astype(np.float32)


# ---- A. wavlm cosine similarity --------------------------------------------

def wavlm_embed(wavs: list[pathlib.Path]):
    """Return (stem_to_path -> embedding vector) map from WavLM speaker head."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    import torch
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    m = AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    embs = {}
    for wav in wavs:
        a = load_16k(wav)
        i = fe(a, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = m(**i).embeddings[0].detach().cpu().numpy()
        embs[str(wav)] = e / (np.linalg.norm(e) + 1e-9)
    return embs


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---- B. whisper WER --------------------------------------------------------

def wer(reference: str, hypothesis: str) -> float:
    """Standard Word Error Rate."""
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein at word level
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


def voxtral_transcribe_batch(wavs: list[pathlib.Path],
                              model_id="mistralai/Voxtral-Mini-3B-2507"):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import torch
    p = AutoProcessor.from_pretrained(model_id)
    m = VoxtralForConditionalGeneration.from_pretrained(model_id).eval()
    out = {}
    for wav in wavs:
        conv = [{"role":"user","content":[
            {"type":"audio","path":str(wav)},
            {"type":"text","text":"Transcribe the audio verbatim. Output only the transcription."},
        ]}]
        inputs = p.apply_chat_template(conv, return_dict=True, return_tensors="pt")
        with torch.no_grad():
            g = m.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = p.batch_decode(g[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip()
        out[str(wav)] = text
    return out


# ---- C. Gemma score (optional, slow) ---------------------------------------

def gemma_score(ref: pathlib.Path, cand: pathlib.Path) -> str:
    cli = pathlib.Path("/Users/ernest.lee/Desktop/weftspun/3-interactor/"
                        "llama-cpp-npu-vision-upstream/build/bin/llama-mtmd-cli")
    model = pathlib.Path.home() / "Models/gemma-4-12B-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
    mmproj = pathlib.Path.home() / "Models/gemma-4-12B-qat-gguf/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
    prompt = ("Rate the second audio's similarity to the first (voice, content, prosody) "
              "as an integer 0-100. Reply with only the number.")
    r = subprocess.run(
        [str(cli), "-m", str(model), "--mmproj", str(mmproj), "--jinja",
         "--audio", str(ref), "--audio", str(cand),
         "-p", prompt, "--temp", "0.0", "--seed", "0", "-n", "500", "--no-warmup"],
        capture_output=True, text=True, timeout=300,
    )
    text = r.stdout
    # Gemma channel-wrapped output: "<channel|>NUMBER" or reasoning then digits.
    if "<channel|>" in text:
        tail = text.split("<channel|>")[-1].strip().split()[0].strip(".,")
    else:
        # last standalone integer in output
        import re
        matches = re.findall(r"\b(\d{1,3})\b", text)
        tail = matches[-1] if matches else ""
    try:
        return int(tail)
    except (ValueError, IndexError):
        return None


# ---- driver ----------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir",       type=pathlib.Path, required=True)
    ap.add_argument("--clones-dir",      type=pathlib.Path, required=True)
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--skip",  action="append", default=[])
    a = ap.parse_args(argv[1:])

    refs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]

    all_wavs = list(refs)
    for ref in refs:
        stem_dir = a.clones_dir / ref.stem
        all_wavs.extend(sorted(stem_dir.glob("rank_*.wav")))

    result = {"clips": {}}

    # A. WavLM cosine similarity
    if "wavlm_cos" not in a.skip:
        print(f"[A] wavlm speaker-embed for {len(all_wavs)} wavs...", flush=True)
        embs = wavlm_embed(all_wavs)
        print("[A] cosine similarity matrix computed", flush=True)

    # B. Whisper transcribe every candidate + canonical
    if "wer" not in a.skip:
        print(f"[B] voxtral transcribe {len(all_wavs)} wavs...", flush=True)
        voxtral_texts = voxtral_transcribe_batch(all_wavs)
        print("[B] transcription done", flush=True)

    for ref in refs:
        stem = ref.stem
        idx = json.loads((a.transcripts_dir / f"{stem}.index.json").read_text())
        canonical = idx["tracks"]["voxtral"]["text"]
        clip = {"canonical_text": canonical, "ranks": {}}
        for r in CANDIDATE_RANKS:
            cand = a.clones_dir / stem / f"rank_{r:02d}.wav"
            row = {}
            if "wavlm_cos" not in a.skip:
                row["wavlm_cos"] = cos(embs[str(ref)], embs[str(cand)])
            if "wer" not in a.skip:
                row["voxtral_text"] = voxtral_texts[str(cand)]
                row["wer"] = wer(canonical, voxtral_texts[str(cand)])
            if "gemma_score" not in a.skip:
                row["gemma_score"] = gemma_score(ref, cand)
            clip["ranks"][r] = row
        result["clips"][stem] = clip
        print(f"  {stem}: 10 ranks scored", flush=True)

    (a.clones_dir / "scores.json").write_text(json.dumps(result, indent=2))

    # Summary: mean per rank across clips
    print("\n=== SUMMARY: mean per rank across all clips ===", flush=True)
    header = "rank |"
    metrics = [m for m in ("wavlm_cos", "wer", "gemma_score") if m not in a.skip]
    for m in metrics:
        header += f"  {m:>12} |"
    print(header, flush=True)
    for r in CANDIDATE_RANKS:
        line = f"  {r:2d} |"
        for m in metrics:
            vals = [result["clips"][s]["ranks"][r].get(m) for s in result["clips"]]
            vals = [v for v in vals if v is not None]
            mean = statistics.mean(vals) if vals else float("nan")
            line += f"  {mean:>12.4f} |"
        print(line, flush=True)
    print(f"\ndone. scores in {a.clones_dir / 'scores.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
