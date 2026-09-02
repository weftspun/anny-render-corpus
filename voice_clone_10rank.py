"""Generate 10 voice-clone rank candidates per SpeakingFaces clip.

MaskScore 5-rank scheme expanded to 10, applied to voice cloning:
  rank1  identity clone (same voice, same text)
  rank2  pitch +2 semitones (mild timbre shift)
  rank3  pitch -2 semitones
  rank4  tempo 1.15x (fast speech)
  rank5  tempo 0.85x (slow speech)
  rank6  pitch +1 semitone + tempo 1.05x (subtle combined)
  rank7  same voice, wrong content (text from NEXT clip in list)
  rank8  same voice, entirely wrong text (fixed random phrase)
  rank9  different-subject clone (LJ Speech reference), same text
  rank10 different-subject + wrong text

Ranks 1-6 use post-hoc librosa pitch/tempo shifts on the identity clone rather than
re-cloning; qwen-tts doesn't expose direct pitch/tempo control, and post-processing
gives a reproducible severity gradient with less compute.

Each rank writes:
    build/voice_clones/<stem>/rank_<n>.wav
Provenance JSON per clip records the Qwen3-TTS model, reference audio, target text,
and the librosa pitch/tempo settings that shaped rank2-6.

Usage:
    pixi run --environment tts python voice_clone_10rank.py \
        --audio-dir /Users/.../mic1_audio_cmd_trim \
        --transcripts-dir build/transcripts \
        --lj-ref /path/to/lj_speech_sample.wav \
        [--limit 15]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WRONG_TEXT_FALLBACK = "The quick brown fox jumps over the lazy dog."
CANDIDATES = [
    # (rank, kind, severity, pitch_semitones, tempo_factor, ref_source, text_source)
    (1,  "identity",         0.0,  0.0, 1.00, "self",     "self"),
    (2,  "pitch_up_mild",    0.9, +2.0, 1.00, "self",     "self"),
    (3,  "pitch_down_mild",  0.9, -2.0, 1.00, "self",     "self"),
    (4,  "tempo_fast",       0.7,  0.0, 1.15, "self",     "self"),
    (5,  "tempo_slow",       0.7,  0.0, 0.85, "self",     "self"),
    (6,  "combined_subtle",  0.5, +1.0, 1.05, "self",     "self"),
    (7,  "wrong_text_next",  0.3,  0.0, 1.00, "self",     "next_clip_text"),
    (8,  "wrong_text_fixed", 0.3,  0.0, 1.00, "self",     "fixed_wrong"),
    (9,  "wrong_subject",    0.2,  0.0, 1.00, "lj_ref",   "self"),
    (10, "wrong_all",        0.0,  0.0, 1.00, "lj_ref",   "fixed_wrong"),
]


def load_canonical_text(transcripts_dir: pathlib.Path, stem: str) -> str:
    """Voxtral output is treated as canonical text per the 12-track panel ranking."""
    idx = json.loads((transcripts_dir / f"{stem}.index.json").read_text())
    return idx["tracks"]["voxtral"]["text"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir",       type=pathlib.Path, required=True)
    ap.add_argument("--transcripts-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir",         type=pathlib.Path,
                    default=pathlib.Path("build/voice_clones"))
    ap.add_argument("--lj-ref",          type=pathlib.Path,
                    help="single reference wav for wrong-subject ranks")
    ap.add_argument("--wrong-subject-dir", type=pathlib.Path,
                    help="dir of wavs from a DIFFERENT speaker; one is picked per clip "
                         "(rotates) as the rank9/10 reference. Overrides --lj-ref.")
    ap.add_argument("--only-ranks",       type=int, nargs="+", default=None,
                    help="only regenerate the listed ranks (default: all 10)")
    ap.add_argument("--limit",           type=int, default=15)
    a = ap.parse_args(argv[1:])

    a.out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(a.audio_dir.glob("*.wav"))[:a.limit]

    from qwen_tts import Qwen3TTSModel
    import librosa, soundfile as sf, numpy as np, torch

    print(f"loading {MODEL_ID}...", flush=True)
    model = Qwen3TTSModel.from_pretrained(MODEL_ID)
    print("model loaded", flush=True)

    wrong_subject_wavs = []
    if a.wrong_subject_dir and a.wrong_subject_dir.is_dir():
        wrong_subject_wavs = sorted(a.wrong_subject_dir.glob("*.wav"))
        print(f"wrong-subject pool: {len(wrong_subject_wavs)} wavs from {a.wrong_subject_dir.name}",
              flush=True)
    elif a.lj_ref and a.lj_ref.is_file():
        wrong_subject_wavs = [a.lj_ref]
    else:
        print(f"WARNING: no wrong-subject reference; ranks 9-10 fall back to same speaker",
              flush=True)

    manifest = {"model": MODEL_ID, "ranks_spec": {}, "clips": {}}
    for r, kind, sev, semi, tempo, ref_src, text_src in CANDIDATES:
        manifest["ranks_spec"][r] = {"kind": kind, "severity": sev, "pitch_semitones": semi,
                                      "tempo_factor": tempo, "ref_source": ref_src,
                                      "text_source": text_src}

    for i, wav in enumerate(wavs):
        stem = wav.stem
        stem_dir = a.out_dir / stem
        stem_dir.mkdir(exist_ok=True)
        own_text = load_canonical_text(a.transcripts_dir, stem)
        next_text = load_canonical_text(a.transcripts_dir, wavs[(i + 1) % len(wavs)].stem)
        clip_ranks = {}

        # Identity clone once, reuse audio for ranks 1-6 (self voice + self text)
        prompt = model.create_voice_clone_prompt(ref_audio=str(wav), ref_text=own_text)
        base_audios, base_sr = model.generate_voice_clone(text=own_text, voice_clone_prompt=prompt,
                                                            non_streaming_mode=True)
        # base_audios is a list of np.ndarray; take the first (only) one.
        base_np = np.asarray(base_audios[0], dtype=np.float32).squeeze()

        for r, kind, sev, semi, tempo, ref_src, text_src in CANDIDATES:
            if a.only_ranks is not None and r not in a.only_ranks:
                continue
            # Decide text: self, next_clip_text, or fixed
            if text_src == "self":
                target_text = own_text
            elif text_src == "next_clip_text":
                target_text = next_text
            else:
                target_text = WRONG_TEXT_FALLBACK

            # Decide reference: self clip or LJ (or fallback to a different SpeakingFaces clip)
            if ref_src == "self" and text_src == "self":
                # ranks 1-6: reuse base_audio and post-process
                y = base_np
                sr = base_sr
                if semi != 0.0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semi)
                if tempo != 1.0:
                    y = librosa.effects.time_stretch(y, rate=tempo)
            elif ref_src == "self":
                # ranks 7-8: same voice, different text -- re-clone with new target text
                ys, sr = model.generate_voice_clone(text=target_text,
                                                     voice_clone_prompt=prompt,
                                                     non_streaming_mode=True)
                y = np.asarray(ys[0], dtype=np.float32).squeeze()
            else:  # ref_src == "lj_ref"
                # ranks 9-10: DIFFERENT-SPEAKER reference from the wrong-subject pool.
                # Rotates through the pool so each of the 15 clips gets a distinct
                # wrong-speaker voice. Falls back to a different clip from the same
                # speaker if no pool provided (last resort control failure).
                if wrong_subject_wavs:
                    ref_wav = wrong_subject_wavs[i % len(wrong_subject_wavs)]
                    # x_vector_only_mode=True: skip ref_text, use speaker embedding only.
                    # We do not have transcripts for the wrong-subject pool, and requiring
                    # them just to test speaker swap adds a whole transcription pass.
                    alt_prompt = model.create_voice_clone_prompt(
                        ref_audio=str(ref_wav), x_vector_only_mode=True)
                else:
                    fallback_wav = wavs[(i + 5) % len(wavs)]
                    ref_wav = fallback_wav
                    ref_txt = load_canonical_text(a.transcripts_dir, fallback_wav.stem)
                    alt_prompt = model.create_voice_clone_prompt(
                        ref_audio=str(ref_wav), ref_text=ref_txt)
                ys, sr = model.generate_voice_clone(text=target_text,
                                                     voice_clone_prompt=alt_prompt,
                                                     non_streaming_mode=True)
                y = np.asarray(ys[0], dtype=np.float32).squeeze()

            out_wav = stem_dir / f"rank_{r:02d}.wav"
            sf.write(str(out_wav), y, int(sr))
            clip_ranks[r] = {"kind": kind, "path": out_wav.name,
                              "target_text": target_text,
                              "ref_source": ref_src, "text_source": text_src,
                              "pitch_semitones": semi, "tempo_factor": tempo}
        manifest["clips"][stem] = {"source_audio": wav.name,
                                    "canonical_text": own_text,
                                    "ranks": clip_ranks}
        print(f"  {stem}: 10 ranks -> {stem_dir}/", flush=True)

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done. {len(wavs)} clips x 10 ranks under {a.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
