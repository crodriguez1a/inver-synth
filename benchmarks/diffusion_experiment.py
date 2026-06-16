#!/usr/bin/env python3
"""
Path 5A — Zero-shot CLAP-conditioned audio diffusion experiment.

AudioLDM2-music generates synthesizer patch audio conditioned on text prompts
derived from tagger output. We measure CLAP cosine similarity between the
generated audio and the original patch to evaluate how much pretrained
diffusion models understand synthesizer timbres without fine-tuning.

Two prompting strategies are tested per patch:
  • short:  CLAP-branch keywords (tagger tags + timbre description)
  • long:   FLAN-T5 branch transcription (more descriptive sentence)

Usage:
    cd /path/to/inver-synth
    python benchmarks/diffusion_experiment.py [--steps 20] [--length 3.0]

Results saved to benchmarks/results/diffusion_experiment_YYYYMMDD/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

# ── Paths ──────────────────────────────────────────────────────────────────────

SYNTH_ROOT  = Path("/Users/carlosrodriguez/Projects/synthetroniq")
PREVIEWS    = SYNTH_ROOT / "data/previews"
REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# Add Synthetroniq backend to sys.path so we can reuse its CLAP encoder.
sys.path.insert(0, str(SYNTH_ROOT / "backend"))

# ── Experiment patches ─────────────────────────────────────────────────────────
# Chosen for diversity: FM hardware, analog-ish pad, ROMpler piano.
# Short prompt → CLAP text encoder (keyword conditioning).
# Long prompt  → FLAN-T5 branch (`transcription` param) for richer detail.

PATCHES = [
    {
        "label": "Alesis Airsynth/14-LFO-Abuse.flac",
        "note":  "FM-like, inver-synth confidence 62%",
        "short": "FM digital synthesizer, bright metallic harmonic lead, LFO modulation, electronic synth patch",
        "long":  "A bright digital FM synthesis lead patch with rapid LFO modulation creating a shimmering, metallic texture. Clean single note, no percussion.",
    },
    {
        "label": "Yamaha Dx100/16-Mono-Sax.flac",
        "note":  "FM hardware saxophone patch, confidence 41%",
        "short": "Yamaha DX FM synthesizer saxophone, bright digital wind instrument, monophonic synth",
        "long":  "A Yamaha DX series FM synthesizer patch emulating a saxophone. Bright, slightly buzzy digital timbre with the characteristic DX attack transient. Single sustained note.",
    },
    {
        "label": "Roland Jv 1080/U-010-Jet-Pad-2.flac",
        "note":  "warm pad, ROMpler digital synth",
        "short": "warm ambient pad synthesizer, lush atmospheric texture, spacious digital synth pad, Roland",
        "long":  "A lush, slow-attack ambient pad synthesizer with warm, spacious texture. Gentle layered tones with soft reverb tail. No melody or rhythm, just sustained atmospheric sound.",
    },
    {
        "label": "Alesis Quadrasynth Plus Piano/101-Gulch.flac",
        "note":  "piano ROMpler, inver-synth confidence 5%",
        "short": "acoustic piano sample, warm concert grand piano, natural piano tone",
        "long":  "An acoustic concert grand piano sample. Warm, natural timbre with clear attack transient and sustained decay. Single note, no accompaniment.",
    },
]

AUDIOLDM2_MODEL = "cvssp/audioldm2-music"
SR_GEN          = 16_000  # AudioLDM2 native output sample rate
SR_CLAP         = 48_000  # CLAP encoder input sample rate


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    return scipy.signal.resample(audio, n_out).astype(np.float32)


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps",  type=int,   default=20,   help="Diffusion inference steps")
    parser.add_argument("--length", type=float, default=3.0,  help="Generated clip length (s)")
    parser.add_argument("--device", type=str,   default="mps", help="torch device")
    args = parser.parse_args()

    today   = date.today().strftime("%Y%m%d")
    out_dir = RESULTS_DIR / f"diffusion_experiment_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CLAP encoder (Synthetroniq backend) ────────────────────────────────────
    print("Loading CLAP encoder (laion/clap-htsat-unfused)...")
    from synthetroniq.embeddings.clap import ClapEncoder
    from synthetroniq.search.tagger  import AudioTagger
    encoder = ClapEncoder()
    tagger  = AudioTagger(encoder)

    # ── Load patches ───────────────────────────────────────────────────────────
    print("\nLoading patches and computing reference CLAP embeddings...")
    loaded: list[dict] = []
    for spec in PATCHES:
        path = PREVIEWS / spec["label"]
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        raw, sr = sf.read(str(path), dtype="float32", always_2d=False)
        audio   = _resample(_mono(raw), sr, SR_CLAP)
        clip    = audio[:int(3.0 * SR_CLAP)]  # first 3 s for embedding
        emb     = encoder.encode_audio(clip, SR_CLAP)
        tags    = tagger.tag(emb)
        print(f"  {spec['label']}")
        print(f"    tags: {tags[:6]}")
        loaded.append({**spec, "audio": audio, "emb": emb, "tags": tags})

    if not loaded:
        print("No patches found. Check SYNTH_ROOT path.")
        return

    # ── AudioLDM2 pipeline ─────────────────────────────────────────────────────
    print(f"\nLoading {AUDIOLDM2_MODEL}  (~1.5 GB, cached after first run)...")
    import torch
    from diffusers import AudioLDM2Pipeline

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("  MPS unavailable — falling back to CPU")

    dtype = torch.float16 if device != "cpu" else torch.float32
    pipe  = AudioLDM2Pipeline.from_pretrained(AUDIOLDM2_MODEL, torch_dtype=dtype).to(device)

    # transformers 5.x split GenerationMixin out of GPT2Model; the cvssp checkpoint
    # sets architectures=["GPT2Model"] so diffusers loads the wrong class.
    # Re-load from the same local cache path as GPT2LMHeadModel (no network request).
    from transformers import GPT2LMHeadModel as _GPT2LMHead
    from transformers.generation.utils import GenerationMixin as _GenMixin
    if not isinstance(pipe.language_model, _GenMixin):
        lm_path = pipe.language_model.config._name_or_path
        pipe.language_model = _GPT2LMHead.from_pretrained(lm_path, torch_dtype=dtype).to(device)
        print("  (applied GPT2LMHeadModel patch for transformers 5.x compatibility)")

    # ── Generate and evaluate ──────────────────────────────────────────────────
    neg_prompt = "speech, voice, noise, drums, percussion, distortion, crackling"
    results    = []

    for pd in loaded:
        label = pd["label"]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  note:  {pd['note']}")
        print(f"  tags:  {pd['tags'][:6]}")
        print(f"  short: {pd['short'][:80]}")

        row: dict = {
            "label":  label,
            "note":   pd["note"],
            "tags":   pd["tags"],
            "short":  pd["short"],
            "long":   pd["long"],
        }

        for strategy in ("short", "long"):
            prompt        = pd["short"]
            transcription = pd["long"] if strategy == "long" else None

            print(f"\n  [{strategy} prompt]")
            with torch.no_grad():
                out = pipe(
                    prompt,
                    transcription=transcription,
                    negative_prompt=neg_prompt,
                    num_waveforms_per_prompt=1,
                    audio_length_in_s=args.length,
                    num_inference_steps=args.steps,
                    generator=torch.Generator(device=device).manual_seed(42),
                )
            gen_16k = out.audios[0]  # float32 ndarray

            safe = label.replace("/", "_").replace(".flac", "")
            wav_path = out_dir / f"{safe}_{strategy}.wav"
            sf.write(str(wav_path), gen_16k, SR_GEN)

            gen_48k = _resample(gen_16k, SR_GEN, SR_CLAP)
            gen_emb = encoder.encode_audio(gen_48k, SR_CLAP)
            sim     = _cosine(pd["emb"], gen_emb)

            print(f"  CLAP similarity: {sim:.4f}  → {wav_path.name}")
            row[f"{strategy}_sim"] = round(sim, 4)
            row[f"{strategy}_wav"] = wav_path.name

        results.append(row)

    # ── Cross-patch baseline ───────────────────────────────────────────────────
    cross = [
        _cosine(loaded[i]["emb"], loaded[j]["emb"])
        for i in range(len(loaded))
        for j in range(i + 1, len(loaded))
    ]
    baseline = round(float(np.mean(cross)), 4) if cross else None

    # ── Summary ────────────────────────────────────────────────────────────────
    short_sims = [r["short_sim"] for r in results if "short_sim" in r]
    long_sims  = [r["long_sim"]  for r in results if "long_sim"  in r]

    summary = {
        "date":             today,
        "model":            AUDIOLDM2_MODEL,
        "steps":            args.steps,
        "length_s":         args.length,
        "device":           device,
        "n_patches":        len(results),
        "cross_patch_baseline": baseline,
        "short_prompt_mean": round(float(np.mean(short_sims)), 4) if short_sims else None,
        "long_prompt_mean":  round(float(np.mean(long_sims)),  4) if long_sims  else None,
        "short_prompt_max":  round(float(np.max(short_sims)),  4) if short_sims else None,
        "long_prompt_max":   round(float(np.max(long_sims)),   4) if long_sims  else None,
        "patches": results,
    }

    json_path = out_dir / f"diffusion_experiment_{today}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Cross-patch baseline (avg pairwise CLAP sim): {baseline:.4f}")
    print()
    print(f"{'Label':<45}  {'short':>6}  {'long':>6}")
    print("-" * 60)
    for r in results:
        name = r["label"].split("/")[-1].replace(".flac", "")[:44]
        s = f"{r.get('short_sim', 0):.4f}"
        g = f"{r.get('long_sim', 0):.4f}"
        print(f"{name:<45}  {s:>6}  {g:>6}")
    print()
    if short_sims:
        print(f"Short mean: {summary['short_prompt_mean']:.4f}   max: {summary['short_prompt_max']:.4f}")
    if long_sims:
        print(f"Long  mean: {summary['long_prompt_mean']:.4f}   max: {summary['long_prompt_max']:.4f}")
    print(f"\nGenerated audio + JSON → {out_dir}")


if __name__ == "__main__":
    main()
