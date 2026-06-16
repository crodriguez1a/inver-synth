# InverSynth — Research Roadmap

## What was built (June 2026)

A CLAP-backbone + MLP regression head that estimates 2-operator FM synthesizer parameters
from a 512-dim CLAP audio embedding.

**Architecture**
- Backbone: `laion/clap-htsat-unfused` (frozen, 512-dim pooled output)
- Head: MLP 512 → 256 → 128 → 7, sigmoid output (all params normalised to [0,1])
- Loss: MSE on parameter coordinates
- Optimizer: AdamW + CosineAnnealingLR

**Generator: 2-operator sine FM (`generators/fm2op.py`)**
```
y(t) = sin(2π·fc·t + I·sin(2π·fm·t))    where fm = ratio × fc
```
Parameters (7): `pitch | ratio (int 1–8) | mod_index (0–10) | attack | decay | sustain | release`

Integer modulator ratios guarantee harmonic sidebands at every point in parameter space —
every random combination sounds musical. This was the key insight that made training viable
after a 4-operator random FM model produced mostly noise.

**Training run**
- Corpus: 50k synthetic clips, 48 kHz, 1 s each
- Epochs: 200 (best at epoch 101)
- Best val MSE: 0.035
- Device: Apple MPS
- Checkpoint: `checkpoints/fm2op.pt`

**Confidence metric: re-synthesis cosine similarity**

At inference time, the model predicts FM params, synthesises a C4 note from them,
re-embeds with CLAP, and computes cosine similarity between the input embedding and the
re-synthesis embedding. This is the only audio-space quality signal available without
ground-truth parameters.

---

## Library confidence results (2026-06-16)

50 patches sampled uniformly across 50 brands from the Synthetroniq patch library (6920 patches).
Full results in `benchmarks/results/library_confidence_20260616.json`.

| Statistic | Score |
|-----------|-------|
| Mean      | 28.7% |
| Median    | 28.5% |
| Max       | 62.4% (Alesis Airsynth / 14-LFO-Abuse) |
| Min       | 5.0%  (Alesis Quadrasynth Plus Piano / 101-Gulch) |

**Top 10 — most FM-approximable**

| Score | Patch |
|-------|-------|
| 62%   | Alesis Airsynth / 14-LFO-Abuse |
| 47%   | Generalmusic Equinox 61 / 004-Outburst |
| 46%   | Moog Rogue / 04-ELECTRIC-PIANO |
| 42%   | Bit One / 30 |
| 42%   | Roland Fantom Xr / 065-Super-G-DX |
| 41%   | Roland SE-02 / 40-Da-Lead-2 |
| 41%   | Access Virus A / B37-IQ-PAD-RP |
| 41%   | Yamaha DX100 / 16-Mono-Sax |
| 40%   | ASM Hydrasynth / C113-Synphony |
| 39%   | Roland XV-5080 / 111-Tap-Bass |

**Bottom 10 — least FM-approximable**

| Score | Patch |
|-------|-------|
| 16%   | E-mu Emulator III / Alex-Stone-Lux-Aeterna (orchestral sample) |
| 12%   | Roland SRX-08 / 381-Lo-Fi-Wurli |
| 9%    | Roland D-10 / A68-Timbass |
| 5%    | Alesis Quadrasynth Plus Piano / 101-Gulch (ROMpler piano) |

**Interpretation**: The model correctly discriminates FM-native hardware (Airsynth, DX100) from
sample-based instruments (Emulator III, Quadrasynth Piano). The ~28% mean reflects the hard
ceiling of 2-op FM expressiveness, not a failure of the regression head — most patches in this
library are multi-oscillator analog, ROM samples, or wavetable sounds that no 2-op FM model
can represent faithfully.

---

## Why it's not production-ready

1. **Wrong training objective**: MSE on normalised parameter coordinates is a poor proxy for
   perceptual similarity. A 5% error in `ratio` crosses an integer boundary and completely
   changes the timbre.

2. **Synthetic → real gap**: Training on random parameter combinations doesn't match the
   distribution of musical FM patches, which cluster around specific timbral regions.

3. **Expressiveness ceiling**: 2-op FM covers a subset of FM sounds. Most hardware patches
   use 4–6 operators with complex routing algorithms.

4. **No generalization beyond FM**: Approximately 70% of the library cannot be meaningfully
   approximated with FM synthesis at all.

---

## Improvement paths

### Path 1 — Perceptual loss in training
*Effort: ~2 days. Highest ROI for the existing architecture.*

Replace or augment the MSE loss with a re-synthesis cosine loss:

```
total_loss = MSE(pred_params, true_params) + α * (1 − cosine(CLAP(render(pred_params)), CLAP(render(true_params))))
```

This directly optimises the confidence metric. CLAP is frozen so the re-synthesis embedding
is computed with `torch.no_grad()` and the gradient only flows back through the MLP head.
Running the perceptual term every N steps (not every batch) keeps training cost manageable.

Expected outcome: noticeably better re-synthesis quality on FM-like patches without any
architectural changes.

### Path 2 — Real DX7 patch data
*Effort: ~3–4 days. Fixes the synthetic→real distribution gap.*

Large banks of Yamaha DX7 sysex presets are freely available (30k+ patches total).
Dexed is an open-source, headless DX7 emulator that can render them programmatically.

Steps:
1. Collect sysex banks, parse with a Python sysex parser
2. Render each patch at C4 via Dexed CLI
3. Embed with CLAP → train on real hardware patch distribution

The DX7 uses 6 operators; options are:
- Build a 6-op renderer (more expressive, harder to train)
- Project 6-op params to 2-op via dimensionality reduction (simpler, lossy)
- Train a 6-op model and keep 2-op as a fallback for non-DX patches

Expected outcome: much stronger results on FM hardware patches specifically; the model
learns the actual distribution of musical FM sounds rather than uniform random noise.

### Path 3 — Differentiable synthesis
*Effort: ~1 week. Most principled approach.*

Rewrite `fm2op_render` in PyTorch (sin, linspace, and ADSR are all differentiable).
Then the full training graph becomes:

```
frozen CLAP emb → MLP head → synth params → PyTorch FM render → frozen CLAP → cosine loss
```

Gradients flow from audio-space reconstruction loss directly back through the synthesiser
into the MLP weights. No separate perceptual loss phase required — the whole thing is one
differentiable objective. This is the DDSP (Differentiable Digital Signal Processing)
paradigm applied to FM.

Expected outcome: Path 1 quality improvement with a cleaner training setup. Also opens the
door to multi-operator models where parameter-space MSE becomes even more meaningless.

### Path 4 — Neural timbre transfer (skip parameter estimation)
*Effort: ~2 weeks. Highest ceiling; works for all patch types.*

Instead of estimating synth parameters, train a CLAP-conditioned neural vocoder:

```
(CLAP embedding, target f0 sequence) → neural vocoder → audio
```

The model learns to synthesise audio with the timbre of any patch at any requested pitch,
without going through a parameter bottleneck. Architecture candidates: HiFi-GAN conditioned
on CLAP, RAVE, or a simpler MLP-Mixer decoder.

Training data already exists: 6920 patches × preview audio, each with a known CLAP
embedding. Generating pitched variants (C2–C7) per patch for training is straightforward.

This approach generalises to the full library — sampled instruments, wavetable, analog,
digital — not just FM-approximable sounds. The tradeoff is training complexity and the need
for a neural inference runtime at the backend.

### Path 5 — CLAP-conditioned audio diffusion
*Effort: days to weeks depending on approach. Potentially the highest quality path and
the most natural architectural fit.*

The most important observation: we already have a 512-dim CLAP embedding for every patch in
the Synthetroniq library. Pretrained latent audio diffusion models (AudioLDM2, Stable Audio
Open) use exactly this signal as their conditioning input. This means pitch-transposed patch
audio may be achievable without any parameter estimation or custom training.

**Sub-path A — Zero-shot / prompting (days) — EXPERIMENT COMPLETED 2026-06-16**

Use an existing CLAP-conditioned diffusion model as-is. Condition on the patch's CLAP
embedding + a pitch-description string ("C4 sine tone", "middle C note"). No fine-tuning.
This is the "much simpler" version: the pretrained model already knows how to generate
audio from CLAP embeddings; we just need to steer it toward the right pitch.

**Results (model: `cvssp/audioldm2-music`, 20 steps, 3 s, MPS, seed 42)**

Script: `benchmarks/diffusion_experiment.py`
Data:   `benchmarks/results/diffusion_experiment_20260616/`

| Patch | Type | CLAP sim | vs baseline |
|---|---|---|---|
| Roland JV-1080 / Jet Pad 2 | warm pad | **0.545** | +96% above baseline |
| Alesis Quadrasynth / 101-Gulch | piano ROMpler | 0.370 | +33% |
| Alesis Airsynth / 14-LFO-Abuse | FM-like lead | 0.333 | +20% |
| Yamaha DX100 / 16-Mono-Sax | FM saxophone | **0.251** | ≈ baseline (0.278) |

Cross-patch baseline (avg pairwise CLAP sim between the 4 patches): **0.278**

Key findings:
1. **Pads and common timbres**: 0.545 — well above baseline, AudioLDM2-music clearly
   understands "warm ambient pad" at the timbral level. Promising path for non-FM patches.
2. **FM synthesis**: DX100 sax (0.251) is barely at baseline — the model has no specific
   knowledge of FM synthesis as a distinct timbral category.
3. **Short vs long prompts**: identical scores in every case. The `transcription` parameter
   (FLAN-T5 branch) had no measurable effect. CLAP text conditioning alone drives results.
4. **Inference speed**: ~3–4 s per clip on Apple M-series MPS at 20 steps (fast).

**Interpretation**: Zero-shot works well enough for broad timbre categories (pads, pianos)
but not for synthesis-specific sounds (FM, wavetable). Sub-path B (fine-tuning on the
patch library) is needed to get the model to understand FM synthesis as a timbre family.
The FLAN-T5 branch not contributing is a compatibility artifact of diffusers 0.38 + the
cvssp checkpoint — a known issue noted in the script.

**Sub-path B — Fine-tune on the patch library (1–2 weeks)**

Fine-tune a pretrained CLAP-conditioned diffusion model on (CLAP_embedding, pitch, audio)
triples generated from the existing 6920 previews:

```
for each patch:
    for each pitch in [C2, E2, G2, C3, ..., C6]:   # ~30 pitches
        render pitched variant via existing pitch-shift
        pair with patch CLAP embedding
→ ~200k training pairs, no new data collection needed
```

The fine-tuned model generates audio "in the style of" any CLAP embedding at any target
pitch. Generalises to the full library by construction.

**Sub-path C — Diffusion in parameter space (1 week)**

Instead of a deterministic MLP regression (Path 1–3), use a diffusion process in the
synthesizer parameter space conditioned on the CLAP embedding. Models the full posterior
distribution — a given timbre can map to multiple valid parameter configurations — and
samples from it. Related to DiffSynth-style approaches. Highest quality within the
FM-synthesis constraint; does not solve the expressiveness ceiling.

**Comparison to Path 4**

Path 4 (neural vocoder) and Path 5B (fine-tuned diffusion) both generalise to all patch
types. The practical differences:

| | Path 4 (vocoder) | Path 5B (diffusion) |
|---|---|---|
| Training data | existing previews + pitch variants | same |
| Inference speed | real-time (10–50ms) | 0.5–5s per note |
| Audio quality ceiling | good | higher |
| Model size | 50–200MB | 500MB–2GB |
| Pitch control | explicit (f0 conditioning) | via embedding + text |

For the Synthetroniq melody use case (render ~5 notes, latency not critical), diffusion
quality ceiling may be worth the inference cost. Sub-path A is the right first experiment —
zero cost to try.

---

## Re-enabling the feature

The melody button in the Flutter app is disabled via `_melodyEnabled = false` in
`mobile/lib/features/search/widgets/result_card.dart`. Set it to `true` to restore
the button, confidence bar, and lazy confidence fetch.

The backend endpoints remain intact:
- `GET /audio/melody?label=...` — renders FM melody via InverSynth (falls back to pitch-shift)
- `GET /audio/melody/confidence?label=...` — returns re-synthesis cosine similarity score

The backend loads InverSynth from `SYNTHETRONIQ_INVERSYNTH_CHECKPOINT` at startup.
Use `make macos-inversynth` or `make backend-inversynth` to start with it enabled.
