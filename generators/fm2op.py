"""Two-operator sine FM synthesizer — DX7-style, always musical.

Classic Chowning FM: y(t) = sin(2π·fc·t + I·sin(2π·fm·t))
where fm = ratio × fc and ratio is always an integer, so all sidebands
fall on harmonics.  Every point in parameter space sounds musical.

Parameter layout (7 floats, all normalised [0,1]):
  pitch   ratio   mod_index   attack   decay   sustain   release

Mappings:
  pitch      0→1  ·  MIDI 24–96   (C1 – C7)
  ratio      0→1  →  integer 1–8  (modulator/carrier frequency ratio)
  mod_index  0→1  →  0.0–10.0     (FM depth: 0=sine, ~2=bell, 8+=metallic)
  attack     0→1  →  0.001–2.0 s
  decay      0→1  →  0.001–2.0 s
  sustain    0→1  →  0.0–1.0      (level during sustain)
  release    0→1  →  0.001–2.0 s
"""

from __future__ import annotations

import numpy as np

FM2OP_PARAMS: list[str] = [
    "pitch", "ratio", "mod_index",
    "attack", "decay", "sustain", "release",
]
N_FM2OP_PARAMS: int = len(FM2OP_PARAMS)  # 7

_MIDI_MIN, _MIDI_MAX = 24, 96


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def denormalize(params_norm: np.ndarray) -> dict:
    p = np.clip(params_norm, 0.0, 1.0)
    midi = _MIDI_MIN + float(p[0]) * (_MIDI_MAX - _MIDI_MIN)
    return {
        "pitch_hz":  _midi_to_hz(midi),
        "ratio":     max(1, round(1 + float(p[1]) * 7)),  # integer 1–8
        "mod_index": float(p[2]) * 10.0,
        "attack":    0.001 + float(p[3]) * 1.999,
        "decay":     0.001 + float(p[4]) * 1.999,
        "sustain":   float(p[5]),
        "release":   0.001 + float(p[6]) * 1.999,
    }


def _adsr(n: int, attack_s: float, decay_s: float, sustain_level: float,
          release_s: float, sr: int) -> np.ndarray:
    a = min(int(attack_s * sr), n)
    d = min(int(decay_s * sr), n - a)
    r = min(int(release_s * sr), n)
    s = max(0, n - a - d - r)

    env = np.empty(n, dtype=np.float32)
    env[:a]           = np.linspace(0.0, 1.0, a,   dtype=np.float32) if a else []
    env[a:a+d]        = np.linspace(1.0, sustain_level, d, dtype=np.float32) if d else []
    env[a+d:a+d+s]    = sustain_level
    env[a+d+s:]       = np.linspace(sustain_level, 0.0, n - a - d - s, dtype=np.float32)
    return env


def fm2op_render(
    params_norm: np.ndarray,
    sr: int = 48_000,
    length: float = 1.0,
    pitch_override_midi: int | None = None,
) -> np.ndarray:
    """Render normalised 2-op FM params to a float32 audio array."""
    p = denormalize(params_norm)
    fc = _midi_to_hz(pitch_override_midi) if pitch_override_midi is not None else p["pitch_hz"]
    fm = p["ratio"] * fc

    n = int(length * sr)
    t = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)

    modulator = p["mod_index"] * np.sin(2.0 * np.pi * fm * t)
    audio     = np.sin(2.0 * np.pi * fc * t + modulator).astype(np.float32)

    env   = _adsr(n, p["attack"], p["decay"], p["sustain"], p["release"], sr)
    audio *= env

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio /= peak
    return audio
