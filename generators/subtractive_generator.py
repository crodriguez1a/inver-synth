"""Subtractive synthesizer — pure numpy.

Oscillator (sawtooth or square) → state-variable low-pass filter → ADSR.
The filter uses a Chamberlin SVF which gives resonance without instability
in the normalised [0, 0.9] range.

Parameter layout (8 floats, all normalised [0,1]):
  osc_type   pitch   cutoff   resonance   attack   decay   sustain   release

osc_type: < 0.5 → sawtooth, >= 0.5 → square
"""

from __future__ import annotations

import numpy as np

SUB_PARAMS: list[str] = [
    "osc_type", "pitch", "cutoff", "resonance",
    "attack", "decay", "sustain", "release",
]
N_SUB_PARAMS: int = len(SUB_PARAMS)  # 8

_MIDI_MIN, _MIDI_MAX = 24, 96


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def denormalize(params_norm: np.ndarray) -> dict:
    p = np.clip(params_norm, 0.0, 1.0)
    return {
        "osc_type":  float(p[0]),
        "pitch_hz":  _midi_to_hz(_MIDI_MIN + float(p[1]) * (_MIDI_MAX - _MIDI_MIN)),
        "cutoff_hz": 100.0 * (100.0 ** float(p[2])),  # 100–10 000 Hz
        "resonance": float(p[3]) * 0.9,
        "attack":    0.001 + float(p[4]) * 1.999,
        "decay":     0.001 + float(p[5]) * 1.999,
        "sustain":   float(p[6]),
        "release":   0.001 + float(p[7]) * 1.999,
    }


def _oscillator(osc_type: float, pitch_hz: float, n: int, sr: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    phase = (pitch_hz * t) % 1.0
    if osc_type < 0.5:  # sawtooth
        return (2.0 * phase - 1.0).astype(np.float32)
    else:               # square
        return np.where(phase < 0.5, 1.0, -1.0).astype(np.float32)


def _svf_lp(x: np.ndarray, cutoff_hz: float, resonance: float, sr: int) -> np.ndarray:
    """Chamberlin state-variable filter — low-pass output."""
    f = 2.0 * np.sin(np.pi * min(cutoff_hz / sr, 0.499))
    q = 1.0 - resonance  # q=1 → no resonance, q→0 → high resonance
    y = np.empty_like(x)
    lp = 0.0
    bp = 0.0
    for i in range(len(x)):
        lp = lp + f * bp
        hp = x[i] - lp - q * bp
        bp = f * hp + bp
        y[i] = lp
    return y


def _adsr(n: int, attack_s: float, decay_s: float, sustain_level: float,
          release_s: float, sr: int) -> np.ndarray:
    a = min(int(attack_s * sr), n)
    d = min(int(decay_s * sr), n - a)
    r = min(int(release_s * sr), n)
    s = max(0, n - a - d - r)

    env = np.empty(n, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32) if a else []
    env[a:a+d] = np.linspace(1.0, sustain_level, d, dtype=np.float32) if d else []
    env[a+d:a+d+s] = sustain_level
    env[a+d+s:] = np.linspace(sustain_level, 0.0, n - a - d - s, dtype=np.float32)
    return env


def subtractive_render(params_norm: np.ndarray, sr: int = 48_000,
                       length: float = 1.0,
                       pitch_override_hz: float | None = None) -> np.ndarray:
    """Render normalised subtractive params to a float32 audio array."""
    p = denormalize(params_norm)
    pitch_hz = pitch_override_hz if pitch_override_hz is not None else p["pitch_hz"]

    n = int(length * sr)
    audio = _oscillator(p["osc_type"], pitch_hz, n, sr)
    audio = _svf_lp(audio, p["cutoff_hz"], p["resonance"], sr)

    env = _adsr(n, p["attack"], p["decay"], p["sustain"], p["release"], sr)
    audio *= env

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio /= peak
    return audio.astype(np.float32)
