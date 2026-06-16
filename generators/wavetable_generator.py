"""Wavetable synthesizer — pure numpy, no external audio library required.

Single-cycle waveforms (sine / saw / square / triangle) played back at a
given pitch via phase accumulation.  A simple one-pole low-pass filter is
applied before the ADSR envelope so the model can learn filter-type timbres
as well as pure oscillator timbres.

Parameter layout (8 floats, all normalised [0,1]):
  wave_type   pitch   cutoff   resonance   attack   decay   sustain   release

wave_type quantisation:
  [0.0, 0.25)  → sine
  [0.25, 0.50) → sawtooth
  [0.50, 0.75) → square
  [0.75, 1.0]  → triangle
"""

from __future__ import annotations

import numpy as np

WT_PARAMS: list[str] = [
    "wave_type", "pitch", "cutoff", "resonance",
    "attack", "decay", "sustain", "release",
]
N_WT_PARAMS: int = len(WT_PARAMS)  # 8

_MIDI_MIN, _MIDI_MAX = 24, 96
_TABLE_SIZE = 2048  # single-cycle wavetable length


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def denormalize(params_norm: np.ndarray) -> dict:
    p = np.clip(params_norm, 0.0, 1.0)
    return {
        "wave_type":  float(p[0]),
        "pitch_hz":   _midi_to_hz(_MIDI_MIN + float(p[1]) * (_MIDI_MAX - _MIDI_MIN)),
        # exponential mapping 100–10 000 Hz
        "cutoff_hz":  100.0 * (100.0 ** float(p[2])),
        "resonance":  float(p[3]) * 0.9,          # 0 – 0.9 (keep stable)
        "attack":     0.001 + float(p[4]) * 1.999,
        "decay":      0.001 + float(p[5]) * 1.999,
        "sustain":    float(p[6]),
        "release":    0.001 + float(p[7]) * 1.999,
    }


def _build_table(wave_type: float) -> np.ndarray:
    phi = np.linspace(0.0, 2.0 * np.pi, _TABLE_SIZE, endpoint=False)
    if wave_type < 0.25:   # sine
        return np.sin(phi).astype(np.float32)
    elif wave_type < 0.50:  # saw (additive, up to 32 harmonics)
        table = np.zeros(_TABLE_SIZE, dtype=np.float32)
        for k in range(1, 33):
            table += ((-1) ** (k + 1)) / k * np.sin(k * phi)
        return (table / np.max(np.abs(table))).astype(np.float32)
    elif wave_type < 0.75:  # square (odd harmonics)
        table = np.zeros(_TABLE_SIZE, dtype=np.float32)
        for k in range(1, 33, 2):
            table += (1.0 / k) * np.sin(k * phi)
        return (table / np.max(np.abs(table))).astype(np.float32)
    else:                   # triangle (odd harmonics, alternating sign)
        table = np.zeros(_TABLE_SIZE, dtype=np.float32)
        for k in range(1, 33, 2):
            table += ((-1) ** ((k - 1) // 2)) / (k * k) * np.sin(k * phi)
        return (table / np.max(np.abs(table))).astype(np.float32)


def _one_pole_lp(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """First-order IIR low-pass filter."""
    fc = min(cutoff_hz / sr, 0.499)
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += alpha * (x[i] - acc)
        y[i] = acc
    return y


def _adsr(n: int, attack_s: float, decay_s: float, sustain_level: float,
          release_s: float, sr: int) -> np.ndarray:
    a = min(int(attack_s * sr), n)
    d = min(int(decay_s * sr), n - a)
    # sustain fills the gap between decay end and the release phase start
    r = min(int(release_s * sr), n)
    s = max(0, n - a - d - r)

    env = np.empty(n, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32) if a else []
    env[a:a+d] = np.linspace(1.0, sustain_level, d, dtype=np.float32) if d else []
    env[a+d:a+d+s] = sustain_level
    env[a+d+s:] = np.linspace(sustain_level, 0.0, n - a - d - s,
                               dtype=np.float32)
    return env


def wavetable_render(params_norm: np.ndarray, sr: int = 48_000,
                     length: float = 1.0,
                     pitch_override_hz: float | None = None) -> np.ndarray:
    """Render normalised wavetable params to a float32 audio array."""
    p = denormalize(params_norm)
    pitch_hz = pitch_override_hz if pitch_override_hz is not None else p["pitch_hz"]

    n = int(length * sr)
    table = _build_table(p["wave_type"])

    # Phase accumulation — fractional index into table
    phase_inc = pitch_hz * _TABLE_SIZE / sr
    phases = (np.arange(n, dtype=np.float64) * phase_inc) % _TABLE_SIZE
    idx0 = phases.astype(np.int32) % _TABLE_SIZE
    idx1 = (idx0 + 1) % _TABLE_SIZE
    frac = (phases - phases.astype(np.int32)).astype(np.float32)
    audio = table[idx0] * (1.0 - frac) + table[idx1] * frac

    audio = _one_pole_lp(audio, p["cutoff_hz"], sr)

    env = _adsr(n, p["attack"], p["decay"], p["sustain"], p["release"], sr)
    audio *= env

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio /= peak
    return audio.astype(np.float32)
