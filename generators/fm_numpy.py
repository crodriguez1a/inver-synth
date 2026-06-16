"""Pure-numpy FM synthesizer — replaces synthplayer dependency.

Four FM oscillators (sine/saw/tri/square carriers, each with its own
sine LFO for frequency modulation) mixed and passed through an ADSR
envelope.  All parameters are normalised to [0, 1]; use `denormalize`
to recover physical values.

FM parameter layout (21 floats, index order matches FM_PARAMS):
  osc 1 (sine):     f1  v1  A1  B1
  osc 2 (saw):      f2  v2  A2  B2
  osc 3 (tri):      f3  v3  A3  B3
  osc 4 (square):   f4  v4  A4  B4
  envelope:         attack  decay  sustain_time  sustain_level  release
"""

from __future__ import annotations

import numpy as np

# Parameter names in vector order.
FM_PARAMS: list[str] = [
    "f1", "v1", "A1", "B1",
    "f2", "v2", "A2", "B2",
    "f3", "v3", "A3", "B3",
    "f4", "v4", "A4", "B4",
    "attack", "decay", "sustain_time", "sustain_level", "release",
]
N_FM_PARAMS: int = len(FM_PARAMS)  # 21

_MIDI_MIN, _MIDI_MAX = 24, 96  # carrier frequency range


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def denormalize(params_norm: np.ndarray) -> dict:
    """Map normalised [0,1] vector → physical parameter dict."""
    p = np.clip(params_norm, 0.0, 1.0)
    d: dict = {}
    for i, name in enumerate(FM_PARAMS):
        v = float(p[i])
        if name.startswith("f"):         # carrier freq
            d[name] = _midi_to_hz(_MIDI_MIN + v * (_MIDI_MAX - _MIDI_MIN))
        elif name.startswith("v"):       # LFO freq (1–30 Hz)
            d[name] = 1.0 + v * 29.0
        elif name.startswith("A"):       # carrier amplitude (0–1)
            d[name] = v
        elif name.startswith("B"):       # mod amplitude (0–1500 Hz)
            d[name] = v * 1500.0
        elif name in ("attack", "decay", "release"):   # 0.001–2 s
            d[name] = 0.001 + v * 1.999
        elif name == "sustain_time":     # 0.1–2 s
            d[name] = 0.1 + v * 1.9
        elif name == "sustain_level":    # 0–1
            d[name] = v
    return d


def _adsr(n: int, attack_s: float, decay_s: float, sustain_level: float,
          sustain_s: float, release_s: float, sr: int) -> np.ndarray:
    a = min(int(attack_s * sr), n)
    d = min(int(decay_s * sr), n - a)
    s = min(int(sustain_s * sr), n - a - d)
    r = min(int(release_s * sr), n - a - d - s)

    env = np.empty(n, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32) if a else []
    env[a:a+d] = np.linspace(1.0, sustain_level, d, dtype=np.float32) if d else []
    env[a+d:a+d+s] = sustain_level
    env[a+d+s:a+d+s+r] = np.linspace(sustain_level, 0.0, r, dtype=np.float32) if r else []
    env[a+d+s+r:] = 0.0
    return env


def fm_render(params_norm: np.ndarray, sr: int = 48_000, length: float = 1.0,
              pitch_override_midi: int | None = None) -> np.ndarray:
    """Render normalised FM params to a float32 audio array.

    If *pitch_override_midi* is given, all carrier frequencies are scaled
    proportionally so that f1 maps to the requested MIDI note (useful for
    melody rendering once you've estimated timbre parameters).
    """
    p = denormalize(params_norm)
    n = int(length * sr)
    t = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    tau = 2.0 * np.pi

    # Optional pitch override: scale all carrier freqs by the same ratio.
    if pitch_override_midi is not None:
        target_hz = _midi_to_hz(pitch_override_midi)
        ratio = target_hz / max(p["f1"], 1e-3)
        for k in ("f1", "f2", "f3", "f4"):
            p[k] = p[k] * ratio

    def _lfo(v: float) -> np.ndarray:
        return np.sin(tau * v * t)

    def _fm_phase(f: float, v: float, B: float) -> np.ndarray:
        return tau * f * t + B * _lfo(v)

    def _saw(ph: np.ndarray) -> np.ndarray:
        return 2.0 * (ph / tau - np.floor(ph / tau + 0.5))

    def _tri(ph: np.ndarray) -> np.ndarray:
        return 1.0 - 4.0 * np.abs(ph / tau - np.floor(ph / tau + 0.5))

    def _sqr(ph: np.ndarray) -> np.ndarray:
        return np.sign(np.sin(ph)).astype(np.float64)

    osc1 = p["A1"] * np.sin(_fm_phase(p["f1"], p["v1"], p["B1"]))
    osc2 = p["A2"] * _saw(_fm_phase(p["f2"], p["v2"], p["B2"]))
    osc3 = p["A3"] * _tri(_fm_phase(p["f3"], p["v3"], p["B3"]))
    osc4 = p["A4"] * _sqr(_fm_phase(p["f4"], p["v4"], p["B4"]))

    mix = (osc1 + osc2 + osc3 + osc4).astype(np.float32)

    env = _adsr(n, p["attack"], p["decay"], p["sustain_level"],
                p["sustain_time"], p["release"], sr)
    mix *= env

    peak = float(np.max(np.abs(mix)))
    if peak > 1e-8:
        mix /= peak
    return mix
