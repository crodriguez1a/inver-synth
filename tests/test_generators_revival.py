"""Smoke tests for the pure-numpy revival generators.

These tests have no ML dependencies (no torch, no CLAP) — just numpy.
They verify shape, dtype, normalisation, and that pitch overrides work.
"""

import numpy as np
import pytest

SR = 48_000
LENGTH = 1.0
N = int(SR * LENGTH)


# ── fm2op ──────────────────────────────────────────────────────────────────────

from generators.fm2op import fm2op_render, denormalize, FM2OP_PARAMS, N_FM2OP_PARAMS


def _rng_params(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(N_FM2OP_PARAMS).astype(np.float32)


def test_fm2op_param_count():
    assert len(FM2OP_PARAMS) == N_FM2OP_PARAMS == 7


def test_fm2op_output_shape():
    audio = fm2op_render(_rng_params(), sr=SR, length=LENGTH)
    assert audio.shape == (N,)


def test_fm2op_output_dtype():
    audio = fm2op_render(_rng_params(), sr=SR, length=LENGTH)
    assert audio.dtype == np.float32


def test_fm2op_normalised():
    audio = fm2op_render(_rng_params(), sr=SR, length=LENGTH)
    assert float(np.max(np.abs(audio))) <= 1.0 + 1e-5


def test_fm2op_pitch_override():
    params = _rng_params(seed=1)
    a = fm2op_render(params, sr=SR, length=LENGTH, pitch_override_midi=60)
    b = fm2op_render(params, sr=SR, length=LENGTH, pitch_override_midi=72)
    # Different pitches → different waveforms
    assert not np.allclose(a, b)


def test_fm2op_deterministic():
    params = _rng_params(seed=42)
    a = fm2op_render(params, sr=SR, length=LENGTH)
    b = fm2op_render(params, sr=SR, length=LENGTH)
    np.testing.assert_array_equal(a, b)


def test_fm2op_denormalize_bounds():
    p = denormalize(np.zeros(N_FM2OP_PARAMS))
    assert p["ratio"] >= 1
    assert p["mod_index"] >= 0.0
    assert p["attack"] > 0
    assert p["sustain"] >= 0.0

    p = denormalize(np.ones(N_FM2OP_PARAMS))
    assert p["ratio"] <= 8
    assert p["mod_index"] <= 10.0


# ── wavetable ──────────────────────────────────────────────────────────────────

from generators.wavetable_generator import wavetable_render, N_WT_PARAMS


def _wt_params(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(N_WT_PARAMS).astype(np.float32)


def test_wavetable_param_count():
    assert N_WT_PARAMS == 8


def test_wavetable_output_shape():
    audio = wavetable_render(_wt_params(), sr=SR, length=LENGTH)
    assert audio.shape == (N,)


def test_wavetable_output_dtype():
    audio = wavetable_render(_wt_params(), sr=SR, length=LENGTH)
    assert audio.dtype == np.float32


def test_wavetable_normalised():
    audio = wavetable_render(_wt_params(), sr=SR, length=LENGTH)
    assert float(np.max(np.abs(audio))) <= 1.0 + 1e-5


def test_wavetable_deterministic():
    params = _wt_params(seed=7)
    np.testing.assert_array_equal(
        wavetable_render(params, sr=SR, length=LENGTH),
        wavetable_render(params, sr=SR, length=LENGTH),
    )


# ── subtractive ────────────────────────────────────────────────────────────────

from generators.subtractive_generator import subtractive_render, N_SUB_PARAMS


def _sub_params(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(N_SUB_PARAMS).astype(np.float32)


def test_subtractive_param_count():
    assert N_SUB_PARAMS == 8


def test_subtractive_output_shape():
    audio = subtractive_render(_sub_params(), sr=SR, length=LENGTH)
    assert audio.shape == (N,)


def test_subtractive_output_dtype():
    audio = subtractive_render(_sub_params(), sr=SR, length=LENGTH)
    assert audio.dtype == np.float32


def test_subtractive_normalised():
    audio = subtractive_render(_sub_params(), sr=SR, length=LENGTH)
    assert float(np.max(np.abs(audio))) <= 1.0 + 1e-5


def test_subtractive_deterministic():
    params = _sub_params(seed=3)
    np.testing.assert_array_equal(
        subtractive_render(params, sr=SR, length=LENGTH),
        subtractive_render(params, sr=SR, length=LENGTH),
    )
