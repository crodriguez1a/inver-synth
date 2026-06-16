"""CLAP encoder (frozen) + lightweight MLP regression head.

Architecture per the DAFx 2024 finding: a frozen pre-trained audio
backbone (CLAP, 512-dim) with a small MLP head trained on synthetic
data outperforms an end-to-end CNN by ~3× on MSE.

Usage
-----
    from models.clap_head import ClapMlpHead, N_PARAMS
    model = ClapMlpHead(synth_type="fm")
    params = model.predict_from_audio(audio_np, sr=48000)
    # params: np.ndarray of shape (N_FM_PARAMS,), all in [0,1]

    # Or embed first, then run only the head (fast at training time):
    emb = model.embed(audio_np, sr=48000)    # (512,)
    params = model.head_forward(emb)         # (N_FM_PARAMS,)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from generators.fm_numpy import N_FM_PARAMS
from generators.fm2op import N_FM2OP_PARAMS
from generators.wavetable_generator import N_WT_PARAMS
from generators.subtractive_generator import N_SUB_PARAMS

SYNTH_TYPES = ("fm", "fm2op", "wavetable", "subtractive")
N_PARAMS: dict[str, int] = {
    "fm":          N_FM_PARAMS,
    "fm2op":       N_FM2OP_PARAMS,
    "wavetable":   N_WT_PARAMS,
    "subtractive": N_SUB_PARAMS,
}

_CLAP_DIM = 512
_CLAP_ID  = "laion/clap-htsat-unfused"


def _build_mlp(in_dim: int, n_params: int, hidden: tuple[int, ...],
               dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, n_params), nn.Sigmoid()]
    return nn.Sequential(*layers)


class ClapMlpHead(nn.Module):
    """CLAP (frozen) backbone + trainable MLP regression head.

    Parameters
    ----------
    synth_type : one of "fm", "wavetable", "subtractive"
    hidden     : hidden layer widths for the MLP (default: two layers)
    dropout    : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        synth_type: str = "fm",
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if synth_type not in N_PARAMS:
            raise ValueError(f"synth_type must be one of {SYNTH_TYPES}")
        self.synth_type = synth_type
        self.n_params   = N_PARAMS[synth_type]

        # Lazy-load CLAP to avoid the import at module level.
        self._clap_loaded = False
        self.register_buffer("_dummy", torch.zeros(1))  # tracks device

        self.head = _build_mlp(_CLAP_DIM, self.n_params, hidden, dropout)

    # ------------------------------------------------------------------
    # Internal CLAP loading (lazy, so the model file can be imported
    # without transformers in environments that only need synthesis).
    # ------------------------------------------------------------------

    def _load_clap(self) -> None:
        if self._clap_loaded:
            return
        from transformers import ClapModel, ClapProcessor  # type: ignore
        self._processor = ClapProcessor.from_pretrained(_CLAP_ID)
        self._clap      = ClapModel.from_pretrained(_CLAP_ID)
        self._clap.eval()
        for param in self._clap.parameters():
            param.requires_grad_(False)
        self._clap_loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int = 48_000) -> np.ndarray:
        """Return the 512-dim CLAP embedding for a single audio clip."""
        self._load_clap()
        inputs = self._processor(
            audio=audio,
            return_tensors="pt",
            sampling_rate=sr,
        )
        out = self._clap.get_audio_features(**inputs)
        # transformers ≥5.x returns BaseModelOutputWithPooling; <5.x returns Tensor
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        return emb[0].cpu().numpy()  # (512,)

    @torch.no_grad()
    def embed_batch(self, audios: list[np.ndarray], sr: int = 48_000) -> np.ndarray:
        """Return CLAP embeddings for a list of audio clips. Shape: (N, 512)."""
        return np.stack([self.embed(a, sr) for a in audios])

    def head_forward(self, emb: np.ndarray) -> np.ndarray:
        """Run the MLP head on a pre-computed embedding. Returns params in [0,1]."""
        t = torch.from_numpy(emb).float().unsqueeze(0)  # (1, 512)
        with torch.no_grad():
            out = self.head(t)
        return out[0].cpu().numpy()

    def predict_from_audio(self, audio: np.ndarray, sr: int = 48_000) -> np.ndarray:
        """End-to-end: audio → CLAP embedding → predicted params."""
        emb = self.embed(audio, sr)
        return self.head_forward(emb)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Training forward pass: (batch, 512) → (batch, n_params)."""
        return self.head(embeddings)
