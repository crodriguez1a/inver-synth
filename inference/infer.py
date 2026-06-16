"""InverSynth inference — audio → synth parameters → re-synthesized audio.

This module is the integration point for Synthetroniq's backend.  The
backend can import `render_melody` directly once a checkpoint exists.

Standalone usage
----------------
    python -m inference.infer \\
        --audio patch.flac \\
        --synth fm \\
        --model checkpoints/fm_head.pt \\
        --midi 64 \\
        --out melody.wav

Python API (for Synthetroniq integration)
-----------------------------------------
    from inference.infer import InverSynthInferencer, render_melody

    # Load once at backend startup:
    inv = InverSynthInferencer("checkpoints/fm_head.pt")

    # Called per result card:
    melody_audio = inv.render_melody(patch_audio_np, sr=48000, notes=notes)
    # notes: list of (onset_s, offset_s, pitch_midi, amplitude)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


class InverSynthInferencer:
    """Loads a trained ClapMlpHead checkpoint and exposes synthesis methods."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        from models.clap_head import ClapMlpHead
        self.model = ClapMlpHead(
            synth_type=ckpt["synth_type"],
            hidden=tuple(ckpt["hidden"]),
            dropout=ckpt["dropout"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.synth_type = ckpt["synth_type"]

    # ------------------------------------------------------------------

    def predict_params(self, audio: np.ndarray, sr: int = 48_000) -> np.ndarray:
        """Return normalised synth parameters in [0, 1]."""
        return self.model.predict_from_audio(audio, sr=sr)

    def compute_confidence(self, audio: np.ndarray, sr: int = 48_000) -> float:
        """Cosine similarity between input and re-synthesized CLAP embeddings.

        Synthesizes a 1-second note at C4 from the predicted parameters, then
        measures how similar its CLAP embedding is to the original patch's
        embedding.  Range [0, 1] — higher means the FM approximation is closer
        to the input timbre in CLAP's embedding space.

        Note: CLAP embeddings are L2-normalised by the model, so cosine
        similarity equals the dot product.
        """
        emb_in = self.model.embed(audio, sr=sr)           # (512,)
        params  = self.model.head_forward(emb_in)          # (n_params,)
        synth   = self.synthesize_note(params, length=1.0, pitch_midi=60, sr=sr)
        emb_out = self.model.embed(synth, sr=sr)           # (512,)

        dot  = float(np.dot(emb_in, emb_out))
        norm = float(np.linalg.norm(emb_in) * np.linalg.norm(emb_out))
        return float(np.clip(dot / (norm + 1e-8), 0.0, 1.0))

    def synthesize_note(
        self,
        params: np.ndarray,
        length: float = 1.0,
        pitch_midi: int | None = None,
        sr: int = 48_000,
    ) -> np.ndarray:
        """Render a single note from predicted params at the given MIDI pitch."""
        if self.synth_type == "fm2op":
            from generators.fm2op import fm2op_render
            return fm2op_render(params, sr=sr, length=length,
                                pitch_override_midi=pitch_midi)
        elif self.synth_type == "fm":
            from generators.fm_numpy import _midi_to_hz, fm_render
            return fm_render(params, sr=sr, length=length,
                             pitch_override_midi=pitch_midi)
        elif self.synth_type in ("wt", "wavetable"):
            from generators.wavetable_generator import _midi_to_hz, wavetable_render
            hz = _midi_to_hz(pitch_midi) if pitch_midi is not None else None
            return wavetable_render(params, sr=sr, length=length,
                                    pitch_override_hz=hz)
        elif self.synth_type in ("sub", "subtractive"):
            from generators.subtractive_generator import _midi_to_hz, subtractive_render
            hz = _midi_to_hz(pitch_midi) if pitch_midi is not None else None
            return subtractive_render(params, sr=sr, length=length,
                                      pitch_override_hz=hz)
        raise ValueError(f"Unknown synth type: {self.synth_type!r}")

    def render_melody(
        self,
        patch_audio: np.ndarray,
        sr: int,
        notes: list[tuple[float, float, int, float]],
        out_sr: int = 48_000,
    ) -> np.ndarray:
        """Given a patch recording, predict timbre params and render a melody.

        Parameters
        ----------
        patch_audio : float32 mono array at *sr*
        sr          : sample rate of patch_audio
        notes       : list of (onset_s, offset_s, pitch_midi, amplitude)
        out_sr      : output sample rate (default 48 kHz, matches Synthetroniq)
        """
        params = self.predict_params(patch_audio, sr=sr)
        if not notes:
            return np.zeros(out_sr, dtype=np.float32)

        total_s = max(off for _, off, _, _ in notes) + 0.3
        out = np.zeros(int(total_s * out_sr), dtype=np.float32)

        # Cache rendered notes by pitch to avoid redundant synthesis
        note_cache: dict[int, np.ndarray] = {}
        for onset_s, offset_s, pitch_midi, amplitude in notes:
            note_len = max(1, int((offset_s - onset_s) * out_sr))
            if pitch_midi not in note_cache:
                note_cache[pitch_midi] = self.synthesize_note(
                    params, length=(offset_s - onset_s), pitch_midi=pitch_midi, sr=out_sr
                )
            segment = note_cache[pitch_midi][:note_len].copy()
            if len(segment) < note_len:
                segment = np.pad(segment, (0, note_len - len(segment)))

            # Short fade-out to avoid clicks on note boundaries
            fade = max(1, note_len // 20)
            segment[-fade:] *= np.linspace(1.0, 0.0, fade)
            segment *= amplitude

            start = int(onset_s * out_sr)
            end   = start + note_len
            if end > len(out):
                out = np.pad(out, (0, end - len(out)))
            out[start:end] += segment

        peak = float(np.max(np.abs(out)))
        if peak > 1e-8:
            out = out / peak * 0.7
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="InverSynth inference: audio → synth params → re-synthesis.")
    ap.add_argument("--audio",  required=True,  help="Input patch audio file (.wav/.flac)")
    ap.add_argument("--model",  required=True,  help="Checkpoint .pt from training/train.py")
    ap.add_argument("--out",    default="output.wav", help="Output WAV path")
    ap.add_argument("--midi",   type=int, nargs="+", default=[60],
                    help="Target MIDI note(s).  Single note or space-separated list for a scale.")
    ap.add_argument("--dur",    type=float, default=0.5, help="Note duration in seconds")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    inv = InverSynthInferencer(args.model)
    params = inv.predict_params(audio, sr=sr)

    print(f"Predicted {len(params)} normalised params for synth={inv.synth_type!r}:")
    print("  " + "  ".join(f"{v:.3f}" for v in params))

    # Build notes from the MIDI list
    notes = [
        (i * args.dur, (i + 1) * args.dur, m, 0.8)
        for i, m in enumerate(args.midi)
    ]
    melody = inv.render_melody(audio, sr=sr, notes=notes)

    sf.write(args.out, melody, 48_000, subtype="PCM_16")
    print(f"Written → {args.out}")


if __name__ == "__main__":
    main()
