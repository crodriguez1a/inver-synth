"""Pre-compute CLAP embeddings for a synthetic training corpus.

This separates the expensive audio generation + CLAP encoding step from
the fast MLP training loop.  Run once per synth type, then train many
times with different hyperparameters on the cached embeddings.

Usage
-----
    python -m training.generate_embeddings --synth fm   --n 50000 --out data/fm.npz
    python -m training.generate_embeddings --synth wt   --n 50000 --out data/wt.npz
    python -m training.generate_embeddings --synth sub  --n 50000 --out data/sub.npz

Output .npz contains two arrays:
    embeddings  float32  (N, 512)
    params      float32  (N, n_params)   — normalised [0, 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _gen_sample(synth_type: str, sr: int, length: float):
    """Return (params_norm, audio) for a random parameter point."""
    if synth_type == "fm":
        from generators.fm_numpy import N_FM_PARAMS, fm_render
        params = np.random.rand(N_FM_PARAMS).astype(np.float32)
        audio  = fm_render(params, sr=sr, length=length)
    elif synth_type in ("fm2op", "fm2"):
        from generators.fm2op import N_FM2OP_PARAMS, fm2op_render
        params = np.random.rand(N_FM2OP_PARAMS).astype(np.float32)
        audio  = fm2op_render(params, sr=sr, length=length)
    elif synth_type in ("wt", "wavetable"):
        from generators.wavetable_generator import N_WT_PARAMS, wavetable_render
        params = np.random.rand(N_WT_PARAMS).astype(np.float32)
        audio  = wavetable_render(params, sr=sr, length=length)
    elif synth_type in ("sub", "subtractive"):
        from generators.subtractive_generator import N_SUB_PARAMS, subtractive_render
        params = np.random.rand(N_SUB_PARAMS).astype(np.float32)
        audio  = subtractive_render(params, sr=sr, length=length)
    else:
        raise ValueError(f"Unknown synth type: {synth_type!r}")
    return params, audio


def generate(
    synth_type: str,
    n: int,
    out_path: str | Path,
    sr: int = 48_000,
    length: float = 1.0,
    batch_size: int = 64,
    seed: int | None = None,
) -> None:
    if seed is not None:
        np.random.seed(seed)

    from models.clap_head import _CLAP_ID
    from transformers import ClapModel, ClapProcessor
    import torch

    processor = ClapProcessor.from_pretrained(_CLAP_ID)
    clap      = ClapModel.from_pretrained(_CLAP_ID)
    clap.eval()
    for p in clap.parameters():
        p.requires_grad_(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine n_params from one sample
    p0, _ = _gen_sample(synth_type, sr, length)
    n_params = len(p0)

    embeddings = np.empty((n, 512), dtype=np.float32)
    params_arr = np.empty((n, n_params), dtype=np.float32)

    print(f"Generating {n:,} {synth_type!r} clips at {sr} Hz, {length}s each…")
    with tqdm(total=n, unit="clips") as bar:
        i = 0
        while i < n:
            bs = min(batch_size, n - i)
            batch_params = []
            batch_audio  = []
            for _ in range(bs):
                p, a = _gen_sample(synth_type, sr, length)
                batch_params.append(p)
                batch_audio.append(a)

            # Batch-process through processor + CLAP in one forward pass
            inputs = processor(audio=batch_audio, return_tensors="pt",
                               sampling_rate=sr, padding=True)
            with torch.no_grad():
                out = clap.get_audio_features(**inputs)
                emb = out.pooler_output if hasattr(out, "pooler_output") else out
            embeddings[i:i+bs] = emb.cpu().numpy()
            params_arr[i:i+bs] = np.stack(batch_params)
            i += bs
            bar.update(bs)

    np.savez_compressed(out_path, embeddings=embeddings, params=params_arr)
    print(f"Saved → {out_path}  "
          f"(embeddings {embeddings.shape}, params {params_arr.shape})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-compute CLAP embeddings for synth training data.")
    ap.add_argument("--synth",  required=True, choices=["fm", "fm2op", "fm2", "wt", "wavetable", "sub", "subtractive"])
    ap.add_argument("--n",      type=int, default=50_000, help="Number of random examples")
    ap.add_argument("--out",    required=True, help="Output .npz path")
    ap.add_argument("--sr",     type=int, default=48_000)
    ap.add_argument("--length", type=float, default=1.0, help="Clip length in seconds")
    ap.add_argument("--seed",   type=int, default=None)
    args = ap.parse_args()
    generate(args.synth, args.n, args.out, sr=args.sr, length=args.length, seed=args.seed)


if __name__ == "__main__":
    main()
