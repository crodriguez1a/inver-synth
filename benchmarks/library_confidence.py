"""Library confidence benchmark.

Samples patches uniformly across brands from the Synthetroniq library and
scores each one using the InverSynth re-synthesis confidence endpoint.

Usage:
    python benchmarks/library_confidence.py [--url URL] [--n N] [--seed SEED] [--out FILE]

    --url   Base URL of the running Synthetroniq backend  (default: http://localhost:8000)
    --n     Number of brands to sample — one patch per brand (default: 50)
    --seed  Random seed for reproducibility               (default: 42)
    --out   Write full results to this JSON file          (default: benchmarks/results/library_confidence_<date>.json)

The backend must be running with InverSynth enabled:
    make backend-inversynth   # or make macos-inversynth
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError


def _get(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def run(base_url: str, n: int, seed: int, out_path: Path) -> None:
    base_url = base_url.rstrip("/")

    print(f"Fetching brand list from {base_url}/synths …")
    try:
        brands_resp = _get(f"{base_url}/synths")
    except URLError as exc:
        sys.exit(f"Cannot reach backend at {base_url}: {exc}")

    brands = brands_resp["brands"]
    rng = random.Random(seed)
    sampled_brands = rng.sample(brands, min(n, len(brands)))

    print(f"Sampling {len(sampled_brands)} brands (seed={seed}) …\n")

    results: list[dict] = []
    errors: list[str] = []

    for i, brand_info in enumerate(sampled_brands):
        brand = brand_info["name"]
        patches_resp = _get(f"{base_url}/synths/{quote(brand)}")
        patches = patches_resp["patches"]
        patch = rng.choice(patches)
        label = f"{brand}/{patch}.mp3"

        try:
            conf_resp = _get(f"{base_url}/audio/melody/confidence?label={quote(label)}")
            confidence = conf_resp.get("confidence")
        except Exception as exc:
            confidence = None
            errors.append(f"{label}: {exc}")

        marker = "" if confidence is not None else "  [no score — InverSynth not loaded?]"
        score_str = f"{confidence:.3f}" if confidence is not None else "None"
        print(f"[{i+1:3d}/{len(sampled_brands)}]  {score_str}  {brand} / {patch}{marker}")

        results.append({"label": label, "brand": brand, "patch": patch, "confidence": confidence})

    scores = [r["confidence"] for r in results if r["confidence"] is not None]

    if not scores:
        print("\nNo scores returned. Is the backend running with InverSynth enabled?")
        return

    results_sorted = sorted(results, key=lambda r: r["confidence"] or 0, reverse=True)

    print("\n─── Top 10 (best FM match) ───")
    for r in results_sorted[:10]:
        print(f"  {r['confidence']:.3f}  {r['brand']} / {r['patch']}")

    print("\n─── Bottom 10 (worst FM match) ───")
    for r in results_sorted[-10:]:
        print(f"  {r['confidence']:.3f}  {r['brand']} / {r['patch']}")

    print(f"\nN={len(scores)}  mean={statistics.mean(scores):.3f}  "
          f"median={statistics.median(scores):.3f}  "
          f"stdev={statistics.stdev(scores):.3f}  "
          f"min={min(scores):.3f}  max={max(scores):.3f}")

    payload = {
        "date": str(date.today()),
        "backend_url": base_url,
        "n_sampled": len(sampled_brands),
        "seed": seed,
        "stats": {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "stdev": round(statistics.stdev(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        },
        "results": results_sorted,
        "errors": errors,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull results written to {out_path}")


def main() -> None:
    today = date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url",  default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--n",    type=int, default=50, help="Number of brands to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out",  default=f"benchmarks/results/library_confidence_{today}.json",
                        help="Output JSON path")
    args = parser.parse_args()
    run(args.url, args.n, args.seed, Path(args.out))


if __name__ == "__main__":
    main()
