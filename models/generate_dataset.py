import librosa
import os
import numpy as np

from models.utils import utils

def collect_paths(base_path: str) -> list:
    dir: list = []
    ignore: set = {'.DS_Store'}
    for (dirpath, dirnames, filenames) in os.walk(base_path):
        dir.extend([dirpath+ name for name in filenames if name not in ignore])

    return dir

def raw_dataset(items: list, sample_rate: int = 16384, duration: float = 1.) -> np.ndarray:
    """
    From wav files, outputs a training set
    with shape `(n_samples, audio_channel, audio_length)`
    """
    t: tuple = tuple(utils.load_audio(item, sample_rate, duration)[0] for item in items)
    # remove samples where sample rate could be normalized
    t = tuple(i for i in t if i.shape[0] == sample_rate)
    stack: np.ndarray = np.vstack(t)
    return np.expand_dims(stack, axis=1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', action="store", dest="wavs", type=str, help="input wav files")
    parser.add_argument("-s", action="store", dest="save", type=str, help="save numpy output")
    args = parser.parse_args()

    """
    Usage:
    `python -m models.generate_dataset -w /path/to/wavs -s /path/to/saved`
    """

    # Collect wavs from path
    base_path = args.wavs
    print(f"Collecting from {base_path}")
    paths: list = collect_paths(base_path)

    # Generate dataset with proper shape
    dataset: np.ndarray = raw_dataset(paths)

    # Create a unique filename
    filename: str = f"{args.save}/dataset_{utils.fingerprint}.npy"

    # Save output
    np.save(filename, dataset)
    print(f"Saved to {filename}")
