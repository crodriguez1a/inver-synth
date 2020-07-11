import argparse

# ParamValue = Tuple[str,float,List[float]]
import os
import os.path
from typing import List

import h5py
import numpy as np
from scipy.io.wavfile import write as write_wav

from generators.parameters import *

"""
This is a base class to derive different kinds of sound generator from (e.g.
custom synthesis, VST plugins)
"""


class SoundGenerator:
    """
    This is now a wrapper round the 'real' generation function
    to handle normalising and saving
    """

    def generate(
        self,
        parameters: dict,
        filename: str,
        length: float,
        sample_rate: int,
        extra: dict,
        normalise: bool = True,
    ) -> np.ndarray:
        audio = self.do_generate(parameters, filename, length, sample_rate, extra)
        if normalise:
            max = np.max(np.absolute(audio))
            if max > 0:
                audio = audio / max
        if not self.creates_wave_file():
            self.write_file(audio, filename, sample_rate)

    def do_generate(
        self,
        parameters: dict,
        filename: str,
        length: float,
        sample_rate: int,
        extra: dict,
    ) -> np.ndarray:
        print(
            "Someone needs to write this method! Generating silence in {} with parameters:{}".format(
                filename, str(parameters)
            )
        )
        return np.zeros(int(length * sample_rate))

    def creates_wave_file(self) -> bool:
        return False

    # Assumes that the data is -1..1 floating point
    def write_file(self, data: np.ndarray, filename: str, sample_rate: int):
        # REVIEW: is this needed?
        # int_data = (data * np.iinfo(np.int16).max).astype(int)
        write_wav(filename, sample_rate, data)


"""
This class runs through a parameter set, gets it to generate parameter settings
then runs the sound generator over it.
"""


class DatasetCreator:
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        wave_file_dir: str,
        parameters: ParameterSet,
        normalise: bool = True,
    ):
        self.name = name
        self.parameters = parameters
        self.dataset_dir = dataset_dir
        self.wave_file_dir = wave_file_dir
        self.normalise = normalise
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(f"{wave_file_dir}/{name}", exist_ok=True)

    def create_parameters(
        self,
        max: int = 10,
        method: str = "complete",
        extra: dict = {},
        force_create=False,
    ) -> str:
        filename = self.get_dataset_filename("data", "hdf5")
        if os.path.isfile(filename) and not force_create:
            print(
                "Parameter file exists, not recreating (use --regenerate_samples if you want to force)"
            )
            return filename
        print("+" * 40)
        print(f"Generating Dataset {self.name}, {max} examples")
        print(f"Datasets: {self.dataset_dir}")
        print("+" * 40)

        # Save out the parameters first
        self.save_parameters()

        # Generate the set of samples (could switch to generators,
        # but need to figure out arbitrary size arrays in HDF5)
        dataset: List[Sample] = []
        if method == "complete":
            dataset = self.parameters.recursively_generate_all()
        else:
            dataset = self.parameters.sample_space(sample_size=max)

        # Create the data file and add all the points to it
        with h5py.File(filename, "w") as datafile:
            # Figure out the sizes to store
            records = len(dataset)
            param_size = len(dataset[0].encode())

            # Add columns to it
            filenames = datafile.create_dataset(
                "files", (records,), dtype=h5py.string_dtype()
            )
            parameters = datafile.create_dataset(
                "parameters", (records,), dtype=h5py.string_dtype()
            )
            labels = datafile.create_dataset("labels", (records, param_size))
            audio_exists = datafile.create_dataset(
                "audio_exists", (records,), dtype=np.bool
            )

            # Generate the sample points
            for index, point in enumerate(dataset):
                params = self.parameters.to_settings(point)
                filenames[index] = self.get_wave_filename(index)
                labels[index] = point.encode()
                parameters[index] = json.dumps(params)
                audio_exists[index] = False
                if index % 1000 == 0:
                    print("Generating parameters for example {}".format(index))
            datafile.flush()
        datafile.close()

        return filename

    def generate_audio(
        self,
        sound_generator: SoundGenerator,
        length: float = 0.1,
        sample_rate: int = 44100,
        extra: dict = {},
        dataset_filename=None,
        force_generate=False,
    ):
        if dataset_filename is None:
            dataset_filename = self.get_dataset_filename("data", "hdf5")

        print("+" * 40)
        print(
            f"Generating Audio for Dataset {self.name} ({dataset_filename}), with {length}s at {sample_rate}/s"
        )
        print(f"Output waves: {self.wave_file_dir}, datasets: {self.dataset_dir}")
        print("+" * 40)

        with h5py.File(dataset_filename, "r+") as datafile:
            for name, value in datafile.items():
                print(f"{name}: {value}")
            # Get the columns
            filenames = datafile.get("files")
            print(filenames)
            parameters = datafile.get("parameters")
            print(parameters)
            audio_exists = datafile.get("audio_exists")
            print(audio_exists)

            for index, filename in enumerate(filenames):
                if (
                    audio_exists[index]
                    and os.path.isfile(filename)
                    and not force_generate
                ):
                    print(f"Audio exists for index {index} ({filename})")
                else:
                    print(f"Generating Audio for index {index} ({filename})")
                    params = json.loads(parameters[index])
                    audio = sound_generator.generate(
                        params,
                        filename,
                        length,
                        sample_rate,
                        extra,
                        normalise=self.normalise,
                    )
                    audio_exists[index] = bool(audio)
                    datafile.flush()
                if index % 1000 == 0:
                    print("Generating example {}".format(index))

    def save_parameters(self):
        self.parameters.save_json(self.get_dataset_filename("params", "json"))
        self.parameters.save(self.get_dataset_filename("params", "pckl"))

    def get_dataset_filename(self, type: str, extension: str = "txt") -> str:
        return f"{self.dataset_dir}/{self.name}_{type}.{extension}"

    def get_wave_filename(self, index: int) -> str:
        return f"{self.wave_file_dir}/{self.name}/{self.name}_{index:05d}.wav"


def default_generator_argparse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--num_examples",
        type=int,
        dest="samples",
        action="store",
        default=150,
        help="Number of examples to create",
    )
    parser.add_argument(
        "--name",
        type=str,
        dest="name",
        default="InverSynth",
        help="Name of datasets to create",
    )
    parser.add_argument(
        "--dataset_directory",
        type=str,
        dest="data_dir",
        default="test_datasets",
        help="Directory to put datasets",
    )
    parser.add_argument(
        "--wavefile_directory",
        type=str,
        dest="wave_dir",
        default="test_waves",
        help="Directory to put wave files. Will have the dataset name appended automatically",
    )
    parser.add_argument(
        "--length",
        type=float,
        dest="length",
        default=1.0,
        help="Length of each sample in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        dest="sample_rate",
        default=16384,
        help="Sample rate (Samples/second)",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        dest="method",
        default="random",
        choices=["random"],
        help="Method to use for generating examples. Currently only random, but may include whole space later",
    )
    parser.add_argument(
        "--regenerate_samples",
        action="store_true",
        help="Regenerate the set of points to explore if it exists (will also force regenerating audio)",
    )
    parser.add_argument(
        "--regenerate_audio",
        action="store_true",
        help="Regenerate audio files if they exists",
    )
    parser.add_argument(
        "--normalise", action="store_true", help="Regenerate audio files if they exists"
    )
    return parser


def generate_examples(
    gen: SoundGenerator, parameters: ParameterSet, args=None, extra={}
):
    if not args:
        parser = default_generator_argparse()
        args = parser.parse_args()

    g = DatasetCreator(
        name=args.name,
        dataset_dir=args.data_dir,
        wave_file_dir=args.wave_dir,
        parameters=parameters,
        normalise=args.normalise,
    )

    g.create_parameters(
        max=args.samples, method=args.method, force_create=args.regenerate_samples
    )

    g.generate_audio(
        sound_generator=gen,
        length=args.length,
        sample_rate=args.sample_rate,
        extra=extra,
        force_generate=args.regenerate_audio | args.regenerate_samples,
    )


if __name__ == "__main__":
    gen = SoundGenerator()
    parameters = ParameterSet(
        [
            Parameter("p1", [100, 110, 120, 130, 140]),
            Parameter("p2", [200, 220, 240, 260, 280]),
        ]
    )
    g = DatasetCreator(
        "example_generator",
        dataset_dir="test_datasets",
        wave_file_dir="test_waves/example/",
        parameters=parameters,
    )
    g.generate_examples(sound_generator=gen, parameters=parameters)
