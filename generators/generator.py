
from scipy.io.wavfile import write as write_wav
import numpy as np
from typing import Dict, Tuple, Sequence, List
#ParamValue = Tuple[str,float,List[float]]
import random
import os
import h5py
import argparse

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
    def generate(self,parameters:dict,filename:str,length:float,sample_rate:int,extra:dict)->np.ndarray:
        audio = self.do_generate(parameters,filename,length,sample_rate,extra)
        if self.normalise():
            max = np.max(np.absolute(audio))
            if max > 0:
                audio = audio / max
        if not self.creates_wave_file():
            self.write_file(audio,filename,sample_rate)

    def do_generate(self,parameters:dict,filename:str,length:float,sample_rate:int,extra:dict)->np.ndarray:
        print("Someone needs to write this method! Generating silence in {} with parameters:{}".format(filename,str(parameters)))
        return np.zeros(int(length*sample_rate))

    def creates_wave_file(self) -> bool:
        return False

    def normalise(self):
        return True

    # Assumes that the data is -1..1 floating point
    def write_file(self,data : np.ndarray,filename:str,sample_rate:int):
        int_data = (data * np.iinfo(np.int16).max).astype(int)
        write_wav(filename, sample_rate, data)


"""
This class runs through a parameter set, gets it to generate parameter settings
then runs the sound generator over it.
"""


class DatasetCreator():
    def __init__(self, name: str, dataset_dir: str, wave_file_dir: str, parameters: ParameterSet, normalise: bool = True):
        self.name = name
        self.parameters = parameters
        self.dataset_dir = dataset_dir
        self.wave_file_dir = wave_file_dir
        self.index = 0
        self.normalise = normalise
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(f"{wave_file_dir}/{name}", exist_ok=True)

    def generate(self, sound_generator: SoundGenerator, length: float = 0.1,
                 sample_rate: int = 44100, max: int = 10, method: str = 'complete', extra: dict = {}):
        print("+"*40)
        print(f"Generating Dataset {self.name}, {max} files of {length}s at {sample_rate}/s")
        print(f"Output waves: {self.wave_file_dir}, datasets: {self.dataset_dir}")
        print("+"*40)
        self.index = 0
        dataset: List[Sample] = []
        if method == "complete":
            dataset = self.parameters.recursively_generate_all()
        else:
            dataset = self.parameters.sample_space(sample_size=max)

        n_samps = int(length*sample_rate)
        #print("First sample: {}".format(dataset[0]))
        #print("First parameters: {}".format(dict(dataset[0].value_list())))
        records = len(dataset)
        param_size = len(dataset[0].encode())

        datafile = h5py.File(self.get_dataset_filename(
            dataset, "data", 'hdf5'), 'w')
        filenames = datafile.create_dataset(
            "files", (records,), dtype=h5py.string_dtype())
        labels = datafile.create_dataset("labels", (records, param_size))

        for p in dataset:
            params = self.parameters.to_settings(p)
            filename = self.get_wave_filename(self.index)
            audio = sound_generator.generate(
                params,filename,
                length, sample_rate, extra)

            filenames[self.index] = filename
            labels[self.index] = p.encode()
            if self.index % 1000 == 0:
                print("Generating example {}".format(self.index))
            self.index = self.index + 1
            datafile.flush()
        datafile.close()
        # self.save_audio(dataset)
        # self.save_labels(dataset)
        self.save_parameters()

    # def save_audio(self,dataset:List[Sample]):
        #audio = tuple(t.audio for t in dataset)
        #audio_data = np.expand_dims(np.vstack(audio), axis=1)
        #print("Audio data: {}".format(audio_data.shape))
        # np.save(self.get_dataset_filename(dataset,"input"),audio_data)

    # def save_labels(self,dataset:List[Sample]):
        #param = tuple(t.encode() for t in dataset)
        ##param_data = np.expand_dims(np.vstack(param), axis=1)
        #param_data = np.vstack(param)
        #print("Param data: {}".format(param_data.shape))
        # np.save(self.get_dataset_filename(dataset,"labels"),param_data)

    def save_parameters(self):
        self.parameters.save_json(
            self.get_dataset_filename(None, "params", 'json'))
        self.parameters.save(self.get_dataset_filename(None, "params", 'pckl'))

    def get_dataset_filename(self, dataset, type: str, extension: str = "txt") -> str:
        return f"{self.dataset_dir}/{self.name}_{type}.{extension}"

    def get_wave_filename(self, index: int) -> str:
        return f"{self.wave_file_dir}/{self.name}/{self.name}_{index:05d}.wav"

def default_generator_argparse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_examples', type=int, dest='samples', action='store', default=64,
                        help='Number of examples to create')
    parser.add_argument('--name', type=str, dest='name',  required=True,
                        help='Name of datasets to create')
    parser.add_argument('--dataset_directory', type=str, dest='data_dir',  default='test_datasets',
                        help='Directory to put datasets')
    parser.add_argument('--wavefile_directory', type=str, dest='wave_dir',  default='test_waves',
                        help='Directory to put wave files. Will have the dataset name appended automatically')
    parser.add_argument('--length', type=float, dest='length', default=1.0,
                        help='Length of each sample in seconds')
    parser.add_argument('--sample_rate', type=int, dest='sample_rate', default=16384,
                        help='Sample rate (Samples/second)')
    parser.add_argument('--sampling_method', type=str, dest='method', default='random', choices=['random'],
                        help='Method to use for generating examples. Currently only random, but may include whole space later')
    return parser



if __name__ == "__main__":
    gen = SoundGenerator()
    parameters = ParameterSet([
        Parameter("p1", [100, 110, 120, 130, 140]),
        Parameter("p2", [200, 220, 240, 260, 280])
    ])
    g = DatasetCreator("example_generator",
                       dataset_dir="test_datasets",
                       wave_file_dir="test_waves/example/",
                       parameters=parameters)
    g.generate(sound_generator=gen, length=1,
               sample_rate=16384, method="random", max=30)
