
from scipy.io.wavfile import write as write_wav
import numpy as np
from typing import Dict, Tuple, Sequence, List
#ParamValue = Tuple[str,float,List[float]]
import random

from generators.parameters import *
from generators.sound_generator import SoundGenerator



class DatasetGenerator():
    def __init__(self,name: str, dataset_dir:str, wave_file_dir:str, parameters: ParameterSet, normalise:bool=True ):
        self.name = name
        self.parameters = parameters
        self.dataset_dir = dataset_dir
        self.wave_file_dir = wave_file_dir
        self.index = 0
        self.normalise = normalise

    def generate(self,sound_generator:SoundGenerator,length:float=0.1,sample_rate:int=44100,max:int=10,method:str='complete'):
        self.index = 0
        dataset:List[Sample] = []
        if method == "complete":
            dataset = self.parameters.recursively_generate_all()
        else:
            dataset = self.parameters.sample_space(sample_size=max)

        n_samps = int(length*sample_rate)
        for p in dataset:
            p.length = length
            p.sample_rate = sample_rate
        print("First sample: {}".format(dataset[0]))
        print("First parameters: {}".format(dict(dataset[0].value_list())))
        for p in dataset:
            params = self.parameters.to_settings(p)
            audio = sound_generator.generate(
                params,self.get_wave_filename(self.index),
                p.length, p.sample_rate)
            if self.normalise:
                max = np.max(np.absolute(audio))
                if max > 0:
                    audio = audio / max
            p.audio = audio[:n_samps]
            if not sound_generator.creates_wave_file():
                self.write_file(audio,self.get_wave_filename(self.index),sample_rate)
            self.index = self.index + 1
        self.save_audio(dataset)
        self.save_labels(dataset)
        self.save_parameters()


    def save_audio(self,dataset:List[Sample]):
        audio = tuple(t.audio for t in dataset)
        audio_data = np.expand_dims(np.vstack(audio), axis=1)
        print("Audio data: {}".format(audio_data.shape))
        np.save(self.get_dataset_filename(dataset,"input"),audio_data)

    def save_labels(self,dataset:List[Sample]):
        param = tuple(t.encode() for t in dataset)
        #param_data = np.expand_dims(np.vstack(param), axis=1)
        param_data = np.vstack(param)
        print("Param data: {}".format(param_data.shape))
        np.save(self.get_dataset_filename(dataset,"labels"),param_data)

    def save_parameters(self):
        self.parameters.save(self.get_dataset_filename(None,"params"))


    def get_dataset_filename(self,dataset,type:str)->str:
        return "{}/{}_{}".format(self.dataset_dir,self.name,type)

    def get_wave_filename(self,index:int)->str:
        return "{}/{}_{:05d}.wav".format(self.wave_file_dir,self.name,index)


    # Assumes that the data is -1..1 floating point
    def write_file(self,data : np.ndarray,filename:str,sample_rate:int):
        int_data = (data * np.iinfo(np.int16).max).astype(int)
        write_wav(filename, sample_rate, data)



if __name__ == "__main__":
    gen = SoundGenerator()
    parameters=ParameterSet([
        Parameter("p1",[100,110,120,130,140]),
        Parameter("p2",[200,220,240,260,280])
    ])
    g = DatasetGenerator("example_generator",
        dataset_dir="test_datasets",
        wave_file_dir="test_waves/example/",
        parameters=parameters )
    g.generate(sound_generator=gen,length=1,sample_rate=16384,method="random",max=30)
