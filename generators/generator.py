
from scipy.io.wavfile import write as write_wav
import numpy as np
from typing import Dict, Tuple, Sequence, List
from keras.utils import to_categorical
from dataclasses import dataclass
ParamValue = Tuple[str,float,List[float]]
import random

from generators.sound_generator import SoundGenerator

"""
A sample point - the parameter values, the oneHOT encoding and the audio
"""
@dataclass
class Sample:
    parameter_values: List[Tuple[str,float]]
    parameter_encoding:List[List[float]]
    length:float=0.1
    sample_rate:int = 44100
    audio:np.ndarray = np.zeros(10)


# Dataset format: numpy.ndarray (num_points,1,num_samples)

class DatasetGenerator():
    def __init__(self,name: str, dataset_dir:str, wave_file_dir:str, parameters: list, normalise:bool=True ):
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
        print("First parameters: {}".format(dict(dataset[0].parameter_values)))
        for p in dataset:
            #print("Sample: {}".format(p))
            audio = sound_generator.generate(
                dict(p.parameter_values),self.get_wave_filename(self.index),
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


    def save_audio(self,dataset:List[Sample]):
        audio = tuple(t.audio for t in dataset)
        audio_data = np.expand_dims(np.vstack(audio), axis=1)
        print("Audio data: {}".format(audio_data.shape))
        np.save(self.get_dataset_filename(dataset,"input"),audio_data)

    def save_labels(self,dataset:List[Sample]):
        param = tuple(t.parameter_encoding for t in dataset)
        #param_data = np.expand_dims(np.vstack(param), axis=1)
        param_data = np.vstack(param)
        print("Param data: {}".format(param_data.shape))
        np.save(self.get_dataset_filename(dataset,"labels"),param_data)


    def get_dataset_filename(self,dataset,type:str)->str:
        return "{}/{}_{}".format(self.dataset_dir,self.name,type)

    def get_wave_filename(self,index:int)->str:
        return "{}/{}_{:05d}.wav".format(self.wave_file_dir,self.name,index)


    # Assumes that the data is -1..1 floating point
    def write_file(self,data : np.ndarray,filename:str,sample_rate:int):
        int_data = (data * np.iinfo(np.int16).max).astype(int)
        write_wav(filename, sample_rate, data)


# Model architectures

class Parameter:
    def __init__(self,name: str, levels: list):
        self.name=name
        self.levels = levels

    def get_levels(self)->List[ParamValue]:
        return [self.get_value(i) for i in range(len(self.levels))]

    def sample(self)->ParamValue:
        index:int = random.choice(range(len(self.levels)))
        return self.get_value(index)

    def get_value(self,index:int)->ParamValue:
        return (
            self.name,
            #Actual value
            self.levels[index],
            #One HOT encoding
            to_categorical(index,num_classes=len(self.levels)))

    def decode(self,one_hot:List[float])->float:
        d2 = list(to_categorical(one_hot).astype(int)[1])
        ind = d2.index(1)
        return self.levels[ind]

    def from_output(self,current_output:List[float])->Tuple[float,List[float]]:
        param_data = current_output[:len(self.levels)]
        remainder = current_output[len(self.levels):]
        my_val = self.decode(param_data)
        return (my_val,remainder)



class ParameterSet:
    def __init__(self,parameters: List[Parameter] ):
        self.parameters = parameters

    def sample_space(self,sample_size=1000)->Sequence[Sample]:
        print("Sampling {} points from parameter space".format(sample_size))
        dataset = []
        for i in range(sample_size):
            params = [p.sample() for p in self.parameters]
            dataset.append(self.to_sample(params))
        return dataset


    # Runs through the whole parameter space, setting up parameters and calling the generation function
    # Excuse slightly hacky recusions - sure there's a more numpy-ish way to do it!
    def recursively_generate_all(self,parameter_list: list=None, parameter_set=[],return_list=[])->Sequence[Sample]:
        print("Generating entire parameter space")
        if parameter_list is None:
            parameter_list = self.parameters
        param = parameter_list[0]
        remaining = parameter_list[1:]
        for p in param.levels:
            ps = parameter_set.copy()
            ps.append((param.name,p))
            if len(remaining) == 0:
                return_list.append(self.generate_sound(ps))
            else:
                self.recursively_generate_all(remaining,ps,return_list)
        return return_list

    def to_sample(self,settings:List[ParamValue])->Sample:
        #print(str(settings))
        oneHOT = np.hstack([p[2] for p in settings])
        params = [(p[0],p[1]) for p in settings] #Tuples of param name,value
        #print("OneHOT: {}".format(oneHOT))
        return Sample(params, oneHOT )


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
