
from scipy.io.wavfile import write as write_wav
import numpy as np
from typing import Dict, Tuple, Sequence, List
Sample = Tuple[int,float,np.ndarray,np.ndarray]
Dataset = Sequence[Sample]


# Dataset format: numpy.ndarray (num_points,1,num_samples)

class Generator():
    def __init__(self,name: str, dataset_dir:str, wave_file_dir:str, parameters: list, length:1.0, sample_rate:44100):
        self.name = name
        self.parameters = parameters
        self.dataset_dir = dataset_dir
        self.wave_file_dir = wave_file_dir
        self.length = length
        self.sample_rate = sample_rate
        self.index = 0

    def run(self):
        dataset = self.generate_all()
        training_samples = []

        audio = tuple(t[2][:16384] for t in dataset)
        audio_data = np.expand_dims(np.vstack(audio), axis=1)
        print("Audio data: {}".format(audio_data.shape))
        np.save(self.get_dataset_filename(dataset,"audio"),audio_data)

        param = tuple(t[3] for t in dataset)
        param_data = np.expand_dims(np.vstack(param), axis=1)
        print("Param data: {}".format(param_data.shape))
        np.save(self.get_dataset_filename(dataset,"params"),audio_data)

    # Runs through the whole parameter space, setting up parameters and calling the generation function
    # Excuse slightly hacky recusions - sure there's a more numpy-ish way to do it!
    def generate_all(self,parameter_list: list = None, parameter_set=[],return_list=[])->Dataset:
        if not parameter_list:
            parameter_list = self.parameters
            self.index = 0
        param = parameter_list[0]
        remaining = parameter_list[1:]
        for p in param.levels:
            ps = parameter_set.copy()
            ps.append((param.name,p))
            if len(remaining) == 0:
                return_list.append(self.generate_sound(ps))
            else:
                self.generate_all(remaining,ps,return_list)
        return return_list

    def generate_sound(self,parameter_set:list) -> Sample:
        index = self.index
        self.index = self.index + 1
        param_vals = [t[1] for t in parameter_set]
        filename:str = self.get_wave_filename(index)
        data = self.do_sound_generation(dict(parameter_set),filename)
        if self.should_write_wave():
            self.write_file(data,filename)
        return (index,filename,data,param_vals)

    def do_sound_generation(self,parameter_set:dict,base_filename:str)->np.ndarray:
        print("Someone needs to write this method! Generating {} with parameters:{}".format(base_filename,str(parameter_set)))

    def get_dataset_filename(self,dataset:Dataset,type:str)->str:
        return "{}/{}_{}".format(self.dataset_dir,self.name,type)

    def get_wave_filename(self,index:int)->str:
        return "{}/{}_{:05d}.wav".format(self.wave_file_dir,self.name,index)

    def should_write_wave(self)->bool:
        return True

    # Assumes that the data is -1..1 floating point
    def write_file(self,data : np.ndarray,filename:str):
        int_data = (data * np.iinfo(np.int16).max).astype(int)
        sample_rate = self.sample_rate
        write_wav(filename, sample_rate, data)

class Parameter:
    def __init__(self,name: str, levels: list):
        self.name=name
        self.levels = levels


if __name__ == "__main__":
    g = Generator(name="test_generator",datset_dir="test_datasets",wave_file_dir="test_waves",
        parameters=[
        Parameter("p1",[100,110,120,130,140]),
        Parameter("p2",[200,220,240,260,280])
    ])
    g.run()
