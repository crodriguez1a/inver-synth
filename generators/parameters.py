import numpy as np
from typing import Dict, Tuple, Sequence, List
from dataclasses import dataclass
import random
from pickle import dump
import math

"""
A setting for a parameter, with its oneHOT encoding
"""
@dataclass
class ParamValue:
    name:str
    value:float
    encoding:List[float]


"""
A sample point - the parameter values, the oneHOT encoding and the audio
"""
@dataclass
class Sample:
    #parameter_values: List[Tuple[str,float]]
    #parameter_encoding:List[List[float]]
    parameters : List[ParamValue]
    #length:float=0.1
    #sample_rate:int = 44100
    #audio:np.ndarray = np.zeros(1)

    def value_list(self)->List[Tuple[str,float]]:
        return [(p.name,p.value) for p in self.parameters]

    def encode(self)->List[float]:
        return np.hstack([p.encoding for p in self.parameters])

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
        encoding = np.zeros(len(self.levels)).astype(float)
        encoding[index] = 1.0
        return ParamValue(
            name=self.name,
            #Actual value
            value = self.levels[index],
            #One HOT encoding
            encoding = encoding )

    def decode(self,one_hot:List[float])->ParamValue:
        ind = np.array(one_hot).argmax()
        return self.get_value(ind)

    def from_output(self,current_output:List[float])->Tuple[ParamValue,List[float]]:
        param_data = current_output[:len(self.levels)]
        remaining = current_output[len(self.levels):]
        my_val = self.decode(param_data)
        return (my_val,remaining)



class ParameterSet:
    def __init__(self,parameters: List[Parameter], fixed_parameters:dict={} ):
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters

    def sample_space(self,sample_size=1000)->Sequence[Sample]:
        print("Sampling {} points from parameter space".format(sample_size))
        dataset = []
        for i in range(sample_size):
            params = [p.sample() for p in self.parameters]
            dataset.append(Sample(params))
            if i % 1000 == 0:
                print("Sampling iteration: {}".format(i))
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

    def to_settings(self,p:Sample):
        params = self.fixed_parameters.copy()
        params.update(dict(p.value_list()))
        return params

    def decode(self,output:List[float])->List[ParamValue]:
        values = []
        for p in self.parameters:
            v,output = p.from_output(output)
            values.append(v)
        if len(output) > 0:
            print("Leftover output!: {}".format(output))
        return values

    def save(self,filename):
        with open(filename, 'wb') as file:
            dump(self,file)

    def explain(self):
        levels = 0
        for p in self.parameters:
            levels += len(p.levels)
        return {
            "n_variable":len(self.parameters),
            "n_fixed":len(self.fixed_parameters),
            "levels":levels
        }

"""
Generates evenly spaced parameter values
paper:
The rest of the synthesizer parameters ranges are quantized evenly to 16
classes according to the following ranges ...
For each parameter, the first and last classes correspond to its range limits
"""
def param_range(steps,min,max):
    ext = float(max - min)
    return [n * ext/(steps-1) + min for n in range(steps)]

"""
Generates a set of frequencies as per paper
paper: f = 2^(n/12)/ 440Hz with n in 0..15
"""
def freq_range(steps):
    return [math.pow(2,n/12) * 440 for n in range(steps)]
