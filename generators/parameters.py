import numpy as np
from typing import Dict, Tuple, Sequence, List
from dataclasses import dataclass
import random
from pickle import dump
import math
import json
import regex

@dataclass
class ParamValue:
    """
    A setting for a parameter, with its oneHOT encoding
    """
    name: str
    value: float
    encoding: List[float]


@dataclass
class Sample:
    """
    A sample point - the parameter values, the oneHOT encoding and the audio
    """
    parameters: List[ParamValue]

    def value_list(self) -> List[Tuple[str, float]]:
        return [(p.name, p.value) for p in self.parameters]

    def encode(self) -> List[float]:
        return np.hstack([p.encoding for p in self.parameters])


class Parameter:
    """
    Representation of a synthesis parameter: name, levels and an ID
    """

    def __init__(self, name: str, levels: list, id=""):
        self.name = name
        self.levels = levels
        self.id = id

    def get_levels(self) -> List[ParamValue]:
        return [self.get_value(i) for i in range(len(self.levels))]

    def sample(self) -> ParamValue:
        index: int = random.choice(range(len(self.levels)))
        return self.get_value(index)

    def get_value(self, index: int) -> ParamValue:
        """ Returns the value for a given index, i.e. a parameter setting given
        the argmax of a OneHOT encoding """

        encoding = np.zeros(len(self.levels)).astype(float)
        encoding[index] = 1.0
        return ParamValue(
            name=self.name,
            # Actual value
            value=self.levels[index],
            # One HOT encoding
            encoding=encoding)

    def decode(self, one_hot: List[float]) -> ParamValue:
        """ Decode a OneHOT array into a parameter value - could do something
        more intelligent than value[argmax(input)] """
        ind = np.array(one_hot).argmax()
        return self.get_value(ind)

    def from_output(self, current_output: List[float]) -> Tuple[ParamValue, List[float]]:
        """ Takes some portion of a OneHOT encoding, decodes the first N values
        for this parameters value, and returns the ParamValue and remainder of
        the list - designed to be rolled over an output vector """
        param_data = current_output[:len(self.levels)]
        remaining = current_output[len(self.levels):]
        my_val = self.decode(param_data)
        return (my_val, remaining)

    def to_json(self):
        """ Returns a dict representing this parameter """
        return {"name": self.name, "levels": self.levels, "id": self.id}


class ParameterSet:
    def __init__(self, parameters: List[Parameter], fixed_parameters: dict = {}, ids={}):
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
        self.ids = ids

    def sample_space(self, sample_size=1000) -> Sequence[Sample]:
        print("Sampling {} points from parameter space".format(sample_size))
        dataset = []
        for i in range(sample_size):
            params = [p.sample() for p in self.parameters]
            dataset.append(Sample(params))
            if i % 1000 == 0:
                print("Sampling iteration: {}".format(i))
        return dataset

    def recursively_generate_all(self, parameter_list: list = None,
                                 parameter_set=[],
                                 return_list=[]) -> Sequence[Sample]:
        """ Runs through the whole parameter space, setting up parameters and
        calling the generation function Excuse slightly hacky recusions -
        sure there's a more numpy-ish way to do it!
        """
        print("Generating entire parameter space")
        if parameter_list is None:
            parameter_list = self.parameters
        param = parameter_list[0]
        remaining = parameter_list[1:]
        for p in param.levels:
            ps = parameter_set.copy()
            ps.append((param.name, p))
            if len(remaining) == 0:
                return_list.append(self.generate_sound(ps))
            else:
                self.recursively_generate_all(remaining, ps, return_list)
        return return_list

    def to_settings(self, p: Sample):
        params = self.fixed_parameters.copy()
        params.update(dict(p.value_list()))
        return params

    def encoding_to_settings(self,output:List[float])->Dict[str,float]:
        params = self.fixed_parameters.copy()
        for p in self.decode(output):
            params[p.name] = p.value
        return params

    def decode(self,output:List[float])->List[ParamValue]:
        values = []
        for p in self.parameters:
            v, output = p.from_output(output)
            values.append(v)
        if len(output) > 0:
            print("Leftover output!: {}".format(output))
        return values

    def save(self, filename):
        with open(filename, 'wb') as file:
            dump(self, file)

    def save_json(self, filename):
        dump = self.to_json()
        with open(filename, 'w') as file:
            json.dump(dump, file, indent=2)

    def explain(self):
        levels = 0
        for p in self.parameters:
            levels += len(p.levels)
        return {
            "n_variable": len(self.parameters),
            "n_fixed": len(self.fixed_parameters),
            "levels": levels
        }

    def to_json(self):
        return {
            "parameters": [p.to_json() for p in self.parameters],
            "fixed": self.fixed_parameters
        }


"""
Generates evenly spaced parameter values
paper:
The rest of the synthesizer parameters ranges are quantized evenly to 16
classes according to the following ranges ...
For each parameter, the first and last classes correspond to its range limits
"""


def param_range(steps, min=0, max=1):
    ext = float(max - min)
    return [n * ext/(steps-1) + min for n in range(int(steps))]


"""
Generates a set of frequencies as per paper
paper: f = 2^(n/12)/ 440Hz with n in 0..15
"""


def freq_range(steps,base=440):
    return [math.pow(2, n/12) * base for n in range(int(steps))]

def switch_range(steps):
    size = 1 / steps
    return [n * size + (size/2) for n in range(int(steps))]

def replace_values(input,subs={}):
    # Return lists straight away
    if isinstance(input,list):
        return input
    # String we look at a direct sub, or calling a function
    if isinstance(input,str):
        if input in subs:
            return subs[input]
        m = regex.match('(\w+)(?:\s+([\d.]+))*', input)
        if m:
            function = m.captures(1)[0]
            print(f"got function: {function}")
            args = [float(d) for d in m.captures(2)]
            print(f"got args: {args}")
            if function == "steps":
                return param_range(*args)
            elif function == "semitones":
                return freq_range(*args)
            elif function == "switch":
                return switch_range(*args)
            else:
                print(f"Bad function: {function}")
        else:
            print(f"bad string: {input}")
            return input




def load_parameter(data,subs={}) -> Parameter:
    name = data['name']
    values = replace_values(data['values'],subs)
    id = data.get('id',"")
    return Parameter(name,values,id)

def load_parameter_set(parameters_file):
    print(f"Loading parameters from: {parameters_file}")
    with open(parameters_file, 'r') as f:
        config = json.load(f)
        #variable = [Parameter(p['name'],p['values'],p.get('id',"")) for p in config['parameters']]
        subs = config.get('substitutions', {})
        variable = [load_parameter(p,subs) for p in config['parameters']]
        fixed = dict([(p['name'],p['value']) for p in config['fixed_parameters']])
        ids = dict([(p['name'], p['id']) for p in config['fixed_parameters']])
        ids.update(dict([(p['name'], p['id']) for p in config['parameters']]))
        print(f"Fixed: {fixed}")
        parameters=ParameterSet(
            parameters = variable,
            fixed_parameters = fixed,
            ids = ids
        )
        print(f"Loaded parameter set: {parameters.explain()}")
        return parameters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Quickly read in a parameters file and output info')
    parser.add_argument('--file',  type=str,
                        help='Parameter file to parse')
    args = parser.parse_args()
    parameters = load_parameter_set(args.file)
    for p in parameters.parameters:
        print(f"[{len(p.levels)}]\t{p.name}")
        #print(f"[{p.levels}]\t{p.name}")
