from generators.generator import *
import numpy as np
import math

class SineGenerator(SoundGenerator):
    def do_generate(self,parameters:dict,filename:str,length:float,sample_rate:int,extra:dict={})->np.ndarray:
        #print("Doing Sine!")
        samples = int(length * sample_rate)
        data = np.zeros(samples)
        params = dict(parameters)
        print("Sine Params: " + str(params))
        for i in range(samples):
            t = float(i) / sample_rate
            v = (
                    (params['a1']*math.sin(t*params['f1']*math.pi)) +
                    (params['a2']*math.sin(t*params['f2']*math.pi) )
                )*0.5
            data[i] = v
        return data



if __name__ == "__main__":
    gen = SineGenerator()
    parameters=ParameterSet(
        parameters = [
            Parameter("f1",[100,200,400]),
            Parameter("a1",[0.5,0.7,1.0]),
            Parameter("f2",[800,1200,1600]),
            Parameter("a2",[0.5,0.7,1.0])
        ],
        fixed_parameters = {
            "a1":1.0,
            "a2":1.0
        }
    )
    g = DatasetCreator("sine_generator",
        dataset_dir="test_datasets",
        wave_file_dir="test_waves/sine/",
        parameters=parameters
    )
    g.generate(sound_generator=gen,length=1,sample_rate=16384,method="random",max=10)
