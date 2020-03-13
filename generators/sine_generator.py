from generators.generator import Generator, Parameter
import numpy as np
import math

class SineGenerator(SoundGenerator):
    def generate(self,parameters:dict,filename:str,length:float,sample_rate:int)->np.ndarray:
        print("Doing Sine!")
        samples = int(length * sample_rate)
        data = np.zeros(samples)
        params = dict(parameter_set)
        print("Params: " + str(params))
        for i in range(samples):
            t = float(i) / self.sample_rate
            v = (math.sin(t*params['f1']*math.pi) + math.sin(t*params['f2']*math.pi))*0.5
            data[i] = v
        return data



if __name__ == "__main__":
    g = SineGenerator(name="test_generator2",dataset_dir="test_datasets",wave_file_dir="test_waves",parameters=[
        Parameter("f1",[100,110,120,130,140,150,170]),
        Parameter("f2",[200,220,240,260,280])
    ], length=1.0, sample_rate=44100)
    g.run()
