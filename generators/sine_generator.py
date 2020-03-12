from generators.generator import Generator, Parameter
import numpy as np
import math

class SineGenerator(Generator):
    def do_sound_generation(self,parameter_set,base_filename)->np.ndarray:
        print("Doing Sine!")
        length = int(self.length * self.sample_rate)
        data = np.zeros(length)
        params = dict(parameter_set)
        print("Params: " + str(params))
        for i in range(length):
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
