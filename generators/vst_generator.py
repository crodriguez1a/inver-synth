from generators.generator import *
from generators.parameters import *

import numpy as np
import math
from scipy import stats

import sys
sys.path.append("/Users/dmrust//Uni/Work/Creative Computing/SynthLearn/RenderMan/Builds/MacOSX/build/Debug/")
import librenderman as rm

class VSTGenerator(SoundGenerator):
    def __init__(self, vst:str, sample_rate, randomise_non_set:bool=True, randomise_all:bool=False ):
        self.vst = vst
        self.randomise_non_set=randomise_non_set
        self.randomise_all=randomise_all
        self.sample_rate = sample_rate
        try:
            print("_____ LOADING VST _______")
            engine = rm.RenderEngine(sample_rate,512,512)
            if engine.load_plugin(self.vst):
                print("Loaded {}".format(self.vst))
                self.engine = engine
                self.patch_generator = rm.PatchGenerator(engine)
            else:
                print("Couldn't load VST {}".format(self.vst))
            print("_____ LOADED VST _______")
        except Exception as e:
            print("Problem: {}".format(e))

    #def do_sound_generation(self,parameter_set,base_filename)->np.ndarray:
    def generate(self,parameters:dict,filename:str,length:float,sample_rate:int,extra:dict={})->np.ndarray:
        if self.engine:
            if not self.sample_rate == sample_rate:
                print("Mismatched sample rate. got {}, asked for {}".format(self.sample_rate, sample_rate))
            engine = self.engine
            #print( engine.get_plugin_parameters_description() )
            print("Params to set:{}".format(parameters))

            if self.randomise_non_set:
                new_patch = self.patch_generator.get_random_patch()
                engine.set_patch(new_patch)

            synth_params = dict(engine.get_patch())
            # Start with defaults

            if not self.randomise_non_set:
                for i in range(155):
                    synth_params[i] = 0.5

            for name,value in parameters.items():
                synth_params[name] = value

            if self.randomise_all:
                new_patch = self.patch_generator.get_random_patch()
                engine.set_patch(new_patch)

            note_length = length * 0.8
            if 'note_length' in extra:
                note_length = extra['note_length']

            engine.set_patch(list(synth_params.items()))
            engine.render_patch(40,127,note_length,length)
            data = engine.get_audio_frames()
            print("Got {} frames as type {}".format(len(data),type(data)))
            nsamps = int(length*sample_rate)
            result = np.array(data[:nsamps])
            return result
        else:
            print("VST not loaded")
            return np.zeros(5)


if __name__ == "__main__":
    sample_rate = 16384
    gen = VSTGenerator(vst="/Library/Audio/Plug-Ins/VST/Dexed.vst", sample_rate = 16384)
    params_to_sample = [0,1,2,5,6,7,9]

    parameters=ParameterSet(
        parameters = [Parameter(i,[0.,0.5,1.0]) for i in params_to_sample],
        fixed_parameters = {
            #"a1":1.0,
            #"a2":1.0
        }
    )
    g = DatasetCreator("dexed_generator",
        dataset_dir="test_datasets",
        wave_file_dir="test_waves/dexed_set/",
        parameters=parameters
    )
    g.generate(sound_generator=gen,length=1,sample_rate=sample_rate,method="random",max=10,extra={'note_length':0.8})
