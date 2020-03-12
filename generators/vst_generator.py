from generators.generator import Generator, Parameter
import numpy as np
import math
from scipy import stats

import sys
sys.path.append("/Users/dmrust//Uni/Work/Creative Computing/SynthLearn/RenderMan/Builds/MacOSX/build/Debug/")
import librenderman as rm

"""
  0: Cutoff
  1: Resonance
  2: Output
  3: MASTER TUNE ADJ
  4: ALGORITHM
  5: FEEDBACK
  6: OSC KEY SYNC
  7: LFO SPEED
  8: LFO DELAY
  9: LFO PM DEPTH
 10: LFO AM DEPTH
 11: LFO KEY SYNC
 12: LFO WAVE
 13: MIDDLE C
 14: P MODE SENS.
 15: PITCH EG RATE 1
 16: PITCH EG RATE 2
 17: PITCH EG RATE 3
 18: PITCH EG RATE 4
 19: PITCH EG LEVEL 1
 20: PITCH EG LEVEL 2
 21: PITCH EG LEVEL 3
 22: PITCH EG LEVEL 4
 23: OP1 EG RATE 1
 24: OP1 EG RATE 2
 25: OP1 EG RATE 3
 26: OP1 EG RATE 4
 27: OP1 EG LEVEL 1
 28: OP1 EG LEVEL 2
 29: OP1 EG LEVEL 3
 30: OP1 EG LEVEL 4
 31: OP1 OUTPUT LEVEL
 32: OP1 MODE
 33: OP1 F COARSE
 34: OP1 F FINE
 35: OP1 OSC DETUNE
 36: OP1 BREAK POINT
 37: OP1 L SCALE DEPTH
 38: OP1 R SCALE DEPTH
 39: OP1 L KEY SCALE
 40: OP1 R KEY SCALE
 41: OP1 RATE SCALING
 42: OP1 A MOD SENS.
 43: OP1 KEY VELOCITY
 44: OP1 SWITCH
 45: OP2 EG RATE 1
 46: OP2 EG RATE 2
 47: OP2 EG RATE 3
 48: OP2 EG RATE 4
 49: OP2 EG LEVEL 1
 50: OP2 EG LEVEL 2
 51: OP2 EG LEVEL 3
 52: OP2 EG LEVEL 4
 53: OP2 OUTPUT LEVEL
 54: OP2 MODE
 55: OP2 F COARSE
 56: OP2 F FINE
 57: OP2 OSC DETUNE
 58: OP2 BREAK POINT
 59: OP2 L SCALE DEPTH
 60: OP2 R SCALE DEPTH
 61: OP2 L KEY SCALE
 62: OP2 R KEY SCALE
 63: OP2 RATE SCALING
 64: OP2 A MOD SENS.
 65: OP2 KEY VELOCITY
 66: OP2 SWITCH
 67: OP3 EG RATE 1
 68: OP3 EG RATE 2
 69: OP3 EG RATE 3
 70: OP3 EG RATE 4
 71: OP3 EG LEVEL 1
 72: OP3 EG LEVEL 2
 73: OP3 EG LEVEL 3
 74: OP3 EG LEVEL 4
 75: OP3 OUTPUT LEVEL
 76: OP3 MODE
 77: OP3 F COARSE
 78: OP3 F FINE
 79: OP3 OSC DETUNE
 80: OP3 BREAK POINT
 81: OP3 L SCALE DEPTH
 82: OP3 R SCALE DEPTH
 83: OP3 L KEY SCALE
 84: OP3 R KEY SCALE
 85: OP3 RATE SCALING
 86: OP3 A MOD SENS.
 87: OP3 KEY VELOCITY
 88: OP3 SWITCH
 89: OP4 EG RATE 1
 90: OP4 EG RATE 2
 91: OP4 EG RATE 3
 92: OP4 EG RATE 4
 93: OP4 EG LEVEL 1
 94: OP4 EG LEVEL 2
 95: OP4 EG LEVEL 3
 96: OP4 EG LEVEL 4
 97: OP4 OUTPUT LEVEL
 98: OP4 MODE
 99: OP4 F COARSE
100: OP4 F FINE
101: OP4 OSC DETUNE
102: OP4 BREAK POINT
103: OP4 L SCALE DEPTH
104: OP4 R SCALE DEPTH
105: OP4 L KEY SCALE
106: OP4 R KEY SCALE
107: OP4 RATE SCALING
108: OP4 A MOD SENS.
109: OP4 KEY VELOCITY
110: OP4 SWITCH
111: OP5 EG RATE 1
112: OP5 EG RATE 2
113: OP5 EG RATE 3
114: OP5 EG RATE 4
115: OP5 EG LEVEL 1
116: OP5 EG LEVEL 2
117: OP5 EG LEVEL 3
118: OP5 EG LEVEL 4
119: OP5 OUTPUT LEVEL
120: OP5 MODE
121: OP5 F COARSE
122: OP5 F FINE
123: OP5 OSC DETUNE
124: OP5 BREAK POINT
125: OP5 L SCALE DEPTH
126: OP5 R SCALE DEPTH
127: OP5 L KEY SCALE
128: OP5 R KEY SCALE
129: OP5 RATE SCALING
130: OP5 A MOD SENS.
131: OP5 KEY VELOCITY
132: OP5 SWITCH
133: OP6 EG RATE 1
134: OP6 EG RATE 2
135: OP6 EG RATE 3
136: OP6 EG RATE 4
137: OP6 EG LEVEL 1
138: OP6 EG LEVEL 2
139: OP6 EG LEVEL 3
140: OP6 EG LEVEL 4
141: OP6 OUTPUT LEVEL
142: OP6 MODE
143: OP6 F COARSE
144: OP6 F FINE
145: OP6 OSC DETUNE
146: OP6 BREAK POINT
147: OP6 L SCALE DEPTH
148: OP6 R SCALE DEPTH
149: OP6 L KEY SCALE
150: OP6 R KEY SCALE
151: OP6 RATE SCALING
152: OP6 A MOD SENS.
153: OP6 KEY VELOCITY
154: OP6 SWITCH
"""



class VSTGenerator(Generator):
    def __init__(self,name: str, dataset_dir:str, wave_file_dir:str, parameters: list, length:1.0, sample_rate:44100,vst:str,note_length:float):
        super().__init__(name,dataset_dir,wave_file_dir,parameters,length,sample_rate)
        self.vst = vst
        self.note_length = note_length
        self.normalise = True
        self.randomise_non_set = True
        self.randomise_all = True
        try:
            print("_____ LOADING VST _______")
            engine = rm.RenderEngine(self.sample_rate,512,512)
            if engine.load_plugin(self.vst):
                print("Loaded {}".format(self.vst))
                #print( engine.get_plugin_parameters_description() )
                generator = rm.PatchGenerator(engine)
                self.engine = engine
                self.generator = generator
            else:
                print("Couldn't load VST {}".format(self.vst))
            print("_____ LOADED VST _______")
        except Exception as e:
            print("Problem: {}".format(e))

    def do_sound_generation(self,parameter_set,base_filename)->np.ndarray:
        print("Running VST")
        print("Made engine. Loading {}".format(self.vst))
        if self.engine:
            engine = self.engine
            print("Loaded {}".format(self.vst))
            #print( engine.get_plugin_parameters_description() )
            print("Params to set:{}".format(parameter_set))

            if self.randomise_non_set:
                new_patch = self.generator.get_random_patch()
                engine.set_patch(new_patch)

            synth_params = dict(engine.get_patch())
            # Start with defaults

            if not self.randomise_non_set:
                for i in range(155):
                    synth_params[i] = 0.5

            for name,value in parameter_set.items():
                synth_params[name] = value

            if self.randomise_all:
                new_patch = self.generator.get_random_patch()
                engine.set_patch(new_patch)

            engine.set_patch(list(synth_params.items()))
            engine.render_patch(40,127,self.note_length,self.length)
            data = engine.get_audio_frames()
            print("Got {} frames as type {}".format(len(data),type(data)))
            nsamps = int(self.length*self.sample_rate)
            result = np.array(data[:nsamps])
            if self.normalise:
                max = np.max(np.absolute(data))
                result = result / max
            #print(stats.describe(result))
            return result
        else:
            print("VST not loaded")
            return np.zeros(5)


if __name__ == "__main__":
    import random
    # Search the whole space
    params_to_sample = [0,1,2,5,6,7,9]
    #params_to_sample = range(155)
    parameters = [Parameter(i,[0.,0.5,1.0]) for i in params_to_sample]

    defaults = [
        Parameter(2,[1.0])
    ]
    parameters.extend(defaults)
    print(parameters)

    # Find some good bits of the space earlier?
    random.shuffle(parameters)

    # For tailored parameters, try it like this:
    # parameters = [ Parameter(0,[0., 0.5, 1.0]), Parameter(1,[0, 0.5, 1.0]) ]

    g = VSTGenerator(name="dexed_generator",dataset_dir="test_datasets",wave_file_dir="test_waves/dexed_samples",parameters=parameters,
        length=4.0, sample_rate=44100,
        vst="/Library/Audio/Plug-Ins/VST/Dexed.vst",
        #vst="/Library/Audio/Plug-Ins/VST/ValhallaFreqEcho.vst/Contents/MacOS/ValhallaFreqEcho",
        note_length=3.2)
    g.ramdomise_all = True;
    g.run()
