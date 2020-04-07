from typing import Generator, List, Sequence, Optional, Tuple, Iterator, Dict
import pickle

from tensorflow import keras

from scipy.io import wavfile
import numpy as np

from synthplayer import params as synth_params
from synthplayer.oscillators import *

from playing.reconstruction.curves import *
from playing.reconstruction.mod_oscillators import *


class FMResynth():
    def __init__(self,samplerate:int = 0) -> None:
        self._samplerate = samplerate or synth_params.norm_samplerate


    def synthesise(self,filename:str,curves:Dict[str,Oscillator],length:float=1.0):
        self._curves = curves
        lfo1 = ModSine(curves['v1'], amplitude=curves['B1'], samplerate=self._samplerate)
        s1 = ModSine(curves['f1'], amplitude=curves['A1'], fm_lfo=lfo1)

        lfo2 = ModSine(curves['v2'], amplitude=curves['B2'], samplerate=self._samplerate)
        s2 = ModSawtooth(curves['f2'], amplitude=curves['A2'], fm_lfo=lfo2)

        lfo3 = ModSine(curves['v3'], amplitude=curves['B3'], samplerate=self._samplerate)
        s3 = ModTriangle(curves['f3'], amplitude=curves['A3'], fm_lfo=lfo3)

        lfo4 = ModSine(curves['v4'], amplitude=curves['B4'], samplerate=self._samplerate)
        s4 = ModSquare(curves['f4'], amplitude=curves['A4'], fm_lfo=lfo4)

        y_osc = MixingFilter(s1, s2, s3, s4)

        n_samps = int(length * self._samplerate)
        data = []
        self._gen = y_osc.blocks()
        prog = Progress("Reconstructing audio from parameter curves", n_samps,char='+')
        while len(data) < n_samps:
            prog.update(len(data))
            data.extend(next(self._gen))
        prog.finish()

        #write_wav(filename, self._samplerate, np.array(data))
        wavfile.write(filename, self._samplerate, np.array(data))

    # Derive curves for each parameter from the model
    def create_model(self,model_file,parameters_file,input_file,window_size=16384,overlap=8) -> Dict[str,Oscillator]:
        jump = int(window_size / overlap )
        # Load in parameter definitions
        print(f"Loading parameters from {parameters_file}")
        parameters = pickle.load(open(parameters_file,"rb"))

        # Load in the model
        print(f"Loading model from {model_file}")
        model = keras.models.load_model( model_file )

        # Load the audio data
        print(f"Loading audio from {input_file}")
        fs, data = wavfile.read(input_file)
        if not fs == 16384:
            print("Warning! Wrong Sample Rate... Models all work with 16834 at the moment")
        print(f"Got {len(data)} samples at {fs}/s")

        # Set up the curves for all of the parameters
        curves = {}
        point = parameters.to_settings(parameters.sample_space(sample_size=1)[0])
        for k,v in point.items():
            curves[k] = LinearCurve([(0.0,v)])
        print(f"Setting up curves for {curves.keys()}")

        # Iterate through the data in blocks
        # Wow this is an ugly way to do this! Sorry! Late night airplane coding
        end = len(data)-window_size
        current = 0
        prog = Progress('Slicing input to parameter curves', max=end)
        while current < end:
            block = data[current:current+window_size]
            time = current / fs
            current += int(jump)
            prog.update(current)
            settings = self.model_slice(model,block,parameters)
            for k,v in settings.items():
                curves[k].append(time,v)
        prog.finish()
        return curves,len(data)/fs


    def model_slice(self,model,data,parameters)->Dict[str,float]:
        X = [data]
        Xd = np.expand_dims(np.vstack(X), axis=2)
        result = model.predict(Xd)[0]
        # Decode prediction, and reconstruct output
        predicted = parameters.encoding_to_settings(result)
        #print(f"Got predicted: {predicted}")
        return predicted

    def reconstruct(self,model_file,params_file,input_file,output_file):
        curves,length = self.create_model(model_file,params_file,input_file)
        self.synthesise(output_file,curves,length)


class Progress():
    def __init__(self,title,max=100,width=40,char="."):
        self._max = max
        self._char = char
        self._width = width
        self._printed = 0
        self._finished = False
        self._title = title
        print("[" + title.center(width) + "]")
        print("[",end="")

    def update(self,amount):
        target = int( amount / self._max * self._width )
        while self._printed < target:
            self._printed += 1
            print(self._char,end="",flush=True)
        if self._printed >= self._width:
            self.finish()

    def finish(self):
        if not self._finished:
            print("]")
        self._finished=True





if __name__ == "__main__":
    model_file = "output/inversynth_full_e2e_best.h5"
    params_file = "test_datasets/inversynth_tiny_params.pckl"
    resynth = FMResynth()
    for i in range(6):
        input = f"reconstruction_waves/example{i+1}.wav"
        output = f"reconstruction_waves/example{i+1}_recon.wav"
        resynth.reconstruct(model_file,params_file,input,output)


if __name__ == "__00main__":
    synth_params.norm_samplerate = 16384
    curves = {}
    for i in [
            'f1','A1','v1','B1',
            'f2','A2','v2','B2',
            'f3','A3','v3','B3',
            'f4','A4','v4','B4' ]:
        curves[i] = LinearCurve([(0.0, 0.5)])

    curves["f1"] = LinearCurve([(2.0, 200),(4.0, 800)])
    curves["A1"] = LinearCurve([(0.0, 0.5)])
    curves["v1"] = LinearCurve([(3.0, 100),(5.0, 400)])
    curves["B1"] = LinearCurve([(2.0, 0.0),(3.0,1.0)])
    curves["f2"] = LinearCurve([(2.0, 200),(4.0, 800)])
    curves["A2"] = LinearCurve([(0.0, 0.5)])
    curves["v2"] = LinearCurve([(3.0, 100),(5.0, 400)])
    curves["B2"] = LinearCurve([(2.0, 0.0),(3.0,1.0)])
    curves["f3"] = LinearCurve([(2.0, 200),(4.0, 800)])
    curves["A3"] = LinearCurve([(0.0, 0.5)])
    curves["v3"] = LinearCurve([(3.0, 100),(5.0, 400)])
    curves["B3"] = LinearCurve([(2.0, 0.0),(3.0,1.0)])
    curves["f4"] = LinearCurve([(2.0, 200),(4.0, 800)])
    curves["A4"] = LinearCurve([(0.0, 0.5)])
    curves["v4"] = LinearCurve([(3.0, 100),(5.0, 400)])
    curves["B4"] = LinearCurve([(2.0, 0.0),(3.0,1.0)])

    resynth = FMResynth()
    resynth.reconstruct("test_resynth_micro.wav",curves,5.0)
